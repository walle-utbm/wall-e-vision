from __future__ import annotations

"""YOLO inference layer.

This module encapsulates model loading and raw prediction decoding. It exposes
normalized `Detection` objects used by the rest of the pipeline.

Supports both PyTorch (.pt) and NCNN (.param/.bin) formats.
NCNN is recommended for Raspberry Pi ARM inference (much faster, lower memory).
"""

import platform
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List

import cv2
import numpy as np
import yaml


def _prefer_system_site_packages() -> None:
    """Make Raspberry Pi OS packages available before virtualenv packages on ARM64."""
    if platform.machine().lower() not in {"aarch64", "arm64"}:
        return

    import sys

    candidates = [
        Path("/usr/lib/python3/dist-packages"),
        Path("/usr/lib/python3.13/dist-packages"),
        Path("/usr/local/lib/python3.13/dist-packages"),
    ]
    for path in reversed(candidates):
        if path.exists():
            path_str = str(path)
            if path_str in sys.path:
                sys.path.remove(path_str)
            sys.path.insert(0, path_str)


_prefer_system_site_packages()

import torch
import ncnn
from ultralytics import YOLO

from .sorting import classify_recycle_bin
from .types import Detection


@dataclass(slots=True)
class DetectorConfig:
    """Inference parameters for `WasteDetector`.

    Optimized for RubikPi 3 (Snapdragon 8 Gen1) with PyTorch inference.
    No NCNN needed—CPU is powerful enough.

    Attributes:
        model_path: Path to .pt model file (PyTorch format).
        conf_threshold: Minimum confidence score kept after inference (0.0-1.0).
        iou_threshold: NMS IoU threshold for duplicate removal (0.0-1.0).
        image_size: Inference input size (768 for RubikPi, 640 for RPi4).
        max_detections: Maximum detections returned per frame.
        use_half: Request FP16 when supported by CPU.
        device: Device to use ('cpu' for RubikPi).
        workers: DataLoader workers (0 to avoid spawn issues on ARM).
    """

    model_path: str
    conf_threshold: float = 0.30
    iou_threshold: float = 0.45
    image_size: int = 640
    max_detections: int = 8
    use_half: bool = True
    device: str = "cpu"
    workers: int = 0
    box_scale: float = 1.0


class WasteDetector:
    """High-level detector that converts YOLO results to project `Detection` objects.
    
    Optimized for RubikPi 3 (Snapdragon 8 Gen1) with PyTorch on powerful ARM CPU.
    """

    def __init__(self, cfg: DetectorConfig) -> None:
        """Load PyTorch YOLO model once and keep in memory for real-time use."""
        self.cfg = cfg
        model_path = Path(cfg.model_path)
        self._arm64 = platform.machine().lower() in {"aarch64", "arm64"}
        self.class_names: dict[int, str] = {}

        if self._arm64:
            try:
                torch.backends.mkldnn.enabled = False
            except Exception:
                pass

        # Load PyTorch model
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        print(f"Loading PyTorch model: {model_path}")
        self.model = YOLO(str(model_path))
        self.model_format = "pt"

        try:
            # Fuse layers to reduce inference overhead on CPU
            self.model.fuse()
        except Exception:
            pass

        self.class_names = {int(k): str(v) for k, v in self.model.names.items()}

    def _load_ncnn_model(self, ncnn_param: Path) -> None:
        """Load a native NCNN model for ARM-friendly inference."""
        bin_path = ncnn_param.with_suffix(".bin")
        if not bin_path.exists():
            raise FileNotFoundError(f"NCNN weights not found: {bin_path}")

        print(f"Loading NCNN model: {ncnn_param}")
        self._ncnn_net = ncnn.Net()
        self._ncnn_net.opt.use_vulkan_compute = False
        try:
            self._ncnn_net.opt.num_threads = max(1, (os.cpu_count() or 4) - 1)
        except Exception:
            pass
        self._ncnn_net.load_param(str(ncnn_param))
        self._ncnn_net.load_model(str(bin_path))
        self.class_names = self._load_ncnn_class_names(ncnn_param.parent / "metadata.yaml")

    def _load_ncnn_class_names(self, metadata_path: Path) -> dict[int, str]:
        """Load class names from NCNN export metadata."""
        if not metadata_path.exists():
            return {}

        try:
            with metadata_path.open("r", encoding="utf-8") as file:
                metadata = yaml.safe_load(file) or {}
        except Exception:
            return {}

        names = metadata.get("names", {})
        if isinstance(names, list):
            return {index: str(name) for index, name in enumerate(names)}
        if isinstance(names, dict):
            return {int(index): str(name) for index, name in names.items()}
        return {}

    def _infer_ncnn(self, frame: np.ndarray) -> List[Detection]:
        """Run inference with the native NCNN runtime and decode YOLO outputs."""
        if self._ncnn_net is None:
            raise RuntimeError("NCNN model is not loaded")

        height, width = frame.shape[:2]
        target_size = int(self.cfg.image_size)

        input_mat = ncnn.Mat.from_pixels_resize(
            frame,
            ncnn.Mat.PixelType.PIXEL_RGB,
            width,
            height,
            target_size,
            target_size,
        )
        input_mat.substract_mean_normalize((0.0, 0.0, 0.0), (1 / 255.0, 1 / 255.0, 1 / 255.0))

        with self._ncnn_net.create_extractor() as extractor:
            extractor.input("in0", input_mat)
            _, output = extractor.extract("out0")

        predictions = np.array(output, copy=True).T
        if predictions.ndim != 2 or predictions.shape[1] < 5:
            return []

        boxes = predictions[:, :4].astype(np.float32, copy=False)
        scores = predictions[:, 4:].astype(np.float32, copy=False)

        if boxes.size == 0 or scores.size == 0:
            return []

        # Some exports emit normalized coordinates; others emit pixel-space boxes.
        if float(np.max(boxes)) <= 2.0:
            boxes = boxes.copy()
            boxes[:, [0, 2]] *= width
            boxes[:, [1, 3]] *= height

        # Post-process scores: some NCNN exports include an objectness score followed by class logits,
        # others output class logits only. Apply sigmoid and combine objectness*class_probs when present.
        def sigmoid(x: np.ndarray) -> np.ndarray:
            return 1.0 / (1.0 + np.exp(-x))

        num_score_cols = scores.shape[1]
        num_classes = max(len(self.class_names), 0)

        # Quick heuristic: if scores already lie in [0,1], treat them as probabilities
        scores_min = float(np.nanmin(scores)) if scores.size > 0 else 0.0
        scores_max = float(np.nanmax(scores)) if scores.size > 0 else 0.0
        is_prob = (scores_min >= 0.0) and (scores_max <= 1.0)
        if is_prob:
            # scores are already probabilities
            if num_score_cols == num_classes + 1 and num_classes > 0:
                obj_prob = scores[:, 0:1]
                class_prob = scores[:, 1:]
                final_scores = class_prob * obj_prob
            else:
                final_scores = scores
        else:
            # format may be logits: apply sigmoid then combine objectness if present
            if num_score_cols == num_classes + 1 and num_classes > 0:
                # format: [obj, class0, class1, ...]
                obj_logits = scores[:, 0:1]
                class_logits = scores[:, 1:]
                obj_prob = sigmoid(obj_logits)
                class_prob = sigmoid(class_logits)
                final_scores = class_prob * obj_prob
            else:
                # format: [class0, class1, ...] (no objectness); use sigmoid per-class
                final_scores = sigmoid(scores)
        try:
            print(f"🔎 NCNN raw scores range: min={scores_min:.6g}, max={scores_max:.6g}, treated_as_prob={is_prob}")
        except Exception:
            pass

        class_ids = final_scores.argmax(axis=1).astype(np.int32, copy=False)
        confidences = final_scores.max(axis=1).astype(np.float32, copy=False)
        keep_mask = confidences >= self.cfg.conf_threshold
        if not np.any(keep_mask):
            # debug: show top raw confidences to help debugging
            try:
                topk = np.sort(confidences)[-5:][::-1]
                print(f"⚠️ NCNN: no detections above conf {self.cfg.conf_threshold}; top confidences: {topk}")
            except Exception:
                pass
            return []

        boxes = boxes[keep_mask]
        class_ids = class_ids[keep_mask]
        confidences = confidences[keep_mask]

        # Optional box scale to slightly tighten predicted boxes (useful for noisy exports)
        if getattr(self.cfg, "box_scale", 1.0) != 1.0:
            boxes = boxes.copy()
            boxes[:, 2] = boxes[:, 2] * float(self.cfg.box_scale)
            boxes[:, 3] = boxes[:, 3] * float(self.cfg.box_scale)

        xyxy_boxes = np.empty_like(boxes, dtype=np.float32)
        xyxy_boxes[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.0
        xyxy_boxes[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.0
        xyxy_boxes[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.0
        xyxy_boxes[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.0

        keep_indices = self._apply_nms(xyxy_boxes, confidences, class_ids, self.cfg.iou_threshold)
        if not keep_indices:
            return []

        keep_indices = keep_indices[: self.cfg.max_detections]

        detections: List[Detection] = []
        frame_area = float(width * height)
        for index in keep_indices:
            x1, y1, x2, y2 = xyxy_boxes[index]
            x1_i = max(0, min(width - 1, int(x1)))
            y1_i = max(0, min(height - 1, int(y1)))
            x2_i = max(0, min(width - 1, int(x2)))
            y2_i = max(0, min(height - 1, int(y2)))

            if x2_i <= x1_i or y2_i <= y1_i:
                continue

            bbox_clipped = x1_i == 0 or y1_i == 0 or x2_i == width - 1 or y2_i == height - 1
            box_width = max(1, x2_i - x1_i)
            box_height = max(1, y2_i - y1_i)
            center = (x1_i + box_width // 2, y1_i + box_height // 2)
            area_ratio = (box_width * box_height) / frame_area
            class_id = int(class_ids[index])
            class_name = self._class_name(class_id)

            detections.append(
                Detection(
                    class_id=class_id,
                    class_name=class_name,
                    recycle_bin=classify_recycle_bin(class_name),
                    confidence=float(confidences[index]),
                    bbox=(x1_i, y1_i, x2_i, y2_i),
                    center=center,
                    pickup_point=center,
                    area_ratio=area_ratio,
                    mask_area_ratio=0.0,
                    segmentation_available=False,
                    bbox_clipped=bbox_clipped,
                )
            )

        return detections

    def _class_name(self, class_id: int) -> str:
        """Resolve a class name from metadata or fall back to the numeric id."""
        return self.class_names.get(class_id, str(class_id))

    def _apply_nms(
        self,
        boxes: np.ndarray,
        confidences: np.ndarray,
        class_ids: np.ndarray,
        iou_threshold: float,
    ) -> list[int]:
        """Apply class-aware greedy NMS to decoded detections."""
        kept_indices: list[int] = []
        for class_id in np.unique(class_ids):
            class_indices = np.where(class_ids == class_id)[0]
            if class_indices.size == 0:
                continue

            order = class_indices[np.argsort(-confidences[class_indices])]
            while order.size > 0:
                current = int(order[0])
                kept_indices.append(current)
                if order.size == 1:
                    break

                remaining = order[1:]
                ious = self._iou(boxes[current], boxes[remaining])
                order = remaining[ious <= iou_threshold]

        kept_indices.sort(key=lambda index: float(confidences[index]), reverse=True)
        return kept_indices

    def _iou(self, box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        """Compute IoU between one box and a batch of boxes."""
        if boxes.size == 0:
            return np.empty((0,), dtype=np.float32)

        x1 = np.maximum(box[0], boxes[:, 0])
        y1 = np.maximum(box[1], boxes[:, 1])
        x2 = np.minimum(box[2], boxes[:, 2])
        y2 = np.minimum(box[3], boxes[:, 3])

        inter_w = np.maximum(0.0, x2 - x1)
        inter_h = np.maximum(0.0, y2 - y1)
        intersection = inter_w * inter_h

        box_area = max(0.0, float(box[2] - box[0])) * max(0.0, float(box[3] - box[1]))
        boxes_area = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0.0, boxes[:, 3] - boxes[:, 1])
        union = box_area + boxes_area - intersection
        return np.where(union > 0.0, intersection / union, 0.0)

    def _infer_pt(self, frame: np.ndarray) -> List[Detection]:
        """Run inference with the Ultralytics PyTorch backend."""
        use_half = self.cfg.use_half and self.cfg.device not in {"cpu", "mps"}
        results = self.model.predict(
            source=frame,
            conf=self.cfg.conf_threshold,
            iou=self.cfg.iou_threshold,
            imgsz=self.cfg.image_size,
            max_det=self.cfg.max_detections,
            half=use_half,
            device=self.cfg.device,
            verbose=False,
            workers=self.cfg.workers,
        )

        if not results:
            print("⚠️ No results from model.predict()")
            return []

        result = results[0]
        boxes = result.boxes
        if boxes is None or boxes.xyxy is None:
            print(f"⚠️ No boxes detected (boxes: {boxes})")
            return []
        
        if len(boxes) > 0:
            print(f"✅ {len(boxes)} detections found (conf threshold: {self.cfg.conf_threshold})")
            for i, conf in enumerate(boxes.conf):
                print(f"   - Detection {i}: conf={float(conf.item()):.3f}")

        masks_xy = result.masks.xy if result.masks is not None and result.masks.xy is not None else []

        detections: List[Detection] = []
        frame_h, frame_w = frame.shape[:2]
        frame_area = float(frame_h * frame_w)

        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes.xyxy[i].tolist()
            conf = float(boxes.conf[i].item())
            class_id = int(boxes.cls[i].item())
            class_name = result.names.get(class_id, str(class_id))

            x1_i = max(0, int(x1))
            y1_i = max(0, int(y1))
            x2_i = min(frame_w - 1, int(x2))
            y2_i = min(frame_h - 1, int(y2))

            bbox_clipped = x1_i == 0 or y1_i == 0 or x2_i == frame_w - 1 or y2_i == frame_h - 1

            width = max(1, x2_i - x1_i)
            height = max(1, y2_i - y1_i)
            area_ratio = (width * height) / frame_area
            center = (x1_i + width // 2, y1_i + height // 2)

            pickup_point = center
            segmentation_available = False
            mask_area_ratio = 0.0

            if i < len(masks_xy):
                polygon = np.asarray(masks_xy[i], dtype=np.float32)
                if polygon.size >= 6:
                    mask_area = float(abs(cv2.contourArea(polygon)))
                    mask_area_ratio = mask_area / frame_area

                    moments = cv2.moments(polygon)
                    if moments["m00"] != 0:
                        cx = int(moments["m10"] / moments["m00"])
                        cy = int(moments["m01"] / moments["m00"])
                        pickup_point = (max(0, min(frame_w - 1, cx)), max(0, min(frame_h - 1, cy)))
                        segmentation_available = True

            detections.append(
                Detection(
                    class_id=class_id,
                    class_name=class_name,
                    recycle_bin=classify_recycle_bin(class_name),
                    confidence=conf,
                    bbox=(x1_i, y1_i, x2_i, y2_i),
                    center=center,
                    pickup_point=pickup_point,
                    area_ratio=area_ratio,
                    mask_area_ratio=mask_area_ratio,
                    segmentation_available=segmentation_available,
                    bbox_clipped=bbox_clipped,
                )
            )

        return detections

    def infer(self, frame: np.ndarray) -> List[Detection]:
        """Run inference on one frame and return decoded detections."""
        if self.model_format == "ncnn":
            return self._infer_ncnn(frame)
        return self._infer_pt(frame)
    def _find_ncnn_model(self, model_path: Path) -> Path | None:
        """Look for NCNN model (.ncnn.param) in the same directory or in a subdirectory.
        
        Args:
            model_path: Path to .pt model (e.g., model/best.pt)
            
        Returns:
            Path to .ncnn.param file if found, None otherwise.
        """
        if not model_path.exists():
            return None
            
        # Check if NCNN files exist in same directory
        stem = model_path.stem  # "best" from "best.pt"
        parent = model_path.parent
        
        # Try direct path: model/best.ncnn.param
        ncnn_param = parent / f"{stem}.ncnn.param"
        ncnn_bin = parent / f"{stem}.ncnn.bin"
        
        if ncnn_param.exists() and ncnn_bin.exists():
            return ncnn_param
        
        # Try subdirectory path: model/best_ncnn_model/model.ncnn.param
        # (created by ultralytics export)
        ncnn_subdir = parent / f"{stem}_ncnn_model"
        ncnn_param_sub = ncnn_subdir / "model.ncnn.param"
        ncnn_bin_sub = ncnn_subdir / "model.ncnn.bin"
        
        if ncnn_param_sub.exists() and ncnn_bin_sub.exists():
            return ncnn_param_sub
        
        return None

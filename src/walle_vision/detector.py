from __future__ import annotations

"""PyTorch inference layer for RubikPi.

This module loads the YOLO model and converts raw predictions into the
project's normalized `Detection` objects.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from .sorting import classify_recycle_bin
from .types import Detection


@dataclass(slots=True)
class DetectorConfig:
    """Inference parameters for `WasteDetector`."""

    model_path: str
    conf_threshold: float = 0.30
    iou_threshold: float = 0.45
    image_size: int = 640
    max_detections: int = 8
    use_half: bool = False
    device: str = "cpu"
    workers: int = 2
    box_scale: float = 1.0


class WasteDetector:
    """High-level detector that converts YOLO results to `Detection` objects."""

    def __init__(self, cfg: DetectorConfig) -> None:
        self.cfg = cfg
        model_path = Path(cfg.model_path)
        self.class_names: dict[int, str] = {}

        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        self.model = YOLO(str(model_path))

        try:
            self.model.fuse()
        except Exception:
            pass

        self.class_names = {int(key): str(value) for key, value in self.model.names.items()}

    def infer(self, frame: np.ndarray) -> List[Detection]:
        """Run inference on one frame and return decoded detections."""
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
            return []

        result = results[0]
        boxes = result.boxes
        if boxes is None or boxes.xyxy is None:
            return []

        masks_xy = result.masks.xy if result.masks is not None and result.masks.xy is not None else []

        detections: List[Detection] = []
        frame_height, frame_width = frame.shape[:2]
        frame_area = float(frame_height * frame_width)

        for index in range(len(boxes)):
            x1, y1, x2, y2 = boxes.xyxy[index].tolist()
            confidence = float(boxes.conf[index].item())
            class_id = int(boxes.cls[index].item())
            class_name = result.names.get(class_id, str(class_id))

            x1_i = max(0, int(x1))
            y1_i = max(0, int(y1))
            x2_i = min(frame_width - 1, int(x2))
            y2_i = min(frame_height - 1, int(y2))

            bbox_clipped = x1_i == 0 or y1_i == 0 or x2_i == frame_width - 1 or y2_i == frame_height - 1

            box_width = max(1, x2_i - x1_i)
            box_height = max(1, y2_i - y1_i)
            area_ratio = (box_width * box_height) / frame_area
            center = (x1_i + box_width // 2, y1_i + box_height // 2)

            pickup_point = center
            segmentation_available = False
            mask_area_ratio = 0.0

            if index < len(masks_xy):
                polygon = np.asarray(masks_xy[index], dtype=np.float32)
                if polygon.size >= 6:
                    mask_area = float(abs(cv2.contourArea(polygon)))
                    mask_area_ratio = mask_area / frame_area

                    moments = cv2.moments(polygon)
                    if moments["m00"] != 0:
                        center_x = int(moments["m10"] / moments["m00"])
                        center_y = int(moments["m01"] / moments["m00"])
                        pickup_point = (
                            max(0, min(frame_width - 1, center_x)),
                            max(0, min(frame_height - 1, center_y)),
                        )
                        segmentation_available = True

            detections.append(
                Detection(
                    class_id=class_id,
                    class_name=class_name,
                    recycle_bin=classify_recycle_bin(class_name),
                    confidence=confidence,
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

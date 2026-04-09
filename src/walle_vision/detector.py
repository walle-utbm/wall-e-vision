from __future__ import annotations

"""YOLO inference layer.

This module encapsulates model loading and raw prediction decoding. It exposes
normalized `Detection` objects used by the rest of the pipeline.
"""

from dataclasses import dataclass
from typing import List

import cv2
import numpy as np
from ultralytics import YOLO

from .sorting import classify_recycle_bin
from .types import Detection


@dataclass(slots=True)
class DetectorConfig:
    """Inference parameters for `WasteDetector`.

    Attributes:
        model_path: Path to YOLO weights file (.pt).
        conf_threshold: Minimum confidence score kept after inference.
        iou_threshold: NMS IoU threshold.
        image_size: Inference input size (typically 640 for your model).
        max_detections: Maximum detections returned per frame.
        use_half: Request FP16 when supported by hardware/runtime.
    """

    model_path: str
    conf_threshold: float = 0.25
    iou_threshold: float = 0.45
    image_size: int = 640
    max_detections: int = 10
    use_half: bool = False


class WasteDetector:
    """High-level detector that converts YOLO results to project `Detection` objects."""

    def __init__(self, cfg: DetectorConfig) -> None:
        """Load YOLO model once and keep it in memory for real-time use."""
        self.cfg = cfg
        self.model = YOLO(cfg.model_path)
        # fuse() can slightly reduce inference overhead on CPU/GPU.
        try:
            self.model.fuse()
        except Exception:
            pass

    def infer(self, frame: np.ndarray) -> List[Detection]:
        """Run inference on one frame and return decoded detections.

        If segmentation masks are available, pickup point uses mask centroid.
        Otherwise, pickup point falls back to bounding-box center.
        """
        results = self.model.predict(
            source=frame,
            conf=self.cfg.conf_threshold,
            iou=self.cfg.iou_threshold,
            imgsz=self.cfg.image_size,
            max_det=self.cfg.max_detections,
            half=self.cfg.use_half,
            device="cpu",
            verbose=False,
        )

        if not results:
            return []

        result = results[0]
        boxes = result.boxes
        if boxes is None or boxes.xyxy is None:
            return []

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

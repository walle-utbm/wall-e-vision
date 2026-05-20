from __future__ import annotations

"""Detector wrapper for PC-side inference with a graceful fallback.

This module provides `DetectorConfig` and `WasteDetector`. If
`ultralytics` is available the detector will run YOLO inference; otherwise
`infer()` returns an empty list so the server can run for testing.
"""

from dataclasses import dataclass
from typing import List

import cv2
import numpy as np

from .types import Detection


@dataclass(slots=True)
class DetectorConfig:
    model_path: str = "model/best.pt"
    conf_threshold: float = 0.1
    iou_threshold: float = 0.45
    image_size: int = 640
    max_detections: int = 8
    use_half: bool = True
    device: str = "cpu"


class WasteDetector:
    """Wraps an object detector. If Ultralytics is unavailable this is a
    no-op detector that returns no detections (useful for testing the
    server without heavy dependencies).
    """

    def __init__(self, cfg: DetectorConfig) -> None:
        self.cfg = cfg
        self.model = None
        self.names = {}

        try:
            from ultralytics import YOLO  # type: ignore

            self.model = YOLO(cfg.model_path)
            # populate names if model exposes them
            try:
                self.names = getattr(self.model, "names", {}) or {}
            except Exception:
                self.names = {}
        except Exception:
            print("ultralytics not available — detector will be a no-op")

    def infer(self, frame: np.ndarray) -> List[Detection]:
        """Run inference on a single BGR `frame` and return list of
        `Detection`. When the real model is absent this returns an empty
        list.
        """
        if self.model is None:
            return []

        # Convert BGR->RGB for the model if needed
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Run model (ultralytics API may accept numpy arrays directly)
        try:
            results = self.model.predict(
                source=img,
                imgsz=self.cfg.image_size,
                conf=self.cfg.conf_threshold,
                iou=self.cfg.iou_threshold,
                max_det=self.cfg.max_detections,
                device=self.cfg.device,
            )
        except TypeError:
            # older ultralytics versions may have different kwargs
            results = self.model(img)

        detections: List[Detection] = []

        # results may be a list-like container; handle common shapes
        seq = results if isinstance(results, (list, tuple)) else [results]
        for res in seq:
            try:
                boxes = getattr(res, "boxes", None)
                if boxes is None:
                    continue

                xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes.xyxy, "cpu") else np.asarray(boxes.xyxy)
                confs = boxes.conf.cpu().numpy() if hasattr(boxes.conf, "cpu") else np.asarray(boxes.conf)
                classes = boxes.cls.cpu().numpy().astype(int) if hasattr(boxes.cls, "cpu") else np.asarray(boxes.cls).astype(int)

                for bb, conf, cls in zip(xyxy, confs, classes):
                    x1, y1, x2, y2 = map(int, bb[:4])
                    bbox = (x1, y1, x2, y2)
                    center = ((x1 + x2) // 2, (y1 + y2) // 2)
                    pickup = center
                    class_name = str(self.names.get(int(cls), str(int(cls))))

                    det = Detection(
                        class_id=int(cls),
                        class_name=class_name,
                        recycle_bin="unknown",
                        confidence=float(conf),
                        bbox=bbox,
                        center=center,
                        pickup_point=pickup,
                        area_ratio=0.0,
                        bbox_clipped=False,
                    )
                    detections.append(det)
            except Exception:
                continue

        return detections

from __future__ import annotations

"""Rendering helpers for live visualization."""

import cv2
import numpy as np

from ..types import Detection


BIN_COLORS = {
    "yellow": (0, 215, 255),
    "glass": (0, 180, 255),
    "other": (80, 80, 80),
}


def draw_detections(frame: np.ndarray, detections: list[Detection]) -> np.ndarray:
    annotated = frame.copy()
    for detection in detections:
        color = BIN_COLORS.get(detection.recycle_bin, BIN_COLORS["other"])
        x1, y1, x2, y2 = detection.bbox
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        cv2.circle(annotated, detection.pickup_point, 4, color, -1)

        if detection.track_id is not None:
            label = f"#{detection.track_id} {detection.class_name} | {detection.recycle_bin} | {detection.confidence:.2f}"
        else:
            label = f"{detection.class_name} | {detection.recycle_bin} | {detection.confidence:.2f}"
        y_text = max(20, y1 - 8)
        cv2.putText(annotated, label, (x1, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

    return annotated
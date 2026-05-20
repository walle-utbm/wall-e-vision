from __future__ import annotations

"""Rendering helpers for debug/live visualization on the PC."""

import cv2
import numpy as np

from .types import Detection

BIN_COLORS = {
    "yellow": (0, 215, 255),
    "glass": (0, 180, 255),
    "other": (80, 80, 80),
}


def draw_detections(frame: np.ndarray, detections: list[Detection]) -> np.ndarray:
    annotated = frame.copy()
    for det in detections:
        color = BIN_COLORS.get(det.recycle_bin, BIN_COLORS["other"])
        x1, y1, x2, y2 = det.bbox
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        cv2.circle(annotated, det.pickup_point, 4, color, -1)

        if det.track_id is not None:
            label = f"#{det.track_id} {det.class_name} | {det.recycle_bin} | {det.confidence:.2f}"
        else:
            label = f"{det.class_name} | {det.recycle_bin} | {det.confidence:.2f}"
        y_text = max(20, y1 - 8)
        cv2.putText(
            annotated,
            label,
            (x1, y_text),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )

    return annotated
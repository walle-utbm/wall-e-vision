from __future__ import annotations

"""Shared data structures used by the vision pipeline.

This module centralizes strongly-typed containers exchanged between detector,
tracker, visualizer, and JSON export. Keeping these types in one place makes
the code easier to read and less error-prone when adding new fields.
"""

from dataclasses import dataclass
from typing import List, Tuple


BBox = Tuple[int, int, int, int]
Center = Tuple[int, int]


@dataclass(slots=True)
class Detection:
    """Single object detection produced by the detector/tracker pipeline.

    Attributes:
        class_id: Numeric class index predicted by YOLO.
        class_name: Human-readable class label.
        recycle_bin: Business category used by sorting logic.
        confidence: Confidence score for the selected class.
        bbox: Bounding box coordinates as (x1, y1, x2, y2).
        center: Geometric center of the bounding box.
        pickup_point: Target point to send to the robot gripper.
        area_ratio: Bounding-box area divided by full image area.
        bbox_clipped: True when box touches image border.
        mask_area_ratio: Segmentation area ratio, 0.0 when unavailable.
        segmentation_available: True when mask-based pickup point is used.
        track_id: Persistent ID assigned by temporal tracker.
        track_confirmed: True when track reached confirmation threshold.
        track_hits: Number of successful frame-to-frame associations.
        track_missed_frames: Number of recent missed associations.
        track_score: Smoothed confidence across a short temporal window.
    """

    class_id: int
    class_name: str
    recycle_bin: str
    confidence: float
    bbox: BBox
    center: Center
    pickup_point: Center
    area_ratio: float
    bbox_clipped: bool
    mask_area_ratio: float = 0.0
    segmentation_available: bool = False
    track_id: int | None = None
    track_confirmed: bool = False
    track_hits: int = 0
    track_missed_frames: int = 0
    track_score: float = 0.0


@dataclass(slots=True)
class FrameResult:
    """Result payload for one processed frame.

    Attributes:
        frame_index: Monotonic frame counter from capture stream.
        timestamp: Unix timestamp of frame acquisition.
        detections: Stable detections after temporal filtering.
    """

    frame_index: int
    timestamp: float
    detections: List[Detection]

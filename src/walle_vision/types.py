from __future__ import annotations

"""Shared data structures used by the unified pipeline."""

from dataclasses import dataclass
from typing import List, Tuple


BBox = Tuple[int, int, int, int]
Center = Tuple[int, int]


@dataclass(slots=True)
class Detection:
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
    frame_index: int
    timestamp: float
    detections: List[Detection]
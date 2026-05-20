from __future__ import annotations

"""Temporal tracking and smoothing for frame-to-frame detection stability.

The tracker associates detections by class + IoU, confirms tracks after a
minimum number of hits, and smooths confidence values over a short window.
"""

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, List

from .types import BBox, Detection


def _bbox_iou(left: BBox, right: BBox) -> float:
    """Compute Intersection-over-Union between two bounding boxes."""
    left_x1, left_y1, left_x2, left_y2 = left
    right_x1, right_y1, right_x2, right_y2 = right

    intersection_x1 = max(left_x1, right_x1)
    intersection_y1 = max(left_y1, right_y1)
    intersection_x2 = min(left_x2, right_x2)
    intersection_y2 = min(left_y2, right_y2)

    if intersection_x2 <= intersection_x1 or intersection_y2 <= intersection_y1:
        return 0.0

    intersection_area = float((intersection_x2 - intersection_x1) * (intersection_y2 - intersection_y1))
    left_area = float(max(1, left_x2 - left_x1) * max(1, left_y2 - left_y1))
    right_area = float(max(1, right_x2 - right_x1) * max(1, right_y2 - right_y1))
    union_area = left_area + right_area - intersection_area
    if union_area <= 0.0:
        return 0.0
    return intersection_area / union_area


@dataclass(slots=True)
class TrackState:
    """Internal state for one tracked object across successive frames."""
    track_id: int
    last_detection: Detection
    hits: int = 1
    missed_frames: int = 0
    confidence_history: Deque[float] = field(default_factory=deque)

    def __post_init__(self) -> None:
        if not self.confidence_history:
            self.confidence_history.append(self.last_detection.confidence)

    @property
    def confirmed(self) -> bool:
        return self.hits >= 1

    def smoothed_confidence(self, window_size: int) -> float:
        if window_size <= 1 or not self.confidence_history:
            return self.last_detection.confidence
        history = list(self.confidence_history)[-window_size:]
        return sum(history) / len(history)


class TemporalDetectionTracker:
    """Lightweight tracker used to reduce flicker and false-positive spikes."""

    def __init__(self, iou_threshold: float = 0.3, confirm_frames: int = 3, max_missed_frames: int = 2, confidence_window: int = 5) -> None:
        """Initialize temporal association and confirmation thresholds."""
        self.iou_threshold = iou_threshold
        self.confirm_frames = max(1, confirm_frames)
        self.max_missed_frames = max(0, max_missed_frames)
        self.confidence_window = max(1, confidence_window)
        self._next_track_id = 1
        self._tracks: list[TrackState] = []

    def update(self, detections: list[Detection]) -> list[Detection]:
        """Associate current detections to tracks and return confirmed detections only."""
        if not self._tracks:
            self._tracks = [self._create_track(det) for det in detections]
            return self._confirmed_detections()

        matched_track_ids: set[int] = set()
        matched_detection_indices: set[int] = set()

        scored_pairs: list[tuple[float, int, int]] = []
        for track_index, track in enumerate(self._tracks):
            if track.missed_frames > self.max_missed_frames:
                continue
            for detection_index, detection in enumerate(detections):
                if detection.class_name != track.last_detection.class_name:
                    continue
                score = _bbox_iou(track.last_detection.bbox, detection.bbox)
                if score >= self.iou_threshold:
                    scored_pairs.append((score, track_index, detection_index))

        scored_pairs.sort(reverse=True, key=lambda item: item[0])

        for _, track_index, detection_index in scored_pairs:
            track = self._tracks[track_index]
            if track.track_id in matched_track_ids or detection_index in matched_detection_indices:
                continue
            detection = detections[detection_index]
            self._update_track(track, detection)
            matched_track_ids.add(track.track_id)
            matched_detection_indices.add(detection_index)

        for track in self._tracks:
            if track.track_id not in matched_track_ids:
                track.missed_frames += 1

        for detection_index, detection in enumerate(detections):
            if detection_index in matched_detection_indices:
                continue
            self._tracks.append(self._create_track(detection))

        self._tracks = [track for track in self._tracks if track.missed_frames <= self.max_missed_frames]
        return self._confirmed_detections()

    def _create_track(self, detection: Detection) -> TrackState:
        """Create a new track for an unmatched detection."""
        track = TrackState(track_id=self._next_track_id, last_detection=detection)
        self._next_track_id += 1
        return track

    def _update_track(self, track: TrackState, detection: Detection) -> None:
        """Update a matched track with the latest detection."""
        track.last_detection = detection
        track.hits += 1
        track.missed_frames = 0
        track.confidence_history.append(detection.confidence)
        while len(track.confidence_history) > self.confidence_window:
            track.confidence_history.popleft()

    def _confirmed_detections(self) -> list[Detection]:
        """Build output detections from currently confirmed tracks."""
        confirmed_detections: list[Detection] = []
        for track in self._tracks:
            if track.hits < self.confirm_frames or track.missed_frames > self.max_missed_frames:
                continue

            base_detection = track.last_detection
            confirmed_detections.append(
                Detection(
                    class_id=base_detection.class_id,
                    class_name=base_detection.class_name,
                    recycle_bin=base_detection.recycle_bin,
                    confidence=track.smoothed_confidence(self.confidence_window),
                    bbox=base_detection.bbox,
                    center=base_detection.center,
                    pickup_point=base_detection.pickup_point,
                    area_ratio=base_detection.area_ratio,
                    bbox_clipped=base_detection.bbox_clipped,
                    mask_area_ratio=base_detection.mask_area_ratio,
                    segmentation_available=base_detection.segmentation_available,
                    track_id=track.track_id,
                    track_confirmed=True,
                    track_hits=track.hits,
                    track_missed_frames=track.missed_frames,
                    track_score=track.smoothed_confidence(self.confidence_window),
                )
            )

        return confirmed_detections
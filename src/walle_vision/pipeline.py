from __future__ import annotations

"""Main orchestration pipeline.

Responsibilities:
1. Capture frames from camera in a dedicated thread.
2. Run inference/tracking in the main processing thread.
3. Save only stable detections for compact production logs.
4. Optionally render debug visualization.
"""

import json
import os
import queue
import threading
from pathlib import Path

import cv2

from .camera import CameraStream
from .detector import DetectorConfig, WasteDetector
from .types import Detection
from .tracking import TemporalDetectionTracker
from .visualization import draw_detections


class VisionPipeline:
    """Coordinate capture, inference, tracking, visualization, and export."""

    def __init__(
        self,
        detector_cfg: DetectorConfig,
        source: int | str,
        output_dir: str,
        show: bool,
        infer_every_n_frames: int,
        save_every_n_frames: int,
        track_iou_threshold: float,
        track_confirm_frames: int,
        track_max_missed_frames: int,
        track_confidence_window: int,
        display_persist_frames: int,
        width: int,
        height: int,
        fps: int,
    ) -> None:
        self.detector = WasteDetector(detector_cfg)
        self.tracker = TemporalDetectionTracker(
            iou_threshold=track_iou_threshold,
            confirm_frames=track_confirm_frames,
            max_missed_frames=track_max_missed_frames,
            confidence_window=track_confidence_window,
        )
        self.source = source
        self.output_dir = Path(output_dir)
        self.show = show
        self.infer_every_n_frames = max(1, infer_every_n_frames)
        self.save_every_n_frames = max(1, save_every_n_frames)
        self.width = width
        self.height = height
        self.fps = fps
        self._detected_frame_count = 0
        self._display_persist_frames = max(0, display_persist_frames)
        self._display_cache: dict[int, tuple[int, Detection]] = {}
        self._last_rendered_frame = None

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results_jsonl = self.output_dir / "detections.jsonl"
        self.frames_dir = self.output_dir / "frames"
        self.frames_dir.mkdir(parents=True, exist_ok=True)

    def run(self) -> None:
        """Execute real-time loop using threaded capture and bounded queue."""
        stream = CameraStream(self.source, width=self.width, height=self.height, fps=self.fps)
        frame_queue: queue.Queue[tuple[int, float, object]] = queue.Queue(maxsize=2)
        stop_event = threading.Event()
        capture_errors: list[Exception] = []

        def capture_loop() -> None:
            try:
                for frame_index, timestamp, frame in stream.frames():
                    if stop_event.is_set():
                        break
                    if frame_queue.full():
                        try:
                            frame_queue.get_nowait()
                        except queue.Empty:
                            pass
                    frame_queue.put_nowait((frame_index, timestamp, frame))
            except Exception as exc:  # pragma: no cover
                capture_errors.append(exc)
            finally:
                stop_event.set()

        capture_thread = threading.Thread(target=capture_loop, name="camera-capture", daemon=True)
        capture_thread.start()

        try:
            while not stop_event.is_set() or not frame_queue.empty():
                try:
                    frame_index, timestamp, frame = frame_queue.get(timeout=0.1)
                except queue.Empty:
                    if self.show and self._last_rendered_frame is not None:
                        cv2.imshow("wall-e-vision", self._last_rendered_frame)
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            stop_event.set()
                    continue

                if frame_index % self.infer_every_n_frames != 0:
                    if self.show:
                        display_frame = self._last_rendered_frame if self._last_rendered_frame is not None else frame
                        cv2.imshow("wall-e-vision", display_frame)
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            stop_event.set()
                            break
                    continue

                raw_detections = self.detector.infer(frame)
                stable_detections = self.tracker.update(raw_detections)

                if stable_detections:
                    self._append_result(frame_index, timestamp, stable_detections)

                display_detections = self._build_display_detections(frame_index, stable_detections)
                annotated = draw_detections(frame, display_detections)
                self._last_rendered_frame = annotated

                if stable_detections:
                    self._detected_frame_count += 1
                    if self._detected_frame_count % self.save_every_n_frames == 0:
                        out_file = self.frames_dir / f"frame_{frame_index:06d}.jpg"
                        cv2.imwrite(str(out_file), annotated)

                if self.show:
                    cv2.imshow("wall-e-vision", annotated)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        stop_event.set()
                        break

            if capture_errors:
                raise RuntimeError(f"Capture thread failed: {capture_errors[0]}")
        except KeyboardInterrupt:
            stop_event.set()
        finally:
            stop_event.set()
            capture_thread.join(timeout=1.0)
            stream.close()
            if self.show:
                cv2.destroyAllWindows()

    def _build_display_detections(self, frame_index: int, stable_detections: list[Detection]) -> list[Detection]:
        """Build live-overlay detections with short visual persistence."""
        if self._display_persist_frames <= 0:
            return stable_detections

        current_track_ids: set[int] = set()
        for detection in stable_detections:
            if detection.track_id is None:
                continue
            current_track_ids.add(detection.track_id)
            self._display_cache[detection.track_id] = (frame_index, detection)

        display_detections = list(stable_detections)

        expired_track_ids: list[int] = []
        for track_id, (last_seen_frame, cached_detection) in self._display_cache.items():
            if track_id in current_track_ids:
                continue
            if frame_index - last_seen_frame <= self._display_persist_frames:
                display_detections.append(cached_detection)
            else:
                expired_track_ids.append(track_id)

        for track_id in expired_track_ids:
            self._display_cache.pop(track_id, None)

        return display_detections

    def _append_result(self, frame_index: int, timestamp: float, stable_detections: list[Detection]) -> None:
        """Append a compact JSONL payload for one frame with stable detections."""
        payload = {
            "frame_index": frame_index,
            "timestamp": timestamp,
            "detections": [
                {
                    "class_id": d.class_id,
                    "class_name": d.class_name,
                    "recycle_bin": d.recycle_bin,
                    "confidence": round(d.confidence, 5),
                    "bbox_xyxy": d.bbox,
                    "pickup_xy": d.pickup_point,
                    "track_id": d.track_id,
                    "track_score": round(d.track_score, 5),
                    "segmentation_available": d.segmentation_available,
                }
                for d in stable_detections
            ],
        }
        with self.results_jsonl.open("a", encoding="utf-8") as file:
            file.write(json.dumps(payload, ensure_ascii=True) + os.linesep)

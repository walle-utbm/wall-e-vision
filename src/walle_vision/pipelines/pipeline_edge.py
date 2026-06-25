from __future__ import annotations

"""Standalone edge inference pipeline."""

import json
import os
import queue
import threading
from pathlib import Path

import cv2

from ..ai.detector import DetectorConfig, WasteDetector
from ..config import AppConfig
from ..core.camera import CameraStream
from ..types import Detection
from ..utils.live import MjpegServer, VideoRecorder
from ..utils.visualization import draw_detections


class EdgeStandalonePipeline:
    _REVIEW_EMPTY_RESET_FRAMES = 5

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.camera = CameraStream.create(config.hardware, config.camera_type, config.camera)
        self.detector = WasteDetector(
            DetectorConfig(
                model_path=None if config.detector.backend == "edge_impulse_http" else str(config.resolve_model_path()),
                backend=config.detector.backend,
                conf_threshold=config.detector.conf_threshold,
                iou_threshold=config.detector.iou_threshold,
                image_size=config.detector.image_size,
                max_detections=config.detector.max_detections,
                edge_impulse_url=config.detector.edge_impulse_url,
                edge_impulse_timeout_sec=config.detector.edge_impulse_timeout_sec,
                debug_inference=config.runtime.debug_inference,
            )
        )
        self.output_dir = Path(config.config_dir / config.paths.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results_jsonl = self.output_dir / "detections.jsonl"
        self.predict_dir = self.output_dir / "predict"
        self.predict_dir.mkdir(parents=True, exist_ok=True)
        self.camera_test_dir = self.output_dir / "camera_test"
        if config.runtime.camera_test_mode:
            self.camera_test_dir.mkdir(parents=True, exist_ok=True)
        self.review_export = config.runtime.review_export
        self.review_dir = Path(config.config_dir / config.paths.review_dir)
        if self.review_export:
            self.review_dir.mkdir(parents=True, exist_ok=True)
        self._last_review_signature: frozenset[int] | None = None
        self._review_empty_streak = 0
        self._last_rendered_frame = None
        self._last_detections: list[Detection] = []
        self._last_test_frame_save_time = 0.0
        self._proc_start_time = 0.0
        self._proc_count = 0

        # Live outputs (SSH-friendly): MJPEG stream and/or .mp4 recording.
        self._stream: MjpegServer | None = None
        if config.runtime.stream_enabled:
            self._stream = MjpegServer(
                config.runtime.stream_host,
                config.runtime.stream_port,
                config.runtime.stream_quality,
            )
            print(
                f"MJPEG live stream on http://{config.runtime.stream_host}:{config.runtime.stream_port}/ "
                "(ouvre l'URL via le port forward VSCode)"
            )
        self._recorder: VideoRecorder | None = None
        if config.runtime.record_enabled:
            record_path = Path(config.runtime.record_path)
            if not record_path.is_absolute():
                record_path = config.config_dir / record_path
            record_fps = config.runtime.record_fps or config.camera.fps
            self._recorder = VideoRecorder(record_path, record_fps)
            print(f"Recording annotated feed to {record_path}")

    def run(self) -> None:
        frame_queue: queue.Queue[tuple[int, float, object]] = queue.Queue(maxsize=2)
        stop_event = threading.Event()
        capture_errors: list[Exception] = []

        def capture_loop() -> None:
            try:
                for frame_index, timestamp, frame in self.camera.frames():
                    if stop_event.is_set():
                        break
                    if frame_queue.full():
                        try:
                            frame_queue.get_nowait()
                        except queue.Empty:
                            pass
                    frame_queue.put_nowait((frame_index, timestamp, frame))
            except Exception as exc:
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
                    if self.config.runtime.show and self._last_rendered_frame is not None:
                        cv2.imshow("wall-e-vision", self._last_rendered_frame)
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            stop_event.set()
                    continue

                if self.config.runtime.camera_test_mode and timestamp - self._last_test_frame_save_time >= self.config.runtime.camera_test_interval_sec:
                    test_frame_path = self.camera_test_dir / f"frame_{int(timestamp)}.jpg"
                    cv2.imwrite(str(test_frame_path), frame)
                    self._last_test_frame_save_time = timestamp

                if frame_index % self.config.runtime.infer_every_n_frames != 0:
                    # Frame non inférée : on réutilise les dernières détections pour
                    # garder un overlay fluide à la cadence caméra.
                    display_frame = draw_detections(frame, self._last_detections)
                    self._emit_live(display_frame)
                    if self.config.runtime.show:
                        cv2.imshow("wall-e-vision", display_frame)
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            stop_event.set()
                            break
                    continue

                raw_detections = self.detector.infer(frame)
                self._append_result(frame_index, timestamp, raw_detections)

                annotated = draw_detections(frame, raw_detections)
                self._last_rendered_frame = annotated
                self._last_detections = raw_detections
                self._emit_live(annotated)

                if raw_detections and frame_index % self.config.runtime.save_every_n_frames == 0:
                    out_file = self.predict_dir / f"frame_{frame_index:06d}.jpg"
                    cv2.imwrite(str(out_file), annotated)

                if self.review_export:
                    if raw_detections:
                        self._review_empty_streak = 0
                        signature = frozenset(d.class_id for d in raw_detections)
                        if signature != self._last_review_signature:
                            self._export_review(frame, raw_detections, frame_index, timestamp)
                            self._last_review_signature = signature
                    else:
                        self._review_empty_streak += 1
                        if self._review_empty_streak >= self._REVIEW_EMPTY_RESET_FRAMES:
                            self._last_review_signature = None

                if self._proc_start_time == 0.0:
                    self._proc_start_time = timestamp
                self._proc_count += 1
                elapsed = timestamp - self._proc_start_time
                if elapsed >= 5.0:
                    fps = float(self._proc_count) / max(1e-6, elapsed)
                    print(f"Pipeline processed {self._proc_count} frames in {elapsed:.1f}s (~{fps:.2f} FPS)")
                    self._proc_start_time = timestamp
                    self._proc_count = 0

                if self.config.runtime.show:
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
            self.camera.close()
            if self._recorder is not None:
                self._recorder.close()
            if self._stream is not None:
                self._stream.close()
            if self.config.runtime.show:
                cv2.destroyAllWindows()

    def _emit_live(self, frame) -> None:
        if self._stream is not None:
            self._stream.update(frame)
        if self._recorder is not None:
            self._recorder.write(frame)

    def _append_result(self, frame_index: int, timestamp: float, detections: list[Detection]) -> None:
        if not detections:
            print(f"Frame {frame_index} @ {timestamp:.2f}s: No detections")
            return
        print(f"Frame {frame_index} @ {timestamp:.2f}s: Detected {detections[0].class_name} objects, confidences {[round(d.confidence, 3) for d in detections]}")
        payload = {
            "frame_index": frame_index,
            "timestamp": timestamp,
            "detections": [
                {
                    "class_id": detection.class_id,
                    "class_name": detection.class_name,
                    "recycle_bin": detection.recycle_bin,
                    "confidence": round(detection.confidence, 5),
                    "bbox_xyxy": detection.bbox,
                    "pickup_xy": detection.pickup_point,
                    "segmentation_available": detection.segmentation_available,
                }
                for detection in detections
            ],
        }
        with self.results_jsonl.open("a", encoding="utf-8") as file:
            file.write(json.dumps(payload, ensure_ascii=True) + os.linesep)

    def _export_review(self, frame, detections: list[Detection], frame_index: int, timestamp: float) -> None:
        # Dépose une paire <id>.jpg (image VIERGE) + <id>.json pour la file de
        # validation humaine de wall-e-core (construction du dataset YOLO).
        item_id = f"frame_{frame_index:06d}_{int(timestamp)}"
        frame_height, frame_width = frame.shape[:2]
        sidecar = {
            "id": item_id,
            "timestamp": timestamp,
            "image": {"width": int(frame_width), "height": int(frame_height)},
            "detections": [
                {
                    "class_id": detection.class_id,
                    "class_name": detection.class_name,
                    "confidence": round(detection.confidence, 5),
                    "bbox_xyxy": list(detection.bbox),
                }
                for detection in detections
            ],
        }
        cv2.imwrite(str(self.review_dir / f"{item_id}.jpg"), frame)
        with (self.review_dir / f"{item_id}.json").open("w", encoding="utf-8") as file:
            json.dump(sidecar, file, ensure_ascii=True)
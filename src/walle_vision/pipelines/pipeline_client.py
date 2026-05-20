from __future__ import annotations

"""Raspberry Pi capture-and-stream pipeline."""

import json
import os
import queue
import socket
import threading
import time
from collections import deque
from pathlib import Path

import cv2

from ..config import AppConfig
from ..core.camera import CameraStream
from ..network.transport import StreamChannel, TransportMessage


class StreamClientPipeline:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.camera = CameraStream.create(config.hardware, config.camera_type, config.camera)
        self.output_dir = Path(config.config_dir / config.paths.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results_jsonl = self.output_dir / "remote_results.jsonl"
        self.camera_test_dir = self.output_dir / "camera_test"
        if config.runtime.camera_test_mode:
            self.camera_test_dir.mkdir(parents=True, exist_ok=True)
        self._last_test_frame_save_time = 0.0
        self._last_rendered_frame: object | None = None
        self._last_result_summary = "waiting for PC"
        self._last_debug_summary = "fps=0.0"
        self._received_results = 0
        self._sent_frames = 0
        self._inflight_frames = 0
        self._inflight_condition = threading.Condition()
        self._timing_lock = threading.Lock()
        self._pending_frames: dict[int, dict[str, float]] = {}
        self._sent_samples: deque[float] = deque()
        self._received_samples: deque[float] = deque()

    def run(self) -> None:
        frame_queue: queue.Queue[tuple[int, float, object]] = queue.Queue(maxsize=max(1, self.config.network.max_inflight_frames))
        stop_event = threading.Event()
        capture_errors: list[Exception] = []
        channel = StreamChannel(self.config.network.client_host, self.config.network.client_port)

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

        def receive_loop() -> None:
            while not stop_event.is_set() or channel.connected:
                try:
                    message = channel.receive_message(timeout_sec=0.25)
                except socket.timeout:
                    continue
                except (ConnectionError, OSError):
                    stop_event.set()
                    break
                received_wall = time.time()
                received_mono = time.perf_counter()
                with self._inflight_condition:
                    if self._inflight_frames > 0:
                        self._inflight_frames -= 1
                    self._inflight_condition.notify_all()
                self._handle_remote_message(message, received_wall, received_mono)

        try:
            while not stop_event.is_set():
                try:
                    channel.connect()
                    break
                except OSError as exc:
                    print(f"Waiting for PC inference server at {self.config.network.client_host}:{self.config.network.client_port} ({exc}); retrying in {self.config.network.reconnect_delay_sec:.1f}s")
                    time.sleep(self.config.network.reconnect_delay_sec)

            if not channel.connected:
                return

            capture_thread = threading.Thread(target=capture_loop, name="camera-capture", daemon=True)
            receive_thread = threading.Thread(target=receive_loop, name="remote-receiver", daemon=True)
            capture_thread.start()
            receive_thread.start()

            while not stop_event.is_set() or not frame_queue.empty():
                try:
                    frame_index, timestamp, frame = frame_queue.get(timeout=0.1)
                except queue.Empty:
                    self._render_status_overlay()
                    continue

                if self.config.runtime.camera_test_mode and timestamp - self._last_test_frame_save_time >= self.config.runtime.camera_test_interval_sec:
                    test_frame_path = self.camera_test_dir / f"frame_{int(timestamp)}.jpg"
                    cv2.imwrite(str(test_frame_path), frame)
                    self._last_test_frame_save_time = timestamp
                    print(f"Camera test frame saved: {test_frame_path.name}")

                with self._inflight_condition:
                    while self._inflight_frames >= self.config.network.max_inflight_frames and not stop_event.is_set():
                        self._inflight_condition.wait(timeout=0.1)
                    if stop_event.is_set():
                        break

                send_started_wall = time.time()
                send_duration_sec = channel.send_frame(frame_index, timestamp, frame, self.config.network.jpeg_quality)
                send_finished_wall = time.time()
                send_finished_mono = time.perf_counter()
                capture_age_ms = max(0.0, (send_started_wall - timestamp) * 1000.0)
                send_total_ms = send_duration_sec * 1000.0
                with self._timing_lock:
                    self._pending_frames[frame_index] = {
                        "capture_age_ms": capture_age_ms,
                        "send_total_ms": send_total_ms,
                        "send_finished_wall": send_finished_wall,
                        "send_finished_mono": send_finished_mono,
                    }
                    self._sent_samples.append(send_finished_mono)
                    self._trim_samples(self._sent_samples, send_finished_mono)
                with self._inflight_condition:
                    self._inflight_frames += 1
                self._sent_frames += 1
                self._last_rendered_frame = frame

                if self.config.runtime.show:
                    self._render_status_overlay(frame)

            if capture_errors:
                raise RuntimeError(f"Capture thread failed: {capture_errors[0]}")
        except KeyboardInterrupt:
            stop_event.set()
        finally:
            stop_event.set()
            if 'capture_thread' in locals():
                capture_thread.join(timeout=1.0)
            if 'receive_thread' in locals():
                receive_thread.join(timeout=1.0)
            channel.close()
            self.camera.close()
            if self.config.runtime.show:
                cv2.destroyAllWindows()

    def _handle_remote_message(self, message: TransportMessage, received_wall: float, received_mono: float) -> None:
        if message.header.get("kind") != "result":
            return

        try:
            payload = json.loads(message.body.decode("utf-8"))
        except Exception as exc:
            print(f"Invalid result payload received from PC: {exc}")
            return

        frame_index = payload.get("frame_index")
        pending = None
        if isinstance(frame_index, int):
            with self._timing_lock:
                pending = self._pending_frames.pop(frame_index, None)
                self._received_samples.append(received_mono)
                self._trim_samples(self._received_samples, received_mono)

        self._received_results += 1
        detection_count = len(payload.get("detections", []))
        self._last_result_summary = f"frame {payload.get('frame_index', '?')} -> {detection_count} detections"

        debug_parts: list[str] = []
        if pending is not None:
            roundtrip_ms = max(0.0, (received_wall - pending["send_finished_wall"]) * 1000.0)
            debug_parts.append(f"capture={pending['capture_age_ms']:.1f}ms")
            debug_parts.append(f"send={pending['send_total_ms']:.1f}ms")
            debug_parts.append(f"rtt={roundtrip_ms:.1f}ms")

        server_timing = payload.get("timing")
        if isinstance(server_timing, dict):
            for key in ("processing_ms", "inference_ms", "model_ms"):
                value = server_timing.get(key)
                if isinstance(value, (int, float)):
                    debug_parts.append(f"pc={float(value):.1f}ms")
                    break

        if isinstance(payload.get("processing_ms"), (int, float)):
            debug_parts.append(f"pc={float(payload['processing_ms']):.1f}ms")

        with self._timing_lock:
            sent_fps = float(len(self._sent_samples))
            received_fps = float(len(self._received_samples))

        debug_parts.append(f"fps_tx={sent_fps:.1f}")
        debug_parts.append(f"fps_rx={received_fps:.1f}")
        self._last_debug_summary = " | ".join(debug_parts)

        with self.results_jsonl.open("a", encoding="utf-8") as file:
            file.write(json.dumps(payload, ensure_ascii=True) + os.linesep)

        print(f"Result received: {self._last_result_summary} | {self._last_debug_summary}")

    def _render_status_overlay(self, frame: object | None = None) -> None:
        if not self.config.runtime.show:
            return
        display_frame = frame if frame is not None else self._last_rendered_frame
        if display_frame is None:
            return
        annotated = display_frame.copy() if hasattr(display_frame, "copy") else display_frame
        cv2.putText(
            annotated,
            f"sent={self._sent_frames} received={self._received_results} {self._last_result_summary} {self._last_debug_summary}",
            (16, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        self._last_rendered_frame = annotated
        cv2.imshow("wall-e-vision", annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            raise KeyboardInterrupt

    def _trim_samples(self, samples, now: float, window_sec: float = 1.0) -> None:
        cutoff = now - window_sec
        while samples and samples[0] < cutoff:
            samples.popleft()
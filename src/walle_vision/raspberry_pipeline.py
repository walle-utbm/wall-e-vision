from __future__ import annotations

"""Raspberry Pi capture-and-stream pipeline.

The Raspberry Pi only captures frames, forwards them to the PC, and stores the
results received back from the model server.
"""

import json
import os
import queue
import threading
import time
from pathlib import Path

import cv2

from .camera import CameraStream
from .transport import StreamChannel, TransportMessage


class RaspberryVisionPipeline:
    """Stream camera frames from the Raspberry Pi to a remote PC."""

    def __init__(
        self,
        source: int | str,
        output_dir: str,
        show: bool,
        width: int,
        height: int,
        fps: int,
        host: str,
        port: int,
        jpeg_quality: int,
        reconnect_delay_sec: float,
        max_inflight_frames: int,
        camera_test_mode: bool = False,
        camera_test_interval_sec: float = 5.0,
    ) -> None:
        self.source = source
        self.output_dir = Path(output_dir)
        self.show = show
        self.width = width
        self.height = height
        self.fps = fps
        self.host = host
        self.port = port
        self.jpeg_quality = jpeg_quality
        self.reconnect_delay_sec = reconnect_delay_sec
        self.max_inflight_frames = max(1, max_inflight_frames)
        self.camera_test_mode = camera_test_mode
        self.camera_test_interval_sec = camera_test_interval_sec

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results_jsonl = self.output_dir / "remote_results.jsonl"
        self.camera_test_dir = self.output_dir / "camera_test"
        if self.camera_test_mode:
            self.camera_test_dir.mkdir(parents=True, exist_ok=True)

        self._last_test_frame_save_time = 0.0
        self._last_rendered_frame: object | None = None
        self._last_result_summary = "waiting for PC"
        self._received_results = 0
        self._sent_frames = 0

    def run(self) -> None:
        """Capture frames and stream them to the PC until interrupted."""
        stream = CameraStream(self.source, width=self.width, height=self.height, fps=self.fps)
        frame_queue: queue.Queue[tuple[int, float, object]] = queue.Queue(maxsize=self.max_inflight_frames)
        stop_event = threading.Event()
        capture_errors: list[Exception] = []
        channel = StreamChannel(self.host, self.port)

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
            except Exception as exc:  # pragma: no cover - hardware dependent
                capture_errors.append(exc)
            finally:
                stop_event.set()

        def receive_loop() -> None:
            while not stop_event.is_set() or channel.connected:
                try:
                    message = channel.receive_message(timeout_sec=0.25)
                except TimeoutError:
                    continue
                except ConnectionError:
                    stop_event.set()
                    break
                except OSError:
                    stop_event.set()
                    break
                self._handle_remote_message(message)

        try:
            while not stop_event.is_set():
                try:
                    channel.connect()
                    break
                except OSError as exc:
                    print(
                        f"Waiting for PC inference server at {self.host}:{self.port} "
                        f"({exc}); retrying in {self.reconnect_delay_sec:.1f}s"
                    )
                    time.sleep(self.reconnect_delay_sec)

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

                if self.camera_test_mode and timestamp - self._last_test_frame_save_time >= self.camera_test_interval_sec:
                    test_frame_path = self.camera_test_dir / f"frame_{int(timestamp)}.jpg"
                    cv2.imwrite(str(test_frame_path), frame)
                    self._last_test_frame_save_time = timestamp
                    print(f"📷 Camera test frame saved: {test_frame_path.name}")

                channel.send_frame(frame_index, timestamp, frame, self.jpeg_quality)
                self._sent_frames += 1
                self._last_rendered_frame = frame

                if self.show:
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
            stream.close()
            if self.show:
                cv2.destroyAllWindows()

    def _handle_remote_message(self, message: TransportMessage) -> None:
        """Store one remote result packet locally and print a short status line."""
        if message.header.get("kind") != "result":
            return

        try:
            payload = json.loads(message.body.decode("utf-8"))
        except Exception as exc:
            print(f"⚠️ Invalid result payload received from PC: {exc}")
            return

        self._received_results += 1
        self._last_result_summary = f"frame {payload.get('frame_index', '?')} -> {len(payload.get('detections', []))} detections"

        with self.results_jsonl.open("a", encoding="utf-8") as file:
            file.write(json.dumps(payload, ensure_ascii=True) + os.linesep)

        print(f"📡 Result received: {self._last_result_summary}")

    def _render_status_overlay(self, frame: object | None = None) -> None:
        """Render a light status overlay when the display flag is enabled."""
        if not self.show:
            return

        display_frame = frame if frame is not None else self._last_rendered_frame
        if display_frame is None:
            return

        if not isinstance(display_frame, cv2.UMat):
            annotated = display_frame.copy() if hasattr(display_frame, "copy") else display_frame
        else:
            annotated = display_frame.get()

        cv2.putText(
            annotated,
            f"sent={self._sent_frames} received={self._received_results} {self._last_result_summary}",
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
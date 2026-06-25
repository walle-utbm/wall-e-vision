from __future__ import annotations

"""Live outputs for the edge pipeline: MJPEG HTTP streaming and video recording.

Both are SSH-friendly: the MJPEG server lets a browser display the annotated feed
through a forwarded port (no X11 / no DISPLAY needed), and the recorder writes the
same annotated frames to an .mp4 file for later review.
"""

import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np


_PAGE = (
    b"<!doctype html><html><head><title>wall-e-vision</title>"
    b"<style>body{margin:0;background:#111}img{width:100vw;height:auto;display:block}</style>"
    b"</head><body><img src=\"/stream.mjpg\"></body></html>"
)


class MjpegServer:
    """Serves the latest annotated frame as a multipart MJPEG stream."""

    def __init__(self, host: str, port: int, quality: int = 80) -> None:
        self._quality = int(quality)
        self._condition = threading.Condition()
        self._latest_jpeg: bytes | None = None
        self._stopped = False

        server_self = self

        class _Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args) -> None:  # silence default logging
                return

            def do_GET(self) -> None:  # noqa: N802 (http.server API)
                if self.path in ("/", "/index.html"):
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html")
                    self.send_header("Content-Length", str(len(_PAGE)))
                    self.end_headers()
                    self.wfile.write(_PAGE)
                    return
                if self.path != "/stream.mjpg":
                    self.send_error(404)
                    return

                self.send_response(200)
                self.send_header("Age", "0")
                self.send_header("Cache-Control", "no-cache, private")
                self.send_header("Pragma", "no-cache")
                self.send_header(
                    "Content-Type", "multipart/x-mixed-replace; boundary=frame"
                )
                self.end_headers()
                try:
                    while not server_self._stopped:
                        frame = server_self._wait_for_frame()
                        if frame is None:
                            break
                        self.wfile.write(b"--frame\r\n")
                        self.send_header("Content-Type", "image/jpeg")
                        self.send_header("Content-Length", str(len(frame)))
                        self.end_headers()
                        self.wfile.write(frame)
                        self.wfile.write(b"\r\n")
                except (BrokenPipeError, ConnectionResetError):
                    # Client closed the tab; just drop this streaming connection.
                    return

        self._server = ThreadingHTTPServer((host, port), _Handler)
        self._server.daemon_threads = True
        self._thread = threading.Thread(
            target=self._server.serve_forever, name="mjpeg-server", daemon=True
        )
        self._thread.start()

    def _wait_for_frame(self, timeout: float = 1.0) -> bytes | None:
        with self._condition:
            self._condition.wait(timeout=timeout)
            return self._latest_jpeg

    def update(self, frame_bgr: np.ndarray) -> None:
        ok, buffer = cv2.imencode(
            ".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, self._quality]
        )
        if not ok:
            return
        with self._condition:
            self._latest_jpeg = buffer.tobytes()
            self._condition.notify_all()

    def close(self) -> None:
        self._stopped = True
        with self._condition:
            self._condition.notify_all()
        self._server.shutdown()
        self._server.server_close()


class VideoRecorder:
    """Lazily-initialized .mp4 writer for the annotated frames."""

    def __init__(self, output_path: Path, fps: float) -> None:
        self._output_path = Path(output_path)
        self._output_path.parent.mkdir(parents=True, exist_ok=True)
        self._fps = float(fps) if fps and fps > 0 else 20.0
        self._writer: cv2.VideoWriter | None = None
        self._size: tuple[int, int] | None = None

    def write(self, frame_bgr: np.ndarray) -> None:
        height, width = frame_bgr.shape[:2]
        if self._writer is None:
            self._size = (width, height)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self._writer = cv2.VideoWriter(
                str(self._output_path), fourcc, self._fps, self._size
            )
            if not self._writer.isOpened():
                raise RuntimeError(
                    f"Unable to open VideoWriter for {self._output_path}"
                )
        if self._size != (width, height):
            frame_bgr = cv2.resize(frame_bgr, self._size)
        self._writer.write(frame_bgr)

    def close(self) -> None:
        if self._writer is not None:
            self._writer.release()
            self._writer = None

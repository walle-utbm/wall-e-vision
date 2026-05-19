from __future__ import annotations

"""Camera capture utilities.

This module provides a tiny abstraction over camera backends so the rest
of the project can consume frames as a simple Python generator.
Supports standard webcams, Raspberry Pi camera, and video files.
"""

import time
import sys
from pathlib import Path
from typing import Generator

import cv2
import numpy as np


def _add_system_site_packages() -> None:
    """Expose Raspberry Pi OS site-packages to a virtualenv when needed."""
    candidates = [
        Path("/usr/lib/python3/dist-packages"),
        Path("/usr/lib/python3.13/dist-packages"),
        Path("/usr/local/lib/python3.13/dist-packages"),
    ]
    for path in candidates:
        if path.exists():
            path_str = str(path)
            if path_str not in sys.path:
                sys.path.append(path_str)


try:
    from picamera2 import Picamera2
except ImportError:  # pragma: no cover - optional dependency
    _add_system_site_packages()
    try:
        from picamera2 import Picamera2
    except ImportError:  # pragma: no cover - optional dependency
        Picamera2 = None


class CameraStream:
    """Stream frames from a webcam, video source, or Raspberry Pi camera.

    The wrapper applies low-latency camera settings and yields
    (frame_index, timestamp, frame) tuples. Optimized for Raspberry Pi
    with ArduCAM support.

    Attributes:
        _cap: OpenCV VideoCapture handle.
        _frame_index: Number of frames already yielded.
    """

    def __init__(self, source: int | str = 0, width: int = 640, height: int = 640, fps: int = 30) -> None:
        """Initialize camera stream.

        Args:
            source: Camera index (0, 1, ...) or stream/video path.
                   For Raspberry Pi IMX708 libcamera: use 0 (default).
                   For USB cameras: try 0 or 1.
            width: Requested capture width (640 matches model training resolution).
            height: Requested capture height (640 matches model training resolution).
            fps: Requested capture frame rate (30 for smooth real-time on 8GB RAM).
        """
        self._source = source
        self._backend = "opencv"
        self._picam2 = None
        self._cap = None

        if isinstance(source, int) and Picamera2 is not None:
            self._backend = "picamera2"
            self._picam2 = Picamera2(source)
            config = self._picam2.create_video_configuration(
                main={"size": (width, height), "format": "RGB888"},
                buffer_count=2,
            )
            self._picam2.configure(config)
            self._picam2.start()
            
            # Warmup period for white balance stabilization (shorter to reduce thermal load)
            print("📷 Warming up camera sensor (stabilizing white balance)...")
            for _ in range(3):  # ~1 second at 30 FPS
                _ = self._picam2.capture_array()
                time.sleep(0.1)
        else:
            self._cap = cv2.VideoCapture(source, cv2.CAP_V4L2)
            if not self._cap.isOpened():
                # Fallback to default backend for webcams, files, and URLs.
                self._cap = cv2.VideoCapture(source)

            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            self._cap.set(cv2.CAP_PROP_FPS, fps)
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            # Best-effort camera stabilization for more consistent detections.
            self._set_if_supported(cv2.CAP_PROP_AUTOFOCUS, 0)
            self._set_if_supported(cv2.CAP_PROP_AUTO_WB, 0)
            self._set_if_supported(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
        self._frame_index = 0

        if self._backend == "opencv" and not self._cap.isOpened():
            raise RuntimeError(
                f"Unable to open camera source: {source}. "
                "For Raspberry Pi CSI cameras, install picamera2 or expose the camera as a V4L2 device."
            )

    def _set_if_supported(self, prop_id: int, value: float) -> None:
        """Set a capture property without failing when backend/camera ignores it."""
        try:
            self._cap.set(prop_id, value)
        except Exception:
            pass

    def frames(self) -> Generator[tuple[int, float, np.ndarray], None, None]:
        """Yield frames continuously until capture fails or is closed."""
        while True:
            if self._backend == "picamera2":
                frame = self._picam2.capture_array()
                if frame is None:
                    if self._frame_index == 0:
                        raise RuntimeError(
                            f"Unable to read a frame from camera source: {self._source}. "
                            "Check that picamera2/libcamera is installed and the camera is accessible."
                        )
                    break
                if frame.ndim == 3 and frame.shape[2] == 3:
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                elif frame.ndim == 3 and frame.shape[2] == 4:
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
                
                # Add small delay to reduce thermal stress on Arducam IMX708
                time.sleep(0.01)
            else:
                ok, frame = self._cap.read()
                if not ok:
                    if self._frame_index == 0:
                        raise RuntimeError(
                            f"Unable to read a frame from camera source: {self._source}. "
                            "Check that the camera is connected and accessible (V4L2/libcamera)."
                        )
                    break
            ts = time.time()
            idx = self._frame_index
            self._frame_index += 1
            yield idx, ts, frame

    def close(self) -> None:
        """Release camera resources."""
        if self._picam2 is not None:
            self._picam2.stop()
        if self._cap is not None:
            self._cap.release()

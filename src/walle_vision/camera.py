from __future__ import annotations

"""Camera capture utilities.

This module provides a tiny abstraction over OpenCV VideoCapture so the rest
of the project can consume frames as a simple Python generator.
"""

import time
from typing import Generator

import cv2
import numpy as np


class CameraStream:
    """Stream frames from a webcam or video source.

    The wrapper applies low-latency camera settings and yields
    (frame_index, timestamp, frame) tuples.

    Attributes:
        _cap: OpenCV VideoCapture handle.
        _frame_index: Number of frames already yielded.
    """

    def __init__(self, source: int | str = 0, width: int = 640, height: int = 640, fps: int = 30) -> None:
        """Initialize camera stream.

        Args:
            source: Camera index (0, 1, ...) or stream/video path.
            width: Requested capture width.
            height: Requested capture height.
            fps: Requested capture frame rate.
        """
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

        if not self._cap.isOpened():
            raise RuntimeError(f"Unable to open camera source: {source}")

    def _set_if_supported(self, prop_id: int, value: float) -> None:
        """Set a capture property without failing when backend/camera ignores it."""
        try:
            self._cap.set(prop_id, value)
        except Exception:
            pass

    def frames(self) -> Generator[tuple[int, float, np.ndarray], None, None]:
        """Yield frames continuously until capture fails or is closed."""
        while True:
            ok, frame = self._cap.read()
            if not ok:
                break
            ts = time.time()
            idx = self._frame_index
            self._frame_index += 1
            yield idx, ts, frame

    def close(self) -> None:
        """Release camera resources."""
        if self._cap is not None:
            self._cap.release()

from __future__ import annotations

"""Camera capture utilities.

This module provides a tiny abstraction over camera backends so the rest
of the project can consume frames as a simple Python generator.
Supports the RubikPi IMX708 camera, webcams, and video files.
"""

import sys
import time
from pathlib import Path
from typing import Generator

import numpy as np


def _add_system_site_packages() -> None:
    """Expose distro-installed bindings to the venv when Python cannot see them.

    This is needed on RubikPi because `picamera2` and `gi` are often installed
    by the OS package manager rather than inside the project virtualenv.
    """
    major = sys.version_info.major
    minor = sys.version_info.minor
    candidates = [
        Path("/usr/lib/python3/dist-packages"),
        Path(f"/usr/lib/python3.{major}/dist-packages"),
        Path(f"/usr/local/lib/python3.{major}/dist-packages"),
        Path(f"/usr/lib/aarch64-linux-gnu/python{major}.{minor}/site-packages"),
        Path(f"/usr/local/lib/aarch64-linux-gnu/python{major}.{minor}/site-packages"),
        Path(f"/usr/lib/python3.{minor}/dist-packages"),
        Path(f"/usr/local/lib/python3.{minor}/dist-packages"),
    ]
    for path in candidates:
        if path.exists():
            path_str = str(path)
            if path_str not in sys.path:
                sys.path.insert(0, path_str)


def _load_gstreamer_bindings() -> tuple[object | None, object | None]:
    """Import GStreamer bindings, with a venv fallback to system packages.

    OpenCV's GStreamer backend is not reliable in this project environment, so
    we keep a direct `gi.repository.Gst` path available as a second capture
    option. If the venv cannot import `gi`, we retry after exposing the system
    `dist-packages` paths.
    """
    try:
        import gi  # type: ignore

        gi.require_version("Gst", "1.0")
        from gi.repository import Gst  # type: ignore

        Gst.init(None)
        return gi, Gst
    except Exception:
        _add_system_site_packages()
        try:
            import gi  # type: ignore

            gi.require_version("Gst", "1.0")
            from gi.repository import Gst  # type: ignore

            Gst.init(None)
            return gi, Gst
        except Exception:
            return None, None


gi, Gst = _load_gstreamer_bindings()


def _build_qtiqmmfsrc_pipeline(source: int, width: int, height: int, fps: int) -> str:
    """Build the single capture pipeline supported by this runtime."""
    return (
        f"qtiqmmfsrc camera={source} ! "
        f"video/x-raw,format=NV12,width={width},height={height},framerate={fps}/1 ! "
        "queue max-size-buffers=1 leaky=downstream ! "
        "videoconvert ! video/x-raw,format=BGR ! "
        "appsink name=sink emit-signals=false sync=false max-buffers=1 drop=true"
    )


class _GStreamerAppsinkCapture:
    """Lightweight GStreamer capture backend using appsink."""

    def __init__(self, pipeline: str, width: int, height: int) -> None:
        if Gst is None:
            raise RuntimeError("GStreamer Python bindings are not available")

        self._pipeline = Gst.parse_launch(pipeline)
        self._sink = self._pipeline.get_by_name("sink")
        if self._sink is None:
            self._pipeline.set_state(Gst.State.NULL)
            raise RuntimeError("GStreamer pipeline does not expose an appsink named 'sink'")

        self._width = width
        self._height = height
        self._opened = False
        self._pipeline.set_state(Gst.State.PLAYING)
        state_change = self._pipeline.get_state(2 * Gst.SECOND)
        self._opened = state_change.state == Gst.State.PLAYING

    def isOpened(self) -> bool:
        return self._opened

    def read(self) -> tuple[bool, np.ndarray | None]:
        if not self._opened:
            return False, None

        sample = self._sink.emit("try-pull-sample", 2 * Gst.SECOND)
        if sample is None:
            return False, None

        buffer = sample.get_buffer()
        success, map_info = buffer.map(Gst.MapFlags.READ)
        if not success:
            return False, None

        try:
            frame = np.frombuffer(map_info.data, dtype=np.uint8)
            frame = frame.reshape((self._height, self._width, 3))
            return True, frame.copy()
        finally:
            buffer.unmap(map_info)

    def release(self) -> None:
        self._pipeline.set_state(Gst.State.NULL)
        self._opened = False


class CameraStream:
    """Stream frames from the RubikPi camera using a single GStreamer backend.

    The runtime uses `qtiqmmfsrc` + `appsink` because that is the path that
    works on the RubikPi device. `source` can be either:
    - an integer camera index, which builds the default RubikPi pipeline
    - a full GStreamer pipeline string
    - a pipeline string prefixed with `gst:` or `gstreamer:`
    """

    def __init__(self, source: int | str = 0, width: int = 640, height: int = 640, fps: int = 30) -> None:
        if Gst is None:
            raise RuntimeError(
                "GStreamer Python bindings are not available. "
                "Install the system GStreamer packages or expose them to the virtualenv."
            )

        if isinstance(source, str) and source.isdigit():
            source = int(source)

        self._source = source
        if isinstance(source, int):
            pipeline = _build_qtiqmmfsrc_pipeline(source, width, height, fps)
        else:
            pipeline = source
            for prefix in ("gst:", "gstreamer:"):
                if pipeline.startswith(prefix):
                    pipeline = pipeline[len(prefix) :].strip()
                    break

        self._capture = _GStreamerAppsinkCapture(pipeline, width, height)
        if not self._capture.isOpened():
            raise RuntimeError(
                f"Unable to open GStreamer camera pipeline: {pipeline}. "
                "Check qtiqmmfsrc, appsink, and device access."
            )

        self._frame_index = 0

    def frames(self) -> Generator[tuple[int, float, np.ndarray], None, None]:
        """Yield frames continuously until capture fails or is closed."""
        while True:
            ok, frame = self._capture.read()
            if not ok:
                if self._frame_index == 0:
                    raise RuntimeError(
                        f"Unable to read a frame from camera source: {self._source}. "
                        "Check that qtiqmmfsrc and appsink are available and the camera is accessible."
                    )
                break

            ts = time.time()
            idx = self._frame_index
            self._frame_index += 1
            yield idx, ts, frame

    def close(self) -> None:
        """Release camera resources."""
        self._capture.release()

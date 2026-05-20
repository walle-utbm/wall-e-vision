from __future__ import annotations

"""Camera factory and backends for the unified runtime."""

import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Generator, Protocol

import cv2
import numpy as np

from ..config import CameraSettings


def _add_system_site_packages() -> None:
    candidates = [
        Path("/usr/lib/python3/dist-packages"),
        Path("/usr/lib/python3.13/dist-packages"),
        Path("/usr/local/lib/python3.13/dist-packages"),
        Path("/usr/lib/aarch64-linux-gnu/python3.12/site-packages"),
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


def _load_gstreamer() -> tuple[object | None, object | None]:
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


gi, Gst = _load_gstreamer()


class CameraBackend(Protocol):
    def frames(self) -> Generator[tuple[int, float, np.ndarray], None, None]:
        ...

    def close(self) -> None:
        ...


@dataclass(slots=True)
class _NullCamera:
    source: int | str

    def frames(self) -> Generator[tuple[int, float, np.ndarray], None, None]:
        raise RuntimeError("Camera type 'none' cannot produce frames")
        yield 0, 0.0, np.zeros((1, 1, 3), dtype=np.uint8)

    def close(self) -> None:
        return None


class _OpenCVCamera:
    def __init__(self, source: int | str, width: int, height: int, fps: int) -> None:
        self._source = source
        self._backend = cv2.VideoCapture(source, cv2.CAP_V4L2)
        if not self._backend.isOpened():
            self._backend = cv2.VideoCapture(source)

        self._backend.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._backend.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self._backend.set(cv2.CAP_PROP_FPS, fps)
        self._backend.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self._set_if_supported(cv2.CAP_PROP_AUTOFOCUS, 0)
        self._set_if_supported(cv2.CAP_PROP_AUTO_WB, 0)
        self._set_if_supported(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
        self._frame_index = 0

        if not self._backend.isOpened():
            raise RuntimeError(f"Unable to open camera source: {source}")

    def _set_if_supported(self, prop_id: int, value: float) -> None:
        try:
            self._backend.set(prop_id, value)
        except Exception:
            pass

    def frames(self) -> Generator[tuple[int, float, np.ndarray], None, None]:
        while True:
            ok, frame = self._backend.read()
            if not ok:
                if self._frame_index == 0:
                    raise RuntimeError(f"Unable to read a frame from camera source: {self._source}")
                break
            timestamp = time.time()
            frame_index = self._frame_index
            self._frame_index += 1
            yield frame_index, timestamp, frame

    def close(self) -> None:
        self._backend.release()


class _Picamera2Camera:
    def __init__(self, source: int | str, width: int, height: int, fps: int, warmup_frames: int) -> None:
        if Picamera2 is None:
            raise RuntimeError("picamera2 is not available")
        self._source = source
        self._camera = Picamera2(source if isinstance(source, int) else 0)
        config = self._camera.create_video_configuration(
            main={"size": (width, height), "format": "BGR888"},
            buffer_count=2,
        )
        self._camera.configure(config)
        self._camera.start()
        for _ in range(max(0, warmup_frames)):
            _ = self._camera.capture_array()
            time.sleep(0.1)
        self._frame_index = 0

    def frames(self) -> Generator[tuple[int, float, np.ndarray], None, None]:
        while True:
            frame = self._camera.capture_array()
            if frame is None:
                if self._frame_index == 0:
                    raise RuntimeError(f"Unable to read a frame from camera source: {self._source}")
                break
            if frame.ndim == 3:
                if frame.shape[2] == 4:
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
                elif frame.shape[2] == 3:
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            time.sleep(0.01)
            timestamp = time.time()
            frame_index = self._frame_index
            self._frame_index += 1
            yield frame_index, timestamp, frame

    def close(self) -> None:
        self._camera.stop()


class _GStreamerCamera:
    def __init__(self, source: int | str, width: int, height: int, fps: int) -> None:
        if Gst is None:
            raise RuntimeError("GStreamer Python bindings are not available")
        self._source = source
        if isinstance(source, str) and source.isdigit():
            source = int(source)
        if isinstance(source, int):
            pipeline = (
                f"qtiqmmfsrc camera={source} ! "
                f"video/x-raw,format=NV12,width={width},height={height},framerate={fps}/1 ! "
                "queue max-size-buffers=1 leaky=downstream ! "
                "videoconvert ! video/x-raw,format=BGR ! "
                "appsink name=sink emit-signals=false sync=false max-buffers=1 drop=true"
            )
        else:
            pipeline = source
            for prefix in ("gst:", "gstreamer:"):
                if pipeline.startswith(prefix):
                    pipeline = pipeline[len(prefix) :].strip()
                    break

        self._pipeline = Gst.parse_launch(pipeline)
        self._sink = self._pipeline.get_by_name("sink")
        if self._sink is None:
            self._pipeline.set_state(Gst.State.NULL)
            raise RuntimeError("GStreamer pipeline does not expose an appsink named 'sink'")

        self._width = width
        self._height = height
        self._frame_index = 0
        self._pipeline.set_state(Gst.State.PLAYING)
        state_change = self._pipeline.get_state(2 * Gst.SECOND)
        self._opened = state_change.state == Gst.State.PLAYING
        if not self._opened:
            self._pipeline.set_state(Gst.State.NULL)
            raise RuntimeError(f"Unable to open GStreamer camera pipeline: {pipeline}")

    def frames(self) -> Generator[tuple[int, float, np.ndarray], None, None]:
        while True:
            sample = self._sink.emit("try-pull-sample", 2 * Gst.SECOND)
            if sample is None:
                if self._frame_index == 0:
                    raise RuntimeError(f"Unable to read a frame from camera source: {self._source}")
                break

            buffer = sample.get_buffer()
            success, map_info = buffer.map(Gst.MapFlags.READ)
            if not success:
                break
            try:
                frame = np.frombuffer(map_info.data, dtype=np.uint8)
                frame = frame.reshape((self._height, self._width, 3)).copy()
            finally:
                buffer.unmap(map_info)

            timestamp = time.time()
            frame_index = self._frame_index
            self._frame_index += 1
            yield frame_index, timestamp, frame

    def close(self) -> None:
        self._pipeline.set_state(Gst.State.NULL)


class CameraStream:
    def __init__(self, backend: CameraBackend) -> None:
        self._backend = backend

    @classmethod
    def create(cls, hardware: str, camera_type: str, settings: CameraSettings) -> CameraStream:
        if camera_type == "none":
            backend: CameraBackend = _NullCamera(settings.source)
        elif camera_type == "usb":
            backend = _OpenCVCamera(settings.source, settings.width, settings.height, settings.fps)
        elif camera_type == "picamera" and hardware == "rubikpi3":
            backend = _GStreamerCamera(settings.source, settings.width, settings.height, settings.fps)
        elif camera_type == "picamera" and Picamera2 is not None:
            backend = _Picamera2Camera(settings.source, settings.width, settings.height, settings.fps, settings.warmup_frames)
        else:
            backend = _OpenCVCamera(settings.source, settings.width, settings.height, settings.fps)
        return cls(backend)

    def frames(self) -> Generator[tuple[int, float, np.ndarray], None, None]:
        yield from self._backend.frames()

    def close(self) -> None:
        self._backend.close()


class CameraFactory:
    @staticmethod
    def create(hardware: str, camera_type: str, settings: CameraSettings) -> CameraStream:
        return CameraStream.create(hardware, camera_type, settings)
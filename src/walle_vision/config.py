from __future__ import annotations

"""Configuration loader for the unified runtime."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


VALID_HARDWARE = {"rpi4", "rubikpi3", "pc"}
VALID_CAMERA_TYPES = {"picamera", "usb", "none"}
VALID_MODES = {"edge_standalone", "stream_client", "stream_server"}


def _as_int(value: Any, default: int) -> int:
    if value is None:
        return default
    return int(value)


def _as_float(value: Any, default: float) -> float:
    if value is None:
        return default
    return float(value)


def _as_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    return bool(value)


def _as_str(value: Any, default: str) -> str:
    if value is None:
        return default
    return str(value)


def _as_source(value: Any) -> int | str:
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    if value is None:
        return 0
    return str(value)


def _resolve_hardware_section(data: dict[str, Any], hardware: str) -> dict[str, Any]:
    resolved = {key: value for key, value in data.items() if key not in VALID_HARDWARE}
    hardware_specific = data.get(hardware, {}) or {}
    if not isinstance(hardware_specific, dict):
        raise TypeError(f"Expected a mapping for '{hardware}' hardware settings")
    resolved.update(hardware_specific)
    return resolved


@dataclass(slots=True)
class PathSettings:
    model_dir: Path = Path("model")
    output_dir: Path = Path("outputs")


@dataclass(slots=True)
class CameraSettings:
    source: int | str = 0
    width: int = 640
    height: int = 640
    fps: int = 30
    warmup_frames: int = 3


@dataclass(slots=True)
class DetectorSettings:
    conf_threshold: float = 0.30
    iou_threshold: float = 0.45
    image_size: int = 640
    max_detections: int = 8
    use_half: bool = False
    device: str = "cpu"
    workers: int = 0
    force_pytorch: bool = False


@dataclass(slots=True)
class RuntimeSettings:
    show: bool = False
    infer_every_n_frames: int = 1
    save_every_n_frames: int = 10
    camera_test_mode: bool = False
    camera_test_interval_sec: float = 5.0


@dataclass(slots=True)
class NetworkSettings:
    host: str = "0.0.0.0"
    port: int = 5000
    client_host: str = "192.168.137.1"
    client_port: int = 5000
    jpeg_quality: int = 80
    reconnect_delay_sec: float = 2.0
    max_inflight_frames: int = 2


@dataclass(slots=True)
class AppConfig:
    hardware: str
    camera_type: str
    mode: str
    paths: PathSettings = field(default_factory=PathSettings)
    model_paths: dict[str, str] = field(default_factory=dict)
    camera: CameraSettings = field(default_factory=CameraSettings)
    detector: DetectorSettings = field(default_factory=DetectorSettings)
    runtime: RuntimeSettings = field(default_factory=RuntimeSettings)
    network: NetworkSettings = field(default_factory=NetworkSettings)
    config_dir: Path = Path(".")

    def __post_init__(self) -> None:
        if self.hardware not in VALID_HARDWARE:
            raise ValueError(f"Unsupported hardware '{self.hardware}'")
        if self.camera_type not in VALID_CAMERA_TYPES:
            raise ValueError(f"Unsupported camera_type '{self.camera_type}'")
        if self.mode not in VALID_MODES:
            raise ValueError(f"Unsupported mode '{self.mode}'")

    def resolve_model_path(self) -> Path:
        model_name = self.model_paths.get(self.hardware) or self.model_paths.get("default")
        if model_name is None:
            raise KeyError(f"No model configured for hardware '{self.hardware}'")

        candidate = Path(model_name)
        if not candidate.is_absolute():
            candidate = (self.config_dir / self.paths.model_dir / candidate).resolve()
        return candidate


def _build_camera_settings(data: dict[str, Any]) -> CameraSettings:
    return CameraSettings(
        source=_as_source(data.get("source", 0)),
        width=_as_int(data.get("width"), 640),
        height=_as_int(data.get("height"), 640),
        fps=_as_int(data.get("fps"), 30),
        warmup_frames=_as_int(data.get("warmup_frames"), 3),
    )


def _build_detector_settings(data: dict[str, Any]) -> DetectorSettings:
    return DetectorSettings(
        conf_threshold=_as_float(data.get("conf_threshold"), 0.30),
        iou_threshold=_as_float(data.get("iou_threshold"), 0.45),
        image_size=_as_int(data.get("image_size"), 640),
        max_detections=_as_int(data.get("max_detections"), 8),
        use_half=_as_bool(data.get("use_half"), False),
        device=_as_str(data.get("device"), "cpu"),
        workers=_as_int(data.get("workers"), 0),
        force_pytorch=_as_bool(data.get("force_pytorch"), False),
    )


def _build_runtime_settings(data: dict[str, Any]) -> RuntimeSettings:
    return RuntimeSettings(
        show=_as_bool(data.get("show"), False),
        infer_every_n_frames=_as_int(data.get("infer_every_n_frames"), 1),
        save_every_n_frames=_as_int(data.get("save_every_n_frames"), 10),
        camera_test_mode=_as_bool(data.get("camera_test_mode"), False),
        camera_test_interval_sec=_as_float(data.get("camera_test_interval_sec"), 5.0),
    )


def _build_network_settings(data: dict[str, Any]) -> NetworkSettings:
    return NetworkSettings(
        host=_as_str(data.get("host"), "0.0.0.0"),
        port=_as_int(data.get("port"), 5000),
        client_host=_as_str(data.get("client_host"), "192.168.137.1"),
        client_port=_as_int(data.get("client_port"), 5000),
        jpeg_quality=_as_int(data.get("jpeg_quality"), 80),
        reconnect_delay_sec=_as_float(data.get("reconnect_delay_sec"), 2.0),
        max_inflight_frames=_as_int(data.get("max_inflight_frames"), 2),
    )


def load_config(config_path: Path) -> AppConfig:
    with config_path.open("r", encoding="utf-8") as file:
        raw = yaml.safe_load(file) or {}

    paths_raw = raw.get("paths", {}) or {}
    model_paths = raw.get("models", {}) or {}

    hardware = _as_str(raw.get("hardware"), "pc")
    camera_raw = _resolve_hardware_section(raw.get("camera", {}) or {}, hardware)
    detector_raw = _resolve_hardware_section(raw.get("detector", {}) or {}, hardware)

    return AppConfig(
        hardware=hardware,
        camera_type=_as_str(raw.get("camera_type"), "none"),
        mode=_as_str(raw.get("mode"), "edge_standalone"),
        paths=PathSettings(
            model_dir=Path(_as_str(paths_raw.get("model_dir"), "model")),
            output_dir=Path(_as_str(paths_raw.get("output_dir"), "outputs")),
        ),
        model_paths={str(key): str(value) for key, value in model_paths.items()},
        camera=_build_camera_settings(camera_raw),
        detector=_build_detector_settings(detector_raw),
        runtime=_build_runtime_settings(raw.get("runtime", {}) or {}),
        network=_build_network_settings(raw.get("network", {}) or {}),
        config_dir=config_path.resolve().parent,
    )
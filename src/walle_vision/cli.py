from __future__ import annotations

"""RubikPi runtime configuration.

This branch is simplified to a single RubikPi 3 profile with PyTorch inference.
Run `python src/main.py` to start the pipeline.
"""

from dataclasses import dataclass
from pathlib import Path

from .detector import DetectorConfig
from .pipeline import VisionPipeline


@dataclass(slots=True)
class RubikPiRuntimeConfig:
    """Runtime profile optimized for RubikPi 3 (Snapdragon 8 Gen1 + IMX708)."""

    model: str = "model/best.onnx"
    source: str = (
        "gstreamer:qtiqmmfsrc camera=0 ! "
        "video/x-raw,format=NV12,width=1280,height=720,framerate=30/1 ! "
        "queue max-size-buffers=1 leaky=downstream ! "
        "videoconvert ! video/x-raw,format=BGR ! appsink name=sink emit-signals=false sync=false max-buffers=1 drop=true"
    )
    output_dir: str = "outputs"

    conf: float = 0.10
    iou: float = 0.45
    imgsz: int = 512
    max_det: int = 12

    width: int = 1280
    height: int = 720
    fps: int = 30

    infer_every: int = 2
    save_every: int = 10

    track_iou: float = 0.35
    confirm_frames: int = 2
    max_missed: int = 3
    track_window: int = 5
    display_persist: int = 0

    show: bool = False
    half: bool = False
    camera_test_mode: bool = True
    camera_test_interval_sec: float = 5.0


def _parse_source(source: str) -> int | str:
    """Convert numeric camera sources to integers."""
    if source.isdigit():
        return int(source)
    return source


def run() -> None:
    """Run the RubikPi vision pipeline."""
    cfg = RubikPiRuntimeConfig()

    model_path = Path(cfg.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    detector_cfg = DetectorConfig(
        model_path=str(model_path),
        conf_threshold=cfg.conf,
        iou_threshold=cfg.iou,
        image_size=cfg.imgsz,
        max_detections=cfg.max_det,
        use_half=cfg.half,
    )

    pipeline = VisionPipeline(
        detector_cfg=detector_cfg,
        source=_parse_source(cfg.source),
        output_dir=cfg.output_dir,
        show=cfg.show,
        infer_every_n_frames=cfg.infer_every,
        save_every_n_frames=cfg.save_every,
        track_iou_threshold=cfg.track_iou,
        track_confirm_frames=cfg.confirm_frames,
        track_max_missed_frames=cfg.max_missed,
        track_confidence_window=cfg.track_window,
        display_persist_frames=cfg.display_persist,
        width=cfg.width,
        height=cfg.height,
        fps=cfg.fps,
        camera_test_mode=cfg.camera_test_mode,
        camera_test_interval_sec=cfg.camera_test_interval_sec,
    )
    pipeline.run()

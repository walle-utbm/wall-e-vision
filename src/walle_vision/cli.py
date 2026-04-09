from __future__ import annotations

"""Single-entry runtime configuration for Raspberry Pi (2 GB RAM).

This module intentionally removes command-line tuning complexity.
Run `python src/main.py` and the application starts with one curated profile.
If you need to tweak values later, edit the `PiRuntimeConfig` defaults below.
"""

from dataclasses import dataclass
from pathlib import Path

from .detector import DetectorConfig
from .pipeline import VisionPipeline


@dataclass(slots=True)
class PiRuntimeConfig:
    """Fixed runtime profile optimized for a low-memory Raspberry Pi.

    Attributes:
        model: Path to model weights.
        source: Camera index as string (converted to int when numeric).
        output_dir: Folder for detections and saved frames.
        conf: Detection confidence threshold.
        iou: NMS IoU threshold.
        imgsz: Inference size. 640 matches training geometry for better precision.
        max_det: Maximum detections per frame.
        width: Camera capture width.
        height: Camera capture height.
        fps: Requested camera FPS.
        infer_every: Run inference every N frames.
        save_every: Save one annotated image every N stable detections.
        track_iou: IoU threshold for temporal association.
        confirm_frames: Minimum matched frames before confirming object.
        max_missed: Allowed missed frames before dropping a track.
        track_window: Number of recent confidences for smoothing.
        display_persist: Frames to keep last confirmed box on display.
        show: Enable OpenCV window (typically False on Raspberry Pi).
        half: Use FP16 when supported.
    """

    model: str = "model/best.pt"
    source: str = "0"
    output_dir: str = "outputs"

    conf: float = 0.25
    iou: float = 0.45
    imgsz: int = 640
    max_det: int = 6

    width: int = 640
    height: int = 640
    fps: int = 25

    infer_every: int = 3
    save_every: int = 4

    track_iou: float = 0.3
    confirm_frames: int = 3
    max_missed: int = 2
    track_window: int = 5
    display_persist: int = 5

    show: bool = False
    half: bool = False


def _parse_source(source: str) -> int | str:
    """Convert camera source to int when numeric, else keep string path/url."""
    return int(source) if source.isdigit() else source


def run() -> None:
    """Run the vision pipeline with the fixed Raspberry profile."""
    cfg = PiRuntimeConfig()

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
    )
    pipeline.run()

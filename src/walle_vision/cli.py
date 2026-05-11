from __future__ import annotations

"""Single-entry runtime configuration for Raspberry Pi (8 GB RAM + IMX708).

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
    """Fixed runtime profile optimized for Raspberry Pi 8GB + IMX708 camera.

    Inference resolution matches training geometry (640×640) for best accuracy.
    Optimized for real-time performance on RPi 4/5 with 8GB RAM.

    Attributes:
        model: Path to YOLO model weights.
        source: Camera index as string (0 for IMX708 on RPi, or path for video).
        output_dir: Output folder for detections.jsonl and frames/.
        conf: Detection confidence threshold (0.30 = good balance precision/recall).
        iou: NMS IoU threshold for duplicate suppression.
        imgsz: Inference size (640 = training geometry for best accuracy).
        max_det: Maximum detections per frame (8 is reasonable for waste bins).
        width: Camera capture width (640 matches model training).
        height: Camera capture height (640 matches model training).
        fps: Camera FPS (30 for smooth real-time on 8GB).
        infer_every: Run inference every N frames (1=every frame, 2=skip one).
        save_every: Save annotated frame every N stable detections.
        track_iou: IoU threshold for temporal track association.
        confirm_frames: Frames needed to confirm a track (3 = more stable).
        max_missed: Frames to keep track before dropping it.
        track_window: Frames for confidence smoothing.
        display_persist: Frames to persist visualization of detection.
        show: Enable OpenCV window display (False for headless RPi).
        half: Use FP16 inference when GPU available (faster on RPi).
        camera_test_mode: Save raw camera frames periodically for SSH verification.
        camera_test_interval_sec: Interval in seconds between camera test frame saves.
        force_pytorch: Force PyTorch inference, skip NCNN for debugging (default False).
    """

    model: str = "model/best.pt"
    source: str = "0"
    output_dir: str = "outputs"

    conf: float = 0.10
    iou: float = 0.45
    imgsz: int = 640
    max_det: int = 8

    width: int = 640
    height: int = 640
    fps: int = 30

    infer_every: int = 1
    save_every: int = 5

    track_iou: float = 0.35
    confirm_frames: int = 3
    max_missed: int = 2
    track_window: int = 5
    display_persist: int = 5

    show: bool = False
    half: bool = True
    camera_test_mode: bool = False
    camera_test_interval_sec: float = 5.0    
    force_pytorch: bool = False

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
        force_pytorch=cfg.force_pytorch,
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

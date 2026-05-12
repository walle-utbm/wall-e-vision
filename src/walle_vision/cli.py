from __future__ import annotations

"""Runtime configuration profiles for different hardware platforms.

This module provides optimized profiles for:
- Raspberry Pi 4/5 (8GB + IMX708): Limited CPU, NCNN acceleration recommended
- RubikPi 3 (Snapdragon 8 Gen1 + IMX708): Powerful CPU, PyTorch sufficient

Run `python src/main.py` and the application auto-detects your hardware.
If you need to override, set WALLE_PLATFORM environment variable.
"""

import os
import platform
from dataclasses import dataclass
from pathlib import Path

from .detector import DetectorConfig
from .pipeline import VisionPipeline


@dataclass(slots=True)
class PiRuntimeConfig:
    """Fixed runtime profile optimized for Raspberry Pi 4/5 (8GB + IMX708).

    Inference resolution matches training geometry (640×640) for best accuracy.
    Conservative settings due to ARM Cortex-A72 CPU limitation.

    Attributes:
        model: Path to YOLO model weights (.pt or NCNN .param).
        source: Camera index as string (0 for IMX708 on RPi, or path for video).
        output_dir: Output folder for detections.jsonl and frames/.
        conf: Detection confidence threshold (0.10 = high recall for waste detection).
        iou: NMS IoU threshold for duplicate suppression.
        imgsz: Inference size (640 = training geometry).
        max_det: Maximum detections per frame (8 for bins).
        width: Camera capture width (640 matches training).
        height: Camera capture height (640 matches training).
        fps: Camera FPS (30 = smooth real-time on 8GB RPi).
        infer_every: Run inference every N frames (1 = every frame on RPi4).
        save_every: Save annotated frame every N stable detections.
        track_iou: IoU threshold for temporal track association.
        confirm_frames: Frames needed to confirm a track (3 = stable).
        max_missed: Frames to keep track before dropping.
        track_window: Frames for confidence smoothing.
        display_persist: Frames to persist visualization.
        show: Enable OpenCV window display (False for headless RPi).
        half: Use FP16 inference when available.
        camera_test_mode: Save raw camera frames for SSH verification.
        camera_test_interval_sec: Interval between test frames.
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


@dataclass(slots=True)
class RubikPiRuntimeConfig:
    """Fixed runtime profile optimized for RubikPi 3 (Snapdragon 8 Gen1 + IMX708).

    Aggressive settings exploiting 8 powerful ARM cores (A78 @ 2.7GHz).
    No NCNN needed—PyTorch inference is fast enough.

    Attributes:
        model: Path to YOLO model weights (.pt only, no NCNN).
        source: Camera index as string (0 for IMX708, or path for video).
        output_dir: Output folder for detections.jsonl and frames/.
        conf: Detection confidence threshold (0.10 = high recall).
        iou: NMS IoU threshold for duplicate suppression.
        imgsz: Inference size (768 = higher resolution for better accuracy).
        max_det: Maximum detections per frame (12 = more objects).
        width: Camera capture width (768 for higher quality).
        height: Camera capture height (768 for higher quality).
        fps: Camera FPS (60 = smooth, CPU can handle it).
        infer_every: Run inference every N frames (1 = every frame).
        save_every: Save annotated frame every N stable detections.
        track_iou: IoU threshold for temporal track association.
        confirm_frames: Frames needed to confirm a track (2 = faster confirmation).
        max_missed: Frames to keep track before dropping.
        track_window: Frames for confidence smoothing.
        display_persist: Frames to persist visualization.
        show: Enable OpenCV window display (False for headless RubikPi).
        half: Use FP16 inference if CPU supports it.
        camera_test_mode: Save raw camera frames for SSH verification.
        camera_test_interval_sec: Interval between test frames.
    """

    model: str = "model/best.pt"
    source: str = "0"
    output_dir: str = "outputs"

    conf: float = 0.10
    iou: float = 0.45
    imgsz: int = 768
    max_det: int = 12

    width: int = 768
    height: int = 768
    fps: int = 60

    infer_every: int = 1
    save_every: int = 3

    track_iou: float = 0.35
    confirm_frames: int = 2
    max_missed: int = 3
    track_window: int = 5
    display_persist: int = 5

    show: bool = False
    half: bool = True
    camera_test_mode: bool = False
    camera_test_interval_sec: float = 5.0


def _detect_platform() -> str:
    """Auto-detect platform (raspberry_pi or rubik_pi).
    
    Can be overridden by WALLE_PLATFORM environment variable.
    """
    override = os.environ.get("WALLE_PLATFORM", "").lower().strip()
    if override in {"raspberry_pi", "rubik_pi"}:
        return override
    
    # Try to detect from device-tree or model
    model_file = Path("/proc/device-tree/model")
    if model_file.exists():
        try:
            model_name = model_file.read_text().lower()
            if "rubik" in model_name:
                return "rubik_pi"
            if "raspberry" in model_name or "bcm" in model_name:
                return "raspberry_pi"
        except Exception:
            pass
    
    # Fallback: check CPU model
    try:
        cpuinfo = Path("/proc/cpuinfo").read_text().lower()
        if "snapdragon" in cpuinfo or "cortex-a78" in cpuinfo:
            return "rubik_pi"
    except Exception:
        pass
    
    # Default to RPi4 (safer, more conservative)
    return "raspberry_pi"


def _parse_source(source: str) -> int | str:
    """Convert camera source to int when numeric, else keep string path/url."""
    return int(source) if source.isdigit() else source


def run() -> None:
    """Run the vision pipeline with auto-detected hardware profile.
    
    Detects platform and uses appropriate config:
    - Raspberry Pi 4/5: PiRuntimeConfig (conservative)
    - RubikPi 3: RubikPiRuntimeConfig (aggressive)
    
    Override with: WALLE_PLATFORM=rubik_pi or WALLE_PLATFORM=raspberry_pi
    """
    platform_detected = _detect_platform()
    
    if platform_detected == "rubik_pi":
        print(f"✅ Detected: RubikPi 3 (Snapdragon)")
        cfg = RubikPiRuntimeConfig()
    else:
        print(f"✅ Detected: Raspberry Pi 4/5")
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
        camera_test_mode=cfg.camera_test_mode,
        camera_test_interval_sec=cfg.camera_test_interval_sec,
    )
    pipeline.run()

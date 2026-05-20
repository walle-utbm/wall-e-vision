from __future__ import annotations

"""Single-entry runtime configuration for the Raspberry Pi sender branch.

This branch captures frames on the Raspberry Pi and streams them to a PC for
inference. The PC sends results back on the same connection, and the Raspberry
Pi stores a compact local log.
"""

from dataclasses import dataclass

from .raspberry_pipeline import RaspberryVisionPipeline


@dataclass(slots=True)
class PiRuntimeConfig:
    """Fixed runtime profile for the Raspberry Pi streaming sender.

    The Raspberry Pi only captures, compresses, and forwards frames. All model
    inference and detection post-processing happen on the PC side.
    """

    source: str = "0"
    output_dir: str = "outputs"

    width: int = 1280
    height: int = 720
    fps: int = 30

    show: bool = False
    pc_host: str = "192.168.137.1"
    pc_port: int = 5000

    jpeg_quality: int = 80
    reconnect_delay_sec: float = 2.0
    max_inflight_frames: int = 2

    camera_test_mode: bool = True
    camera_test_interval_sec: float = 5.0

def _parse_source(source: str) -> int | str:
    """Convert camera source to int when numeric, else keep string path/url."""
    return int(source) if source.isdigit() else source


def run() -> None:
    """Run the Raspberry Pi streaming pipeline."""
    cfg = PiRuntimeConfig()

    pipeline = RaspberryVisionPipeline(
        source=_parse_source(cfg.source),
        output_dir=cfg.output_dir,
        show=cfg.show,
        width=cfg.width,
        height=cfg.height,
        fps=cfg.fps,
        host=cfg.pc_host,
        port=cfg.pc_port,
        jpeg_quality=cfg.jpeg_quality,
        reconnect_delay_sec=cfg.reconnect_delay_sec,
        max_inflight_frames=cfg.max_inflight_frames,
        camera_test_mode=cfg.camera_test_mode,
        camera_test_interval_sec=cfg.camera_test_interval_sec,
    )
    pipeline.run()

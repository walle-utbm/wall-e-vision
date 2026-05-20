"""Wall-E vision package for real-time waste detection and sorting.

Modules:
    camera: webcam/video source capture.
    cli: single RubikPi runtime profile entrypoint.
    detector: YOLO inference and pickup point computation.
    labels: class names and recycle-bin mapping.
    pipeline: orchestration of capture, inference, tracking, and export.
    sorting: business mapping class -> recycle bin.
    tracking: temporal smoothing and track confirmation.
    types: shared typed dataclasses.
    visualization: image annotation utilities.
"""

__all__ = [
    "camera",
    "detector",
    "cli",
    "labels",
    "pipeline",
    "sorting",
    "tracking",
    "types",
    "visualization",
]

"""Wall-E vision package for real-time waste detection and sorting.

Modules:
    camera: webcam/video source capture.
    cli: Raspberry streaming runtime entrypoint.
    detector: YOLO inference and pickup point computation (PC-side branch).
    labels: class names and recycle-bin mapping.
    pipeline: local orchestration of capture, inference, tracking, and export.
    raspberry_pipeline: Raspberry Pi frame streaming client.
    transport: TCP frame/result protocol helpers.
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
    "raspberry_pipeline",
    "sorting",
    "transport",
    "tracking",
    "types",
    "visualization",
]

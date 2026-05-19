"""PC-side streaming server for wall-e-vision.

This subpackage receives frames from the Raspberry Pi, runs YOLO inference,
and sends structured results back over the same TCP connection.
"""

from .detector import DetectorConfig, WasteDetector
from .server import PCRuntimeConfig, PCVisionServer, run

__all__ = ["DetectorConfig", "WasteDetector", "PCRuntimeConfig", "PCVisionServer", "run"]
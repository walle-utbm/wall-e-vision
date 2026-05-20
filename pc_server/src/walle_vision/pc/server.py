from __future__ import annotations

"""PC-side TCP server that receives frames, runs inference, and replies."""

import json
import os
import socket
import signal
import threading
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .detector import DetectorConfig, WasteDetector
from .tracking import TemporalDetectionTracker
from .transport import StreamConnection, TransportMessage
from .types import Detection, FrameResult
from .visualization import draw_detections


@dataclass(slots=True)
class PCRuntimeConfig:
    """Fixed runtime profile for the PC inference server."""

    host: str = "0.0.0.0"
    port: int = 5000

    model: str = "model/best.pt"
    output_dir: str = "outputs"

    conf: float = 0.10
    iou: float = 0.45
    imgsz: tuple[int, int] = (1280, 720)
    max_det: int = 8

    track_iou: float = 0.35
    confirm_frames: int = 3
    max_missed: int = 2
    track_window: int = 5

    show: bool = False
    device: str = "cpu"


class PCVisionServer:
    """Receive frames from the Raspberry Pi and return structured results."""

    def __init__(self, cfg: PCRuntimeConfig) -> None:
        self.cfg = cfg
        self.detector = WasteDetector(
            DetectorConfig(
                model_path=cfg.model,
                conf_threshold=cfg.conf,
                iou_threshold=cfg.iou,
                image_size=cfg.imgsz,
                max_detections=cfg.max_det,
                device=cfg.device,
            )
        )
        self.tracker = TemporalDetectionTracker(
            iou_threshold=cfg.track_iou,
            confirm_frames=cfg.confirm_frames,
            max_missed_frames=cfg.max_missed,
            confidence_window=cfg.track_window,
        )

        self.output_dir = Path(cfg.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results_jsonl = self.output_dir / "detections.jsonl"
        self.predict_dir = self.output_dir / "predict"
        self.predict_dir.mkdir(parents=True, exist_ok=True)
        # Event used to request shutdown from signal handler or other threads
        self.stop_event = threading.Event()

    def run(self) -> None:
        """Start the TCP server and process clients until interrupted."""
        # Install signal handlers to request graceful shutdown
        def _signal_handler(sig, frame):
            print("Signal received, shutting down server...")
            self.stop_event.set()

        signal.signal(signal.SIGINT, _signal_handler)
        try:
            signal.signal(signal.SIGTERM, _signal_handler)
        except Exception:
            # SIGTERM may not be available on some platforms (e.g. Windows)
            pass

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
            server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server_socket.bind((self.cfg.host, self.cfg.port))
            server_socket.listen(1)
            server_socket.settimeout(1.0)  # make accept interruptible
            print(f"PC server listening on {self.cfg.host}:{self.cfg.port}")

            try:
                while not self.stop_event.is_set():
                    print("Waiting for Raspberry Pi client...")
                    try:
                        client_socket, address = server_socket.accept()
                    except socket.timeout:
                        continue
                    except OSError:
                        break

                    print(f"Client connected from {address[0]}:{address[1]}")
                    # make client socket receive calls timeout so we can exit promptly
                    try:
                        client_socket.settimeout(1.0)
                    except Exception:
                        pass

                    with client_socket:
                        # handle client until it disconnects or stop_event is set
                        self._handle_client(StreamConnection(client_socket))
            finally:
                print("PC server stopped")
                try:
                    cv2.destroyAllWindows()
                except Exception:
                    pass

    def _handle_client(self, connection: StreamConnection) -> None:
        """Process a connected Raspberry Pi client."""
        try:
            while not self.stop_event.is_set():
                try:
                    message = connection.receive_message()
                except socket.timeout:
                    # check stop_event periodically
                    continue
                if message.header.get("kind") != "frame":
                    continue

                frame_index = int(message.header.get("frame_index", 0))
                timestamp = float(message.header.get("timestamp", 0.0))
                frame = self._decode_frame(message)
                if frame is None:
                    continue

                raw_detections = self.detector.infer(frame)
                stable_detections = self.tracker.update(raw_detections)
                self._store_results(frame_index, timestamp, frame, stable_detections)

                response = FrameResult(frame_index=frame_index, timestamp=timestamp, detections=stable_detections)
                payload = self._frame_result_to_payload(response)
                connection.send_packet(
                    {"kind": "result", "frame_index": frame_index, "timestamp": timestamp},
                    json.dumps(payload, ensure_ascii=True).encode("utf-8"),
                )
        except ConnectionError:
            print("Client disconnected")
        except KeyboardInterrupt:
            print("PC client session interrupted")
        except Exception as exc:
            # unexpected errors should be logged but shouldn't crash the server
            print(f"Client handler error: {exc}")
        finally:
            connection.close()

    def _decode_frame(self, message: TransportMessage) -> np.ndarray | None:
        """Decode one JPEG frame received from the Raspberry Pi."""
        if message.header.get("encoding") != "jpeg":
            return None

        buffer = np.frombuffer(message.body, dtype=np.uint8)
        frame = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
        if frame is None:
            print("⚠️ Unable to decode received JPEG frame")
        return frame

    def _store_results(self, frame_index: int, timestamp: float, frame: np.ndarray, detections: list[Detection]) -> None:
        """Persist stable detections as JSONL and annotated frames."""
        if not detections:
            return

        payload = self._frame_result_to_payload(FrameResult(frame_index=frame_index, timestamp=timestamp, detections=detections))
        with self.results_jsonl.open("a", encoding="utf-8") as file:
            file.write(json.dumps(payload, ensure_ascii=True) + os.linesep)

        annotated = draw_detections(frame, detections)
        out_file = self.predict_dir / f"frame_{frame_index:06d}.jpg"
        cv2.imwrite(str(out_file), annotated)

        if self.cfg.show:
            cv2.imshow("wall-e-vision-pc", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                raise KeyboardInterrupt

    def _frame_result_to_payload(self, frame_result: FrameResult) -> dict[str, object]:
        """Convert a frame result into a JSON-serializable payload."""
        return {
            "frame_index": frame_result.frame_index,
            "timestamp": frame_result.timestamp,
            "detections": [
                {
                    "class_id": detection.class_id,
                    "class_name": detection.class_name,
                    "recycle_bin": detection.recycle_bin,
                    "confidence": round(detection.confidence, 5),
                    "bbox_xyxy": list(detection.bbox),
                    "center_xy": list(detection.center),
                    "pickup_xy": list(detection.pickup_point),
                    "area_ratio": round(detection.area_ratio, 5),
                    "mask_area_ratio": round(detection.mask_area_ratio, 5),
                    "segmentation_available": detection.segmentation_available,
                    "bbox_clipped": detection.bbox_clipped,
                    "track_id": detection.track_id,
                    "track_confirmed": detection.track_confirmed,
                    "track_hits": detection.track_hits,
                    "track_missed_frames": detection.track_missed_frames,
                    "track_score": round(detection.track_score, 5),
                }
                for detection in frame_result.detections
            ],
        }


def run() -> None:
    """Run the PC-side inference server with the default configuration."""
    PCVisionServer(PCRuntimeConfig()).run()
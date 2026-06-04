from __future__ import annotations

"""PC-side TCP inference server pipeline."""

import json
import os
import signal
import socket
import threading
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from ..ai.detector import DetectorConfig, WasteDetector
from ..config import AppConfig
from ..network.transport import StreamConnection, TransportMessage
from ..types import Detection, FrameResult
from ..utils.visualization import draw_detections


@dataclass(slots=True)
class ServerRuntime:
    host: str = "0.0.0.0"
    port: int = 5000


class StreamServerPipeline:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.detector = WasteDetector(
            DetectorConfig(
                model_path=None if config.detector.backend == "edge_impulse_http" else str(config.resolve_model_path()),
                backend=config.detector.backend,
                conf_threshold=config.detector.conf_threshold,
                iou_threshold=config.detector.iou_threshold,
                image_size=config.detector.image_size,
                max_detections=config.detector.max_detections,
                use_half=config.detector.use_half,
                device=config.detector.device,
                workers=config.detector.workers,
                force_pytorch=config.detector.force_pytorch,
                edge_impulse_url=config.detector.edge_impulse_url,
                edge_impulse_timeout_sec=config.detector.edge_impulse_timeout_sec,
                debug_inference=config.runtime.debug_inference,
            )
        )
        self.output_dir = Path(config.config_dir / config.paths.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results_jsonl = self.output_dir / "detections.jsonl"
        self.predict_dir = self.output_dir / "predict"
        self.predict_dir.mkdir(parents=True, exist_ok=True)
        self.stop_event = threading.Event()

    def run(self) -> None:
        def _signal_handler(_sig: int, _frame: object | None) -> None:
            self.stop_event.set()

        signal.signal(signal.SIGINT, _signal_handler)
        try:
            signal.signal(signal.SIGTERM, _signal_handler)
        except Exception:
            pass

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
            server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server_socket.bind((self.config.network.host, self.config.network.port))
            server_socket.listen(1)
            server_socket.settimeout(1.0)
            print(f"PC server listening on {self.config.network.host}:{self.config.network.port}")

            try:
                while not self.stop_event.is_set():
                    try:
                        client_socket, address = server_socket.accept()
                    except socket.timeout:
                        continue
                    except OSError:
                        break

                    print(f"Client connected from {address[0]}:{address[1]}")
                    try:
                        client_socket.settimeout(1.0)
                    except Exception:
                        pass

                    with client_socket:
                        self._handle_client(StreamConnection(client_socket))
            finally:
                try:
                    cv2.destroyAllWindows()
                except Exception:
                    pass

    def _handle_client(self, connection: StreamConnection) -> None:
        try:
            while not self.stop_event.is_set():
                try:
                    message = connection.receive_message()
                except socket.timeout:
                    continue
                if message.header.get("kind") != "frame":
                    continue

                frame_index = int(message.header.get("frame_index", 0))
                timestamp = float(message.header.get("timestamp", 0.0))
                frame = self._decode_frame(message)
                if frame is None:
                    continue

                raw_detections = self.detector.infer(frame)
                self._store_results(frame_index, timestamp, frame, raw_detections)

                response = FrameResult(frame_index=frame_index, timestamp=timestamp, detections=raw_detections)
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
            print(f"Client handler error: {exc}")
        finally:
            connection.close()

    def _decode_frame(self, message: TransportMessage) -> np.ndarray | None:
        if message.header.get("encoding") != "jpeg":
            return None
        buffer = np.frombuffer(message.body, dtype=np.uint8)
        frame = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
        if frame is None:
            print("Unable to decode received JPEG frame")
        return frame

    def _store_results(self, frame_index: int, timestamp: float, frame: np.ndarray, detections: list[Detection]) -> None:
        if not detections:
            return
        payload = self._frame_result_to_payload(FrameResult(frame_index=frame_index, timestamp=timestamp, detections=detections))
        with self.results_jsonl.open("a", encoding="utf-8") as file:
            file.write(json.dumps(payload, ensure_ascii=True) + os.linesep)

        annotated = draw_detections(frame, detections)
        out_file = self.predict_dir / f"frame_{frame_index:06d}.jpg"
        cv2.imwrite(str(out_file), annotated)

        if self.config.runtime.show:
            cv2.imshow("wall-e-vision-pc", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                raise KeyboardInterrupt

    def _frame_result_to_payload(self, frame_result: FrameResult) -> dict[str, object]:
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
                }
                for detection in frame_result.detections
            ],
        }
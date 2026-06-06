from __future__ import annotations

"""Inference layer for the Rubik Pi 3 (NPU/DLC, ONNX and Edge Impulse backends)."""

from dataclasses import dataclass
import json
from pathlib import Path
from typing import List
from urllib import error, request
from uuid import uuid4

import cv2
import numpy as np

from ..types import Detection
from .sorting import classify_recycle_bin


@dataclass(slots=True)
class DetectorConfig:
    model_path: str | None = None
    backend: str = "pysnpe"
    conf_threshold: float = 0.10
    iou_threshold: float = 0.45
    image_size: int = 640
    max_detections: int = 8
    edge_impulse_url: str = "http://127.0.0.1:1337"
    edge_impulse_timeout_sec: float = 5.0
    debug_inference: bool = False


class WasteDetector:
    def __init__(self, cfg: DetectorConfig) -> None:
        self.cfg = cfg
        self.class_names: dict[int, str] = {}
        self.label_to_class_id: dict[str, int] = {}
        self.onnx_session = None
        self.onnx_input_name = None
        self.onnx_input_size = cfg.image_size

        self.snpe_engine = None
        self.pysnpe_module = None

        if self.cfg.backend == "edge_impulse_http":
            self.edge_impulse_base_url = self._normalize_edge_impulse_url(cfg.edge_impulse_url)
            self.edge_impulse_image_url = f"{self.edge_impulse_base_url}/api/image"
            self.edge_impulse_info_url = f"{self.edge_impulse_base_url}/api/info"
            self.class_names = self._load_edge_impulse_labels()
            self.label_to_class_id = {label: class_id for class_id, label in self.class_names.items()}
            self.model_path = None
            self.model_format = "edge_impulse_http"
            return

        if cfg.model_path is None:
            raise ValueError("model_path is required when detector.backend is not edge_impulse_http")

        self.model_path = self._resolve_model_path(Path(cfg.model_path))
        self.model_format = self.model_path.suffix.lower().lstrip(".")

        if self.model_path.suffix == ".onnx":
            try:
                import onnxruntime as ort
            except ImportError as exc:  # pragma: no cover - optional dependency
                raise RuntimeError("onnxruntime is required to run ONNX detector models") from exc

            # ONNX models are handled directly here because Ultralytics can misread some exports
            # and return raw tensors instead of proper boxes on this project.
            self.onnx_session = ort.InferenceSession(str(self.model_path), providers=["CPUExecutionProvider"])
            self.onnx_input_name = self.onnx_session.get_inputs()[0].name
            input_shape = self.onnx_session.get_inputs()[0].shape
            if len(input_shape) >= 4:
                input_height = input_shape[2]
                input_width = input_shape[3]
                if isinstance(input_height, int) and isinstance(input_width, int) and input_height == input_width:
                    self.onnx_input_size = int(input_height)

            from ..utils.labels import CLASS_NAMES
            self.class_names = {i: name for i, name in enumerate(CLASS_NAMES)}
        elif self.model_path.suffix == ".dlc":
            try:
                import snpe_native
                self.pysnpe_module = snpe_native
            except ImportError as exc:
                raise RuntimeError("snpe_native is required to run DLC detector models. Please build it from src/snpe_native") from exc

            self.snpe_engine = snpe_native.SnpeYoloDetector(str(self.model_path))
            
            self.snpe_input_size = cfg.image_size
            
            # Labels par défaut pour les modèles DLC (la sortie NPU ne porte pas les noms de classe).
            from ..utils.labels import CLASS_NAMES
            self.class_names = {i: name for i, name in enumerate(CLASS_NAMES)}
        else:
            raise ValueError(
                f"Unsupported model format '{self.model_path.suffix}'. "
                "Rubik Pi 3 supports .dlc (pysnpe), .onnx (onnxruntime) or Edge Impulse .eim."
            )

    def _resolve_model_path(self, model_path: Path) -> Path:
        if model_path.exists():
            return model_path
        alternatives = []
        if model_path.suffix == ".onnx":
            alternatives.append(model_path.with_suffix(".dlc"))
        elif model_path.suffix == ".dlc":
            alternatives.append(model_path.with_suffix(".onnx"))
        for candidate in alternatives:
            if candidate.exists():
                return candidate
        raise FileNotFoundError(f"Model not found: {model_path}")

    def _normalize_edge_impulse_url(self, url: str) -> str:
        base_url = url.rstrip("/")
        if base_url.endswith("/api/image"):
            base_url = base_url[: -len("/api/image")]
        if base_url.endswith("/api/info"):
            base_url = base_url[: -len("/api/info")]
        return base_url.rstrip("/")

    def _load_edge_impulse_labels(self) -> dict[int, str]:
        info_request = request.Request(self.edge_impulse_info_url, method="GET")
        try:
            with request.urlopen(info_request, timeout=self.cfg.edge_impulse_timeout_sec) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except Exception as exc:
            raise RuntimeError(f"Unable to reach Edge Impulse runner at {self.edge_impulse_info_url}: {exc}") from exc

        labels = payload.get("modelParameters", {}).get("labels", []) or []
        return {index: str(label) for index, label in enumerate(labels)}

    def _build_multipart_request(self, frame: np.ndarray) -> tuple[bytes, str]:
        success, encoded = cv2.imencode(".jpg", frame)
        if not success:
            raise RuntimeError("Failed to encode frame as JPEG for Edge Impulse inference")

        boundary = f"----wall-e-vision-{uuid4().hex}"
        body = bytearray()
        body.extend(f"--{boundary}\r\n".encode("utf-8"))
        body.extend(b'Content-Disposition: form-data; name="file"; filename="frame.jpg"\r\n')
        body.extend(b"Content-Type: image/jpeg\r\n\r\n")
        body.extend(encoded.tobytes())
        body.extend(b"\r\n")
        body.extend(f"--{boundary}--\r\n".encode("utf-8"))
        return bytes(body), f"multipart/form-data; boundary={boundary}"

    def _letterbox(self, frame: np.ndarray, new_size: int) -> tuple[np.ndarray, float, int, int]:
        original_height, original_width = frame.shape[:2]
        scale = min(new_size / original_height, new_size / original_width)
        resized_width = int(round(original_width * scale))
        resized_height = int(round(original_height * scale))
        pad_width = new_size - resized_width
        pad_height = new_size - resized_height
        pad_left = pad_width // 2
        pad_top = pad_height // 2
        pad_right = pad_width - pad_left
        pad_bottom = pad_height - pad_top

        if (resized_width, resized_height) != (original_width, original_height):
            resized = cv2.resize(frame, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
        else:
            resized = frame

        padded = cv2.copyMakeBorder(
            resized,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            borderType=cv2.BORDER_CONSTANT,
            value=(114, 114, 114),
        )
        return padded, scale, pad_left, pad_top

    def _softmax(self, values: np.ndarray) -> np.ndarray:
        shifted = values - np.max(values)
        exponent = np.exp(shifted)
        total = float(np.sum(exponent))
        if total <= 0.0:
            return np.zeros_like(values)
        return exponent / total

    def _iou(self, box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        x1 = np.maximum(box[0], boxes[:, 0])
        y1 = np.maximum(box[1], boxes[:, 1])
        x2 = np.minimum(box[2], boxes[:, 2])
        y2 = np.minimum(box[3], boxes[:, 3])

        inter_w = np.maximum(0.0, x2 - x1)
        inter_h = np.maximum(0.0, y2 - y1)
        inter_area = inter_w * inter_h

        box_area = np.maximum(0.0, box[2] - box[0]) * np.maximum(0.0, box[3] - box[1])
        boxes_area = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0.0, boxes[:, 3] - boxes[:, 1])
        union = box_area + boxes_area - inter_area
        return np.where(union > 0.0, inter_area / union, 0.0)

    def _nms(self, boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> list[int]:
        if boxes.size == 0:
            return []

        order = scores.argsort()[::-1]
        keep: list[int] = []

        while order.size > 0:
            current = int(order[0])
            keep.append(current)
            if order.size == 1:
                break

            remaining = order[1:]
            ious = self._iou(boxes[current], boxes[remaining])
            order = remaining[ious <= iou_threshold]

        return keep

    def _postprocess_onnx(self, output: np.ndarray, frame: np.ndarray) -> List[Detection]:
        frame_height, frame_width = frame.shape[:2]
        frame_area = float(frame_height * frame_width)
        padded, scale, pad_left, pad_top = self._letterbox(frame, self.onnx_input_size)

        # Keep the preprocessing path aligned with the model input even though the actual
        # inference already ran; this lets us project the decoded coordinates back to the camera frame.
        _ = padded

        predictions = np.transpose(output[0], (1, 0))
        raw_boxes = predictions[:, :4].astype(np.float32)
        class_logits = predictions[:, 4:].astype(np.float32)

        class_probabilities = np.apply_along_axis(self._softmax, 1, class_logits)
        class_ids = np.argmax(class_probabilities, axis=1)
        confidences = class_probabilities[np.arange(class_probabilities.shape[0]), class_ids]

        valid = confidences >= self.cfg.conf_threshold
        if not np.any(valid):
            return []

        raw_boxes = raw_boxes[valid]
        class_ids = class_ids[valid]
        confidences = confidences[valid]

        xyxy = np.empty_like(raw_boxes, dtype=np.float32)
        xyxy[:, 0] = raw_boxes[:, 0] - raw_boxes[:, 2] / 2.0
        xyxy[:, 1] = raw_boxes[:, 1] - raw_boxes[:, 3] / 2.0
        xyxy[:, 2] = raw_boxes[:, 0] + raw_boxes[:, 2] / 2.0
        xyxy[:, 3] = raw_boxes[:, 1] + raw_boxes[:, 3] / 2.0

        xyxy[:, [0, 2]] -= float(pad_left)
        xyxy[:, [1, 3]] -= float(pad_top)
        xyxy /= float(scale)

        xyxy[:, 0] = np.clip(xyxy[:, 0], 0.0, frame_width - 1)
        xyxy[:, 1] = np.clip(xyxy[:, 1], 0.0, frame_height - 1)
        xyxy[:, 2] = np.clip(xyxy[:, 2], 0.0, frame_width - 1)
        xyxy[:, 3] = np.clip(xyxy[:, 3], 0.0, frame_height - 1)

        widths = np.maximum(1.0, xyxy[:, 2] - xyxy[:, 0])
        heights = np.maximum(1.0, xyxy[:, 3] - xyxy[:, 1])
        areas = widths * heights

        keep_indices: list[int] = []
        for class_id in np.unique(class_ids):
            class_mask = class_ids == class_id
            class_boxes = xyxy[class_mask]
            class_scores = confidences[class_mask]
            selected = self._nms(class_boxes, class_scores, self.cfg.iou_threshold)
            if not selected:
                continue
            class_global_indices = np.flatnonzero(class_mask)
            keep_indices.extend(class_global_indices[selected].tolist())

        detections: List[Detection] = []
        for index in sorted(keep_indices, key=lambda i: float(confidences[i]), reverse=True)[: self.cfg.max_detections]:
            x1, y1, x2, y2 = xyxy[index]
            x1_i = int(round(x1))
            y1_i = int(round(y1))
            x2_i = int(round(x2))
            y2_i = int(round(y2))
            x1_i = max(0, min(frame_width - 1, x1_i))
            y1_i = max(0, min(frame_height - 1, y1_i))
            x2_i = max(0, min(frame_width - 1, x2_i))
            y2_i = max(0, min(frame_height - 1, y2_i))
            if x2_i <= x1_i or y2_i <= y1_i:
                continue

            class_id = int(class_ids[index])
            class_name = self.class_names.get(class_id, str(class_id))
            center = ((x1_i + x2_i) // 2, (y1_i + y2_i) // 2)

            detections.append(
                Detection(
                    class_id=class_id,
                    class_name=class_name,
                    recycle_bin=classify_recycle_bin(class_name),
                    confidence=float(confidences[index]),
                    bbox=(x1_i, y1_i, x2_i, y2_i),
                    center=center,
                    pickup_point=center,
                    area_ratio=float(areas[index]) / frame_area,
                    bbox_clipped=x1_i == 0 or y1_i == 0 or x2_i == frame_width - 1 or y2_i == frame_height - 1,
                )
            )

        return detections

    def _infer_edge_impulse(self, frame: np.ndarray) -> List[Detection]:
        frame_height, frame_width = frame.shape[:2]
        frame_area = float(frame_height * frame_width)
        padded_frame, scale, pad_left, pad_top = self._letterbox(frame, self.cfg.image_size)
        body, content_type = self._build_multipart_request(padded_frame)
        image_request = request.Request(
            self.edge_impulse_image_url,
            data=body,
            headers={"Content-Type": content_type},
            method="POST",
        )

        try:
            with request.urlopen(image_request, timeout=self.cfg.edge_impulse_timeout_sec) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except error.HTTPError as exc:
            message = exc.read().decode("utf-8", errors="replace") if exc.fp else str(exc)
            raise RuntimeError(f"Edge Impulse inference failed with HTTP {exc.code}: {message}") from exc
        except Exception as exc:
            raise RuntimeError(f"Edge Impulse inference failed: {exc}") from exc

        result = payload.get("result", {}) or {}
        detections: List[Detection] = []
        boxes = result.get("bounding_boxes") or result.get("object_tracking") or []

        if not boxes:
            # Some Edge Impulse exports are classifiers, not detectors. In that case there are no
            # boxes to decode, so we keep the best class and map it to a full-frame detection.
            classification_candidates = (
                result.get("classification")
                or result.get("classifications")
                or result.get("predictions")
                or payload.get("classification")
                or payload.get("predictions")
                or []
            )
            if isinstance(classification_candidates, dict):
                classification_candidates = [classification_candidates]

            best_label = None
            best_score = 0.0
            for item in classification_candidates:
                if not isinstance(item, dict):
                    continue
                label = str(item.get("label") or item.get("className") or item.get("name") or "")
                score = item.get("value", item.get("score", item.get("probability", 0.0)))
                try:
                    score_value = float(score)
                except Exception:
                    continue
                if score_value > best_score:
                    best_score = score_value
                    best_label = label

            if best_label and best_score >= self.cfg.conf_threshold:
                return [
                    Detection(
                        class_id=self.label_to_class_id.get(best_label, -1),
                        class_name=best_label,
                        recycle_bin=classify_recycle_bin(best_label),
                        confidence=best_score,
                        bbox=(0, 0, frame_width - 1, frame_height - 1),
                        center=(frame_width // 2, frame_height // 2),
                        pickup_point=(frame_width // 2, frame_height // 2),
                        area_ratio=1.0,
                        bbox_clipped=False,
                    )
                ]

        for box in boxes:
            label = str(box.get("label", ""))
            class_id = self.label_to_class_id.get(label, -1)
            
            raw_x = box.get("x", 0)
            raw_y = box.get("y", 0)
            raw_w = box.get("width", 0)
            raw_h = box.get("height", 0)
            
            x1_original = (raw_x - pad_left) / scale
            y1_original = (raw_y - pad_top) / scale
            x2_original = (raw_x + raw_w - pad_left) / scale
            y2_original = (raw_y + raw_h - pad_top) / scale
            
            # Clip aux bords de l'image de la caméra
            x1_i = max(0, int(round(x1_original)))
            y1_i = max(0, int(round(y1_original)))
            x2_i = min(frame_width - 1, int(round(x2_original)))
            y2_i = min(frame_height - 1, int(round(y2_original)))

            clipped_width = max(1, x2_i - x1_i)
            clipped_height = max(1, y2_i - y1_i)
            confidence = float(box.get("value", 0.0))
            center = (x1_i + clipped_width // 2, y1_i + clipped_height // 2)

            detections.append(
                Detection(
                    class_id=class_id,
                    class_name=label,
                    recycle_bin=classify_recycle_bin(label),
                    confidence=confidence,
                    bbox=(x1_i, y1_i, x2_i, y2_i),
                    center=center,
                    pickup_point=center,
                    area_ratio=(clipped_width * clipped_height) / frame_area,
                    bbox_clipped=x1_i == 0 or y1_i == 0 or x2_i == frame_width - 1 or y2_i == frame_height - 1,
                )
            )

        return detections

    def _infer_pysnpe(self, frame: np.ndarray) -> List[Detection]:
        # 1. Prépare l'image (Letterbox)
        padded_frame, scale, pad_left, pad_top = self._letterbox(frame, self.snpe_input_size)
        
        # 2. Format Float32 (Normalisé 0-1)
        model_input = cv2.cvtColor(padded_frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        
        # 3. Transposition virtuelle (NCHW)
        model_input = np.transpose(model_input, (2, 0, 1))
        
        # 4. LE CORRECTIF MAGIQUE : Réécriture physique en mémoire RAM !
        model_input = np.ascontiguousarray(model_input)[np.newaxis, ...]

        # 5. Inférence NPU ultra-rapide
        snpe_out = self.snpe_engine.infer(model_input)
        
        # Check if the output is a dictionary (multiple outputs) or a single array
        if isinstance(snpe_out, dict):
            if self.cfg.debug_inference:
                print(f"DEBUG: snpe_out is a dict with keys: {snpe_out.keys()}")
                for k, v in snpe_out.items():
                    print(f"DEBUG: snpe_out['{k}'] shape: {v.shape}")
                    print(f"DEBUG: {k} content: {v[0][0]}")
            # If the user split the outputs to avoid quantization issues, we reconstruct them here.
            # Assuming outputs are named 'boxes', 'conf', 'cls'
            if 'boxes' in snpe_out and 'conf' in snpe_out and 'cls' in snpe_out:
                boxes_arr = snpe_out['boxes'][0]
                conf_arr = snpe_out['conf'][0]
                cls_arr = snpe_out['cls'][0]
                boxes = np.concatenate([boxes_arr, conf_arr, cls_arr], axis=-1)
            else:
                # Fallback: take the first tensor in the dict
                first_key = list(snpe_out.keys())[0]
                boxes = snpe_out[first_key][0]
        else:
            if self.cfg.debug_inference:
                print(f"DEBUG: snpe_out is a single array of shape: {snpe_out.shape}")
                print(f"DEBUG: Boxes content: {snpe_out[0][0]}")
            # Old behavior: snpe_out is a single array of shape (1, 300, 6)
            boxes = snpe_out[0]

        if self.cfg.debug_inference and len(boxes) > 0:
            print(f"DEBUG 1ère boîte : {boxes[0]}")
        
        detections: List[Detection] = []
        frame_height, frame_width = frame.shape[:2]
        frame_area = float(frame_height * frame_width)

        # 6. Décodage direct.
        # YOLO26 est end-to-end (NMS-free) : chaque ligne est déjà
        # [x1, y1, x2, y2, score, class] dans l'espace letterboxé 640x640.
        # Pas de conversion cx,cy,w,h -> xyxy, pas de NMS à refaire.
        for box in boxes:
            x1, y1, x2, y2, conf, cls_id = box

            # On ignore les boîtes sous le seuil
            if conf < self.cfg.conf_threshold:
                continue

            # On ramène les coordonnées à l'image d'origine (avant le letterbox)
            x1_orig = (x1 - pad_left) / scale
            y1_orig = (y1 - pad_top) / scale
            x2_orig = (x2 - pad_left) / scale
            y2_orig = (y2 - pad_top) / scale
            
            # On arrondit et on clip aux bords de l'image
            x1_i = max(0, int(round(x1_orig)))
            y1_i = max(0, int(round(y1_orig)))
            x2_i = min(frame_width - 1, int(round(x2_orig)))
            y2_i = min(frame_height - 1, int(round(y2_orig)))
            
            # On vérifie que la boîte est valide
            if x2_i <= x1_i or y2_i <= y1_i:
                continue
                
            class_id_int = int(cls_id)
            class_name = self.class_names.get(class_id_int, str(class_id_int))
            
            box_width = max(1, x2_i - x1_i)
            box_height = max(1, y2_i - y1_i)
            center = (x1_i + box_width // 2, y1_i + box_height // 2)
            
            detections.append(
                Detection(
                    class_id=class_id_int,
                    class_name=class_name,
                    recycle_bin=classify_recycle_bin(class_name),
                    confidence=float(conf),
                    bbox=(x1_i, y1_i, x2_i, y2_i),
                    center=center,
                    pickup_point=center,
                    area_ratio=(box_width * box_height) / frame_area,
                    bbox_clipped=x1_i == 0 or y1_i == 0 or x2_i == frame_width - 1 or y2_i == frame_height - 1,
                )
            )
            
        return detections

    def infer(self, frame: np.ndarray) -> List[Detection]:
        if self.cfg.backend == "edge_impulse_http":
            return self._infer_edge_impulse(frame)

        if self.model_path is not None and self.model_path.suffix == ".dlc" and self.snpe_engine is not None:
            return self._infer_pysnpe(frame)

        if self.model_path is not None and self.model_path.suffix == ".onnx" and self.onnx_session is not None and self.onnx_input_name is not None:
            padded_frame, _, _, _ = self._letterbox(frame, self.onnx_input_size)
            model_input = cv2.cvtColor(padded_frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            # model_input = np.transpose(model_input, (2, 0, 1))[np.newaxis, ...]
            # model_input = model_input[np.newaxis, ...]
            output = self.onnx_session.run(None, {self.onnx_input_name: model_input})[0]
            return self._postprocess_onnx(output, frame)

        raise RuntimeError("Detector is not initialized with a supported backend")
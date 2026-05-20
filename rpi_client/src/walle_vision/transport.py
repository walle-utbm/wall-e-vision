from __future__ import annotations

"""Full-duplex transport utilities for the Raspberry Pi streaming branch.

Frames are sent as JPEG payloads and results are received as JSON payloads over
the same TCP connection. The protocol is intentionally small and explicit so the
PC branch can mirror it without depending on any extra framework.
"""

from dataclasses import dataclass
import json
import socket
import struct
import threading
import time
from typing import Any

import cv2
import numpy as np


_UINT32 = struct.Struct("!I")


@dataclass(slots=True)
class TransportMessage:
    """Decoded transport message."""

    header: dict[str, Any]
    body: bytes


def _read_exact(sock: socket.socket, size: int) -> bytes:
    """Read exactly `size` bytes or raise on disconnect."""
    chunks: list[bytes] = []
    remaining = size
    while remaining > 0:
        chunk = sock.recv(remaining)
        if not chunk:
            raise ConnectionError("Socket closed while reading transport payload")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def encode_packet(header: dict[str, Any], body: bytes = b"") -> bytes:
    """Serialize a framed transport packet."""
    header_bytes = json.dumps(header, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return _UINT32.pack(len(header_bytes)) + header_bytes + _UINT32.pack(len(body)) + body


def decode_packet(sock: socket.socket) -> TransportMessage:
    """Read a framed transport packet from a socket."""
    header_size = _UINT32.unpack(_read_exact(sock, _UINT32.size))[0]
    header = json.loads(_read_exact(sock, header_size).decode("utf-8"))
    body_size = _UINT32.unpack(_read_exact(sock, _UINT32.size))[0]
    body = _read_exact(sock, body_size) if body_size else b""
    return TransportMessage(header=header, body=body)


def encode_frame_to_jpeg(frame: np.ndarray, quality: int) -> bytes:
    """Encode a frame as JPEG for network transport."""
    encoded_frame = np.ascontiguousarray(frame)
    success, buffer = cv2.imencode(
        ".jpg",
        encoded_frame,
        [int(cv2.IMWRITE_JPEG_QUALITY), max(1, min(100, int(quality)))],
    )
    if not success:
        raise RuntimeError("Unable to JPEG-encode camera frame")
    return buffer.tobytes()


class StreamChannel:
    """Thread-safe TCP channel used to send frames and receive results."""

    def __init__(self, host: str, port: int, timeout_sec: float = 5.0) -> None:
        self.host = host
        self.port = port
        self.timeout_sec = timeout_sec
        self._socket: socket.socket | None = None
        self._write_lock = threading.Lock()

    def connect(self) -> None:
        """Open a TCP connection to the PC receiver."""
        self._socket = socket.create_connection((self.host, self.port), timeout=self.timeout_sec)
        self._socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        # Keep writes blocking so large JPEG frames are not aborted by a short
        # global socket timeout. Read operations apply their own timeout.
        self._socket.settimeout(None)

    @property
    def connected(self) -> bool:
        return self._socket is not None

    def send_packet(self, header: dict[str, Any], body: bytes = b"") -> float:
        """Send one framed packet on the TCP stream and return the send duration."""
        if self._socket is None:
            raise ConnectionError("Transport channel is not connected")

        packet = encode_packet(header, body)
        send_started = time.perf_counter()
        with self._write_lock:
            self._socket.sendall(packet)
        return time.perf_counter() - send_started

    def send_frame(self, frame_index: int, timestamp: float, frame: np.ndarray, jpeg_quality: int) -> float:
        """Encode and send one camera frame and return the send duration."""
        body = encode_frame_to_jpeg(frame, jpeg_quality)
        header = {
            "kind": "frame",
            "frame_index": frame_index,
            "timestamp": timestamp,
            "encoding": "jpeg",
            "shape": list(frame.shape),
            "dtype": str(frame.dtype),
        }
        return self.send_packet(header, body)

    def receive_message(self, timeout_sec: float | None = None) -> TransportMessage:
        """Receive one framed packet from the TCP stream."""
        sock = self._socket
        if sock is None:
            raise ConnectionError("Transport channel is not connected")

        previous_timeout = sock.gettimeout()
        try:
            if timeout_sec is not None:
                sock.settimeout(timeout_sec)
            return decode_packet(sock)
        finally:
            if self._socket is not None:
                try:
                    self._socket.settimeout(previous_timeout)
                except OSError:
                    pass

    def close(self) -> None:
        """Close the TCP socket."""
        if self._socket is not None:
            try:
                self._socket.shutdown(socket.SHUT_RDWR)
            except Exception:
                pass
            self._socket.close()
            self._socket = None
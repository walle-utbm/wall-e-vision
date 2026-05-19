from __future__ import annotations

"""TCP frame/result protocol helpers for the PC server."""

from dataclasses import dataclass
import json
import socket
import struct
import threading
from typing import Any


_UINT32 = struct.Struct("!I")


@dataclass(slots=True)
class TransportMessage:
    header: dict[str, Any]
    body: bytes


def _read_exact(sock: socket.socket, size: int) -> bytes:
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
    header_bytes = json.dumps(header, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return _UINT32.pack(len(header_bytes)) + header_bytes + _UINT32.pack(len(body)) + body


def decode_packet(sock: socket.socket) -> TransportMessage:
    header_size = _UINT32.unpack(_read_exact(sock, _UINT32.size))[0]
    header = json.loads(_read_exact(sock, header_size).decode("utf-8"))
    body_size = _UINT32.unpack(_read_exact(sock, _UINT32.size))[0]
    body = _read_exact(sock, body_size) if body_size else b""
    return TransportMessage(header=header, body=body)


class StreamConnection:
    """Thread-safe wrapper around one accepted TCP socket."""

    def __init__(self, sock: socket.socket) -> None:
        self._socket = sock
        self._write_lock = threading.Lock()

    def receive_message(self) -> TransportMessage:
        return decode_packet(self._socket)

    def send_packet(self, header: dict[str, Any], body: bytes = b"") -> None:
        packet = encode_packet(header, body)
        with self._write_lock:
            self._socket.sendall(packet)

    def close(self) -> None:
        try:
            self._socket.shutdown(socket.SHUT_RDWR)
        except Exception:
            pass
        self._socket.close()
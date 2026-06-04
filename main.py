from __future__ import annotations

"""Root entrypoint for the unified wall-e-vision application."""

import sys
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path
from time import sleep
from urllib import request
from urllib.parse import urlparse


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from walle_vision.config import load_config
from walle_vision.pipelines import build_pipeline


def _wait_for_edge_impulse_server(base_url: str, timeout_sec: float = 30.0) -> None:
    # The Python pipeline should only start once the local HTTP runner is ready.
    info_url = f"{base_url.rstrip('/')}/api/info"
    deadline = time.monotonic() + timeout_sec
    last_error: Exception | None = None

    while time.monotonic() < deadline:
        try:
            with request.urlopen(info_url, timeout=2.0) as response:
                if response.status == 200:
                    return
        except Exception as exc:
            last_error = exc
            sleep(1.0)

    raise RuntimeError(f"Edge Impulse HTTP server did not become ready at {info_url}: {last_error}")


@contextmanager
def _edge_impulse_http_runner(config):
    runner_process: subprocess.Popen[str] | None = None
    try:
        if config.detector.backend == "edge_impulse_http":
            base_url = config.detector.edge_impulse_url.rstrip("/")
            parsed = urlparse(base_url)
            if parsed.hostname in {"127.0.0.1", "localhost", "0.0.0.0"}:
                # Rubik Pi 3 starts the Edge Impulse runner locally, then the Python code sends
                # frames to its HTTP API instead of launching a second inference stack.
                model_path = config.resolve_model_path()
                port = parsed.port or 1337
                command = [
                    "edge-impulse-linux-runner",
                    "--model-file",
                    str(model_path),
                    "--run-http-server",
                    str(port),
                    "--thresholds 0.1",
                    "19.min_score=0.1",
                    # "--silent",
                ]
                runner_process = subprocess.Popen(
                    command,
                    cwd=str(ROOT),
                    # Keep the console clean: the app only needs the service, not the runner logs.
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                _wait_for_edge_impulse_server(base_url)
        yield
    finally:
        if runner_process is not None and runner_process.poll() is None:
            runner_process.terminate()
            try:
                runner_process.wait(timeout=10.0)
            except Exception:
                runner_process.kill()


def main() -> None:
    config = load_config(ROOT / "config.yaml")
    with _edge_impulse_http_runner(config):
        pipeline = build_pipeline(config)
        pipeline.run()


if __name__ == "__main__":
    main()
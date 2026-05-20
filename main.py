from __future__ import annotations

"""Root entrypoint for the unified wall-e-vision application."""

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from walle_vision.config import load_config
from walle_vision.pipelines import build_pipeline


def main() -> None:
    config = load_config(ROOT / "config.yaml")
    pipeline = build_pipeline(config)
    pipeline.run()


if __name__ == "__main__":
    main()
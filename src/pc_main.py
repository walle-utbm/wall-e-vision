"""PC-side executable entrypoint.

Run this file to start the TCP inference server:
    python src/pc_main.py
"""

from walle_vision.pc.server import run


if __name__ == "__main__":
    run()
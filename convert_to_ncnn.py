#!/usr/bin/env python3
"""Convert YOLO model from PyTorch (.pt) to NCNN format for Raspberry Pi.

NCNN is an extremely fast neural network inference framework optimized for ARM.
This script exports your trained YOLO model to NCNN (.param + .bin files).

Required for NCNN export:
    pip install openvino-dev

Usage:
    python convert_to_ncnn.py --model model/best.pt --output model/
    python convert_to_ncnn.py --model model/best.pt --cpu-only  # Force CPU export

Output:
    - model/best.ncnn.param  (network structure)
    - model/best.ncnn.bin    (quantized weights)
"""

import argparse
import sys
from pathlib import Path

import torch
import torch
from ultralytics import YOLO


def convert_to_ncnn(model_path: str, output_dir: str = "model/", use_gpu: bool = True) -> None:
    """Convert YOLO .pt model to NCNN format.

    Args:
        model_path: Path to the YOLO .pt model file.
        output_dir: Output directory for NCNN files.
        use_gpu: Try to use GPU if available (CUDA). Falls back to CPU.

    Raises:
        FileNotFoundError: If model_path does not exist.
        RuntimeError: If export fails.
    """
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Detect available device
    if use_gpu and torch.cuda.is_available():
        device = 0
        device_name = f"GPU (CUDA {torch.cuda.get_device_name(0)})"
    else:
        device = "cpu"
        device_name = "CPU"

    print(f"Loading YOLO model from: {model_path}")
    model = YOLO(str(model_path))

    print(f"Exporting to NCNN format on {device_name}...")
    print("  - This may take a few minutes")
    print(f"  - Output will be in: {output_dir}")

    try:
        # Export to NCNN format
        export_result = model.export(
            format="ncnn",
            imgsz=640,
            device=device,  # Auto-detected: 0 for GPU, 'cpu' for CPU
            half=False,  # Set to True if you want FP16 (requires GPU)
            dynamic=False,
            simplify=True,
            opset=None,
            workspace=4,
        )

        print(f"\n✅ Export successful!")
        print(f"NCNN model saved to: {export_result}")
        print("\nFiles generated:")
        print(f"  - {output_dir}/best.ncnn.param")
        print(f"  - {output_dir}/best.ncnn.bin")
        print("\nTo use the NCNN model:")
        print("  1. The detector.py will automatically find and use it")
        print("  2. Run: python src/main.py")
        print("  3. You should see: 'Loading NCNN model: model/best.ncnn.param'")

    except Exception as e:
        print(f"\n❌ Export failed: {e}")
        print("\nTroubleshooting:")
        print("  1. Ensure ultralytics is up to date: pip install -U ultralytics")
        print("  2. For NCNN export, you may need additional dependencies:")
        print("     pip install openvino-dev")
        print("  3. Check CUDA installation:")
        print(f"     torch.cuda.is_available() = {torch.cuda.is_available()}")
        print(f"     torch.cuda.device_count() = {torch.cuda.device_count()}")
        print("  4. Alternatively, try forcing CPU export:")
        print("     python convert_to_ncnn.py --model model/best.pt --cpu-only")
        sys.exit(1)


def main() -> None:
    """Parse arguments and run conversion."""
    # Check if openvino is available
    try:
        import openvino
    except ImportError:
        print("❌ OpenVINO is required for NCNN export!")
        print("\nInstall with:")
        print("  pip install openvino-dev")
        print("\nOr on Windows with GPU support:")
        print("  pip install openvino-dev[caffe,onnx]")
        sys.exit(1)
    
    parser = argparse.ArgumentParser(
        description="Convert YOLO model to NCNN format for Raspberry Pi ARM inference"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="model/best.pt",
        help="Path to YOLO .pt model (default: model/best.pt)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="model/",
        help="Output directory for NCNN files (default: model/)",
    )
    parser.add_argument(
        "--cpu-only",
        action="store_true",
        help="Force CPU-only export (skip GPU detection)",
    )

    args = parser.parse_args()

    try:
        convert_to_ncnn(args.model, args.output, use_gpu=not args.cpu_only)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

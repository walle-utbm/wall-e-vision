from __future__ import annotations

"""Inspect an Edge Impulse .eim package.

The .eim file used in this repository is not an ONNX graph. It is an ELF
executable that embeds the Edge Impulse runtime, model metadata and labels.
This script prints the ELF information plus the most relevant embedded strings
so it is easier to understand what the exported model actually contains.
"""

from collections import Counter
import re
import struct
import subprocess
from pathlib import Path


MODEL_PATH = Path("model/model.eim")


def run_command(command: list[str]) -> str:
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    output = result.stdout.strip()
    if result.stderr.strip():
        output = f"{output}\n{result.stderr.strip()}" if output else result.stderr.strip()
    return output


def read_strings(path: Path, min_length: int = 4) -> list[str]:
    data = path.read_bytes()
    pattern = re.compile(rb"[ -~]{%d,}" % min_length)
    return [match.decode("utf-8", errors="replace") for match in pattern.findall(data)]


def parse_elf_header(path: Path) -> dict[str, int | str]:
    with path.open("rb") as file:
        header = file.read(64)

    if len(header) < 64 or header[:4] != b"\x7fELF":
        raise ValueError(f"{path} is not an ELF file")

    unpacked = struct.unpack("<16sHHIQQQIHHHHHH", header)
    e_ident = unpacked[0]
    elf_class = "ELF64" if e_ident[4] == 2 else "ELF32"
    data_encoding = "little" if e_ident[5] == 1 else "big"

    return {
        "class": elf_class,
        "data": data_encoding,
        "type": unpacked[1],
        "machine": unpacked[2],
        "entry_point": unpacked[4],
        "program_headers_offset": unpacked[5],
        "section_headers_offset": unpacked[6],
        "flags": unpacked[7],
        "program_header_count": unpacked[10],
        "section_header_count": unpacked[12],
        "section_header_string_index": unpacked[13],
    }


def main() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    print("=== FILE TYPE ===")
    print(run_command(["file", str(MODEL_PATH)]))
    print()

    print("=== ELF HEADER ===")
    header = parse_elf_header(MODEL_PATH)
    for key, value in header.items():
        print(f"{key}: {value}")
    print()

    print("=== READelf SUMMARY ===")
    print(run_command(["readelf", "-h", str(MODEL_PATH)]))
    print()

    print("=== SECTIONS ===")
    print(run_command(["readelf", "-S", str(MODEL_PATH)]))
    print()

    strings = read_strings(MODEL_PATH)

    print("=== RELEVANT STRINGS ===")
    keywords = [
        "classification",
        "bounding_boxes",
        "TFLite_Detection_PostProcess",
        "labels",
        "label_count",
        "image_input_width",
        "image_input_height",
        "min_score",
        "min_score_box",
        "min_score_pixel",
        "qnn",
        "int8",
        "object detection",
    ]
    found: list[str] = []
    for string in strings:
        lowered = string.lower()
        if any(keyword.lower() in lowered for keyword in keywords):
            found.append(string)

    if found:
        for string in found[:200]:
            print(string)
    else:
        print("Aucune chaîne pertinente trouvée.")
    print()

    counter = Counter()
    for string in strings:
        lowered = string.lower()
        if "bounding_boxes" in lowered or "detection" in lowered:
            counter["detection"] += 1
        if "classification" in lowered or "label_count" in lowered:
            counter["classification"] += 1
        if "qnn" in lowered or "int8" in lowered:
            counter["quantized_npu"] += 1

    print("=== QUICK CONCLUSION ===")
    print(f"Detection clues: {counter['detection']}")
    print(f"Classification clues: {counter['classification']}")
    print(f"Quantized/NPU clues: {counter['quantized_npu']}")

    if counter["quantized_npu"] > 0:
        print("The package is an int8 / QNN-oriented build.")
    if counter["detection"] > 0 and counter["classification"] > 0:
        print("The package contains both detection and classification metadata, so the export type must be checked carefully.")
    elif counter["classification"] > 0 and counter["detection"] == 0:
        print("The package looks like a classifier rather than an object detector.")
    elif counter["detection"] > 0:
        print("The package looks like a detector, but the runtime/output mapping still needs to match the Edge Impulse export.")


if __name__ == "__main__":
    main()
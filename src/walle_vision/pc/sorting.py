from __future__ import annotations

"""Sorting business rules wrapper for the PC-side branch."""

from .labels import SORTING_MAP


def classify_recycle_bin(class_name: str) -> str:
    """Map a model class name to the target recycle-bin category."""
    return SORTING_MAP.get(class_name, "other")
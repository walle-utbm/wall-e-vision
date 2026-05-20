from __future__ import annotations

"""Sorting business rules wrapper."""

from ..utils.labels import SORTING_MAP


def classify_recycle_bin(class_name: str) -> str:
    return SORTING_MAP.get(class_name, "other")
from __future__ import annotations

"""Sorting business rules wrapper."""

from .labels import SORTING_MAP


def classify_recycle_bin(class_name: str) -> str:
    """Map model class name to target recycle bin category.

    Args:
        class_name: Label predicted by the vision model.

    Returns:
        One of: `yellow`, `glass`, `other`.
    """
    return SORTING_MAP.get(class_name, "other")

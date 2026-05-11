from __future__ import annotations

"""Class catalog and business mapping used by the sorting module."""

CLASS_NAMES = [
    "Plastic bottle",
    "Glass bottle",
    "Cardboard",
    "Cup",
    "Paper bag",
    "Soft plastic",
    "Food Packet",
    "Paper",
    "Organic",
    "Metal",
    "Ramen Cup",
    "Printing industry",
    "Plastic bottle cap",
    "Straw",
]

# yellow: recyclable (french yellow bin), glass: glass-only bin, other: residual or specific handling.
SORTING_MAP = {
    "Plastic bottle": "yellow",
    "Glass bottle": "glass",
    "Cardboard": "yellow",
    "Cup": "other",
    "Paper bag": "yellow",
    "Soft plastic": "yellow",
    "Food Packet": "yellow",
    "Paper": "yellow",
    "Organic": "other",
    "Metal": "yellow",
    "Ramen Cup": "other",
    "Printing industry": "yellow",
    "Plastic bottle cap": "yellow",
    "Straw": "other",
}

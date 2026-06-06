from __future__ import annotations

"""Pipeline selection and construction."""

from ..config import AppConfig
from .pipeline_edge import EdgeStandalonePipeline


def build_pipeline(config: AppConfig):
    if config.mode == "edge_standalone":
        return EdgeStandalonePipeline(config)
    raise ValueError(f"Unsupported mode '{config.mode}'")

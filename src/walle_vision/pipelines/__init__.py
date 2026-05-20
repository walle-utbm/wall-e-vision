from __future__ import annotations

"""Pipeline selection and construction."""

from ..config import AppConfig
from .pipeline_client import StreamClientPipeline
from .pipeline_edge import EdgeStandalonePipeline
from .pipeline_server import StreamServerPipeline


def build_pipeline(config: AppConfig):
    if config.mode == "edge_standalone":
        return EdgeStandalonePipeline(config)
    if config.mode == "stream_client":
        return StreamClientPipeline(config)
    if config.mode == "stream_server":
        return StreamServerPipeline(config)
    raise ValueError(f"Unsupported mode '{config.mode}'")
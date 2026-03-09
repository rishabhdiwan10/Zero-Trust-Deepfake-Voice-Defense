"""
src.pipeline — real-time async and batch processing pipelines for the
Zero-Trust Deepfake Voice Defense System.
"""

from .realtime_pipeline import RealtimePipeline
from .batch_pipeline import BatchPipeline

__all__ = [
    "RealtimePipeline",
    "BatchPipeline",
]

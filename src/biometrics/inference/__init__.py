"""
Biometric Inference Engines: Cloud WebSocket & Local Edge Async
"""

from .cloud_client import CloudInferenceClient
from .async_pipeline import EdgeAsyncPipeline

__all__ = [
    "CloudInferenceClient",
    "EdgeAsyncPipeline",
]

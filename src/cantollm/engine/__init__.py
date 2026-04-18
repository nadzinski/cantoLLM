"""Inference engine package.

Exposes the request/event domain types and the engine protocol. The sequential
implementation is the only one today; a continuous-batching engine can be
dropped in without touching the HTTP or adapter layers.
"""

from cantollm.engine.engine import InferenceEngine
from cantollm.engine.sequential import SequentialEngine
from cantollm.engine.types import InferenceRequest, SamplingParams, TokenEvent

__all__ = [
    "InferenceEngine",
    "InferenceRequest",
    "SamplingParams",
    "SequentialEngine",
    "TokenEvent",
]

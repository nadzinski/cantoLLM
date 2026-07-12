"""Inference engine package.

Exposes the request/event domain types and the engine protocol, plus the two
implementations: `SequentialEngine` (one request at a time; the only home
for speculative decoding) and `ContinuousBatchingEngine` (many requests per
forward pass, dense models only). Both sit behind `InferenceEngine`, so the
HTTP and adapter layers can't tell them apart.
"""

from cantollm.engine.batching import ContinuousBatchingEngine
from cantollm.engine.engine import InferenceEngine
from cantollm.engine.sequential import SequentialEngine
from cantollm.engine.types import InferenceRequest, SamplingParams, TokenEvent

__all__ = [
    "ContinuousBatchingEngine",
    "InferenceEngine",
    "InferenceRequest",
    "SamplingParams",
    "SequentialEngine",
    "TokenEvent",
]

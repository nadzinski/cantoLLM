"""Domain types for the inference engine.

These are the sole contract between the HTTP layer and whatever engine
implementation is running. Keeping them token-level and framework-agnostic
lets the adapter own Anthropic-specific concerns and lets the engine grow
toward batching without rippling into the API surface.
"""

from dataclasses import dataclass


@dataclass
class SamplingParams:
    temperature: float = 0.7
    top_p: float = 0.9


@dataclass
class InferenceRequest:
    request_id: str
    prompt_token_ids: list[int]
    sampling_params: SamplingParams
    max_tokens: int
    stop_token_ids: set[int]


@dataclass
class TokenEvent:
    token_id: int

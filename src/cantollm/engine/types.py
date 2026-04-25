"""Domain types for the inference engine.

These are the sole contract between the HTTP layer and whatever engine
implementation is running. Keeping them token-level and framework-agnostic
lets the adapter own Anthropic-specific concerns and lets the engine grow
toward batching without rippling into the API surface.
"""

import threading
from dataclasses import dataclass, field
from typing import Literal

from cantollm.engine.logits_processors import (
    LogitsProcessor,
    TemperatureProcessor,
    TopPProcessor,
)
from cantollm.kv_cache import KVCache

FinishReason = Literal["end_turn", "max_tokens", "abort"]


@dataclass
class SamplingParams:
    processors: list[LogitsProcessor] = field(default_factory=list)
    greedy: bool = False

    @classmethod
    def from_temperature_top_p(
        cls, temperature: float, top_p: float,
    ) -> "SamplingParams":
        if temperature == 0:
            return cls(greedy=True)
        processors: list[LogitsProcessor] = [TemperatureProcessor(temperature)]
        if top_p < 1.0:
            processors.append(TopPProcessor(top_p))
        return cls(processors=processors, greedy=False)


@dataclass
class InferenceRequest:
    request_id: str
    prompt_token_ids: list[int]
    sampling_params: SamplingParams
    max_tokens: int
    stop_token_ids: set[int]


@dataclass
class Sequence:
    """Engine-side runtime state for one in-flight request.

    Constructed from an `InferenceRequest` at submit time and passed to the
    backend's `generate` call. Backends read the immutable fields and tick
    `tokens_emitted` per yielded token; the engine inspects the sequence
    after `generate` returns to derive `finish_reason`. Once continuous
    batching lands, the scheduler will hold a list of these.
    """

    request_id: str
    prompt_token_ids: list[int]
    sampling_params: SamplingParams
    stop_token_ids: set[int]
    max_tokens: int
    cache: KVCache
    stop_event: threading.Event
    tokens_emitted: int = 0

    def finish_reason_after_normal_exit(self) -> FinishReason:
        """Map post-generate state to a FinishReason.

        Only valid when `generate` returned without raising. The backend
        having stopped without hitting `max_tokens` or the stop_event is,
        by elimination, an EOS / configured stop-token exit.
        """
        if self.tokens_emitted >= self.max_tokens:
            return "max_tokens"
        if self.stop_event.is_set():
            return "abort"
        return "end_turn"


@dataclass
class TokenEvent:
    """One event in the engine's per-request output stream.

    An event carries a token (`token_id` set) *or* marks end-of-stream
    (`finish_reason` set) *or* signals a failure (`error` set). Exactly one of
    those three is populated. `request_id` is always set on engine-produced
    events and becomes load-bearing once a batching scheduler multiplexes
    per-request queues into a shared stream.
    """

    token_id: int | None = None
    finish_reason: FinishReason | None = None
    error: str | None = None
    request_id: str | None = None

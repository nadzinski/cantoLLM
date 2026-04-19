"""Generation-strategy boundary between the engine and model execution.

An `InferenceBackend` owns model weights for its lifetime and exposes a single
per-request `generate` call. Sampling and cancellation flow through arguments,
not instance state, so one backend can safely serve many concurrent requests.
"""

import threading
from collections.abc import Iterator
from typing import Protocol

from cantollm.engine.types import SamplingParams
from cantollm.kv_cache import KVCache


class InferenceBackend(Protocol):
    def reset(self) -> None:
        """Clear any per-request state (internal caches, stats).

        Called by the engine before each `generate` to give a clean slate;
        safe no-op for stateless backends.
        """
        ...

    def generate(
        self,
        input_ids: list[int],
        cache: KVCache,
        sampling: SamplingParams,
        stop_token_ids: set[int],
        max_tokens: int,
        stop_event: threading.Event | None = None,
    ) -> Iterator[int]: ...

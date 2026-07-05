"""Generation-strategy boundary between the engine and model execution.

An `InferenceBackend` owns model weights for its lifetime and exposes a single
per-request `generate` call. All per-request state — token ids, KV cache,
sampling params, stop conditions, abort flag — is bundled into a `Sequence`
that the engine constructs and passes in. One backend can safely serve many
concurrent requests because no per-request state lives on the backend itself.
"""

from collections.abc import Iterator
from typing import Protocol

from cantollm.engine.types import Sequence


class InferenceBackend(Protocol):
    def generate(self, sequence: Sequence) -> Iterator[int]: ...

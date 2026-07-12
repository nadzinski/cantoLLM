"""EngineRegistry: maps model name to (engine, runtime).

The API layer looks up which engine owns a request's `body.model` here.
Today the CLI registers a single entry, but the interface is shaped for
multi-model serving — `/v1/messages` dispatch and tokenizer selection both
route through the registry, not through a hardcoded closure.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from cantollm.engine.engine import InferenceEngine
from cantollm.runtime import ModelRuntime, TokenizerRuntime


@dataclass
class RegistryEntry:
    engine: InferenceEngine
    runtime: ModelRuntime | TokenizerRuntime
    """Full runtime for in-process engines; tokenizer-only for models whose
    weights live in an engine process. The API layer only reads
    `.tokenizer` and the lifecycle hooks either way."""
    registered_at: float = field(default_factory=time.time)
    max_request_tokens: int | None = None
    """Admission cap: reject requests with prompt + max_tokens above this.
    None (sequential engines) means no cap. For a CB engine this is the
    per-slot KV capacity (`BatchingConfig.max_seq_len`) — an over-cap
    request would otherwise take a slot it can never fit in."""


class EngineRegistry:
    def __init__(self) -> None:
        self._entries: dict[str, RegistryEntry] = {}

    def register(
        self,
        name: str,
        engine: InferenceEngine,
        runtime: ModelRuntime | TokenizerRuntime,
        *,
        max_request_tokens: int | None = None,
    ) -> None:
        if name in self._entries:
            raise ValueError(f"Model '{name}' is already registered")
        self._entries[name] = RegistryEntry(
            engine=engine, runtime=runtime, max_request_tokens=max_request_tokens
        )

    def get(self, name: str) -> RegistryEntry:
        return self._entries[name]

    def names(self) -> list[str]:
        return list(self._entries)

    def items(self):
        return self._entries.items()

    async def start_all(self) -> None:
        for entry in self._entries.values():
            await entry.runtime.start()
            await entry.engine.start()

    async def shutdown_all(self) -> None:
        for entry in self._entries.values():
            await entry.engine.shutdown()
            await entry.runtime.shutdown()

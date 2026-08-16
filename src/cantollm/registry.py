"""EngineRegistry: maps model name to a lifecycle-managed engine.

The API layer looks up which engine owns a request's `body.model` here.
Since Phase 3.5, registration takes a *factory* rather than a built engine:
the entry's `EngineHandle` builds (and, after failures, rebuilds) the engine
in a background supervisor task, so the server accepts connections while
weights load and warm-up runs. Routers go through `entry.ensure_ready()`,
which returns the current engine or raises `NotReadyError` (-> 503).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

from cantollm.lifecycle import BuiltEngine, EngineHandle, RequestTicket


@dataclass
class RegistryEntry:
    handle: EngineHandle
    registered_at: float = field(default_factory=time.time)
    max_request_tokens: int | None = None
    """Admission cap: reject requests with prompt + max_tokens above this.
    None (sequential engines) means no cap. For a CB engine this is the
    per-slot KV capacity (`BatchingConfig.max_seq_len`) — an over-cap
    request would otherwise take a slot it can never fit in."""

    @property
    def engine(self) -> Any | None:
        """Current engine generation; None before the first Ready."""
        return self.handle.engine

    @property
    def runtime(self) -> Any | None:
        """Full runtime for in-process engines; tokenizer-only for models
        whose weights live in an engine process. Set once the handle has an
        engine (eagerly at registration for the subprocess path)."""
        return self.handle.runtime

    def ensure_ready(self) -> Any:
        return self.handle.ensure_ready()

    def begin_request(self) -> RequestTicket:
        return self.handle.begin_request()


class EngineRegistry:
    def __init__(self) -> None:
        self._entries: dict[str, RegistryEntry] = {}

    def register(
        self,
        name: str,
        factory: Callable[[], BuiltEngine],
        *,
        max_request_tokens: int | None = None,
        runtime: Any | None = None,
        drain_timeout_s: float = 30.0,
    ) -> None:
        """Register a model by factory. `runtime` may be supplied eagerly
        when it is cheap and generation-independent (the subprocess path's
        TokenizerRuntime) so tokenization metadata exists pre-Ready.
        `drain_timeout_s` bounds the reload/restart drain."""
        if name in self._entries:
            raise ValueError(f"Model '{name}' is already registered")
        handle = EngineHandle(
            name, factory, runtime=runtime, drain_timeout_s=drain_timeout_s
        )
        self._entries[name] = RegistryEntry(
            handle=handle, max_request_tokens=max_request_tokens
        )

    def get(self, name: str) -> RegistryEntry:
        return self._entries[name]

    def names(self) -> list[str]:
        return list(self._entries)

    def items(self):
        return self._entries.items()

    def launch_all(self) -> None:
        """Spawn every handle's supervisor task; returns immediately.
        Call from a running event loop (the app lifespan)."""
        for entry in self._entries.values():
            entry.handle.launch()

    async def stop_all(self) -> None:
        for entry in reversed(list(self._entries.values())):
            await entry.handle.stop()

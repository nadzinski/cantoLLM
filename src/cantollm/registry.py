"""EngineRegistry: maps model name to (engine, runtime).

The API layer looks up which engine owns a request's `body.model` here.
Today the CLI registers a single entry, but the interface is shaped for
multi-model serving — `/v1/messages` dispatch and tokenizer selection both
route through the registry, not through a hardcoded closure.
"""

from __future__ import annotations

from dataclasses import dataclass

from cantollm.engine.engine import InferenceEngine
from cantollm.runtime import ModelRuntime


@dataclass
class RegistryEntry:
    engine: InferenceEngine
    runtime: ModelRuntime


class EngineRegistry:
    def __init__(self) -> None:
        self._entries: dict[str, RegistryEntry] = {}

    def register(self, name: str, engine: InferenceEngine, runtime: ModelRuntime) -> None:
        if name in self._entries:
            raise ValueError(f"Model '{name}' is already registered")
        self._entries[name] = RegistryEntry(engine=engine, runtime=runtime)

    def get(self, name: str) -> RegistryEntry:
        return self._entries[name]

    def names(self) -> list[str]:
        return list(self._entries)

    async def start_all(self) -> None:
        for entry in self._entries.values():
            await entry.runtime.start()
            await entry.engine.start()

    async def shutdown_all(self) -> None:
        for entry in self._entries.values():
            await entry.engine.shutdown()
            await entry.runtime.shutdown()

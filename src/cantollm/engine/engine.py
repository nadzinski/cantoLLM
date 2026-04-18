"""Inference engine interface."""

from collections.abc import AsyncIterator
from typing import Protocol

from cantollm.engine.types import InferenceRequest, TokenEvent


class InferenceEngine(Protocol):
    """Submits a request and streams token events back.

    Async-generator `submit` returns an AsyncIterator synchronously (no await
    on call). Implementations may run requests sequentially or multiplex many
    through a batched scheduler; callers only see the event stream.

    Lifecycle hooks exist so a future batching implementation can own a
    background scheduler task. The sequential engine's start/shutdown are
    no-ops today but the interface is there so callers don't need to change.
    """

    async def start(self) -> None: ...

    async def shutdown(self) -> None: ...

    def submit(self, req: InferenceRequest) -> AsyncIterator[TokenEvent]: ...

    def abort(self, request_id: str) -> None: ...

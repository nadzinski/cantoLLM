"""Per-request event multiplexing — the API-facing half of the CB engine,
shared by both transports.

`EventMultiplexer` owns the pattern the shell established: `submit()`
registers an unbounded asyncio.Queue per request, `_dispatch` routes
engine-produced event batches into them on the event loop, `_fail` closes
everything batch-wide. What differs between `ContinuousBatchingEngine`
(scheduler thread in-process) and `EngineProcessClient` (scheduler in a
child process) is only where commands go — `_send_command` — and which
thread feeds `_dispatch`. Sharing the class keeps the two engines
behaviorally identical where the API layer can see them.

Threading rules (unchanged from the shell): `_queues` is touched ONLY on
the event loop — producers reach it via `call_soon_threadsafe`. Queues are
unbounded by design: a request's event count is capped by admission's
prompt+max_tokens bound, events are tiny, and disconnect→abort reclaims
the KV slot, which is the resource worth reclaiming.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

from cantollm.engine.batching.stats import EngineStatsAccumulator, StepUpdate
from cantollm.engine.batching.types import Abort, AddRequest, Command
from cantollm.engine.types import InferenceRequest, TokenEvent


class EventMultiplexer:
    def __init__(self) -> None:
        self._queues: dict[str, asyncio.Queue[TokenEvent | None]] = {}
        self._loop: asyncio.AbstractEventLoop | None = None
        self._failed: str | None = None
        # Bench-harness view (batching/stats.py): both engines' per-step
        # updates land here; /debug/engine-stats reads it.
        self.engine_stats = EngineStatsAccumulator()

    def _send_command(self, command: Command) -> None:
        raise NotImplementedError

    def abort(self, request_id: str) -> None:
        self._send_command(Abort(request_id))

    async def submit(self, req: InferenceRequest) -> AsyncIterator[TokenEvent]:
        rid = req.request_id
        if self._failed is not None:
            yield TokenEvent(error=self._failed, request_id=rid)
            return

        # Register the queue before the command goes in, so the request's
        # very first event can't race past an unregistered consumer.
        events: asyncio.Queue[TokenEvent | None] = asyncio.Queue()
        self._queues[rid] = events
        self._send_command(AddRequest(req))

        finished = False
        try:
            while (evt := await events.get()) is not None:
                yield evt
            finished = True
        finally:
            self._queues.pop(rid, None)
            if not finished:
                # Consumer went away mid-stream (disconnect) — free the slot.
                self._send_command(Abort(rid))

    # --- event loop side ----------------------------------------------

    def _dispatch_update(self, update: StepUpdate) -> None:
        """Per-step entry point: route the events, then record the stats.
        Order matters only for tests that assert on both — clients see
        events exactly as `_dispatch` always delivered them."""
        self._dispatch(update.events)
        self.engine_stats.record(update)

    def _dispatch(self, events: list[TokenEvent]) -> None:
        for evt in events:
            q = self._queues.get(evt.request_id)
            if q is None:
                continue  # consumer already disconnected; drop silently
            q.put_nowait(evt)
            if evt.finish_reason is not None or evt.error is not None:
                q.put_nowait(None)
                self._queues.pop(evt.request_id, None)

    def _fail(self, reason: str) -> None:
        # _failed holds the complete client-facing message; submit() and this
        # sweep both surface it verbatim.
        self._failed = f"engine failed: {reason}"
        for rid, q in list(self._queues.items()):
            q.put_nowait(TokenEvent(error=self._failed, request_id=rid))
            q.put_nowait(None)
        self._queues.clear()

    def _close_all_streams(self) -> None:
        """Shutdown sweep: nothing will produce events anymore, so close out
        every in-flight iterator so no submit() hangs on a queue that can't
        fill."""
        for rid, q in list(self._queues.items()):
            q.put_nowait(TokenEvent(finish_reason="abort", request_id=rid))
            q.put_nowait(None)
        self._queues.clear()

"""ContinuousBatchingEngine: one scheduler thread, N async consumers.

The shell half of the CB engine — everything except the scheduling policy,
which lives behind `SchedulerLike`. Implements the `InferenceEngine`
Protocol, so the API layer and registry can't tell it from SequentialEngine.

Shape (decisions 5 and 6 of the design note):

  submit()/abort()  ──AddRequest/Abort/Shutdown──▶  one thread-safe
                                                    command queue
                                                          │ drained at the
                                                          ▼ top of each step
                                                  scheduler thread:
                                                  apply commands → step()
                                                          │ one
                                                          ▼ call_soon_threadsafe
                                                    _dispatch on the loop
                                                          │ put_nowait
                                                          ▼
                              unbounded per-request asyncio.Queues → clients

Threading rules that keep this lock-free:
  - `_queues` is touched ONLY on the event loop (submit registers, _dispatch
    routes and closes, shutdown/_fail sweep). The scheduler thread never
    reads it.
  - The scheduler is touched ONLY by the scheduler thread.
  - The command queue is the only object both sides touch.

Backpressure: none, by design. Queues are unbounded (a request's event
count is capped by admission's prompt+max_tokens bound, and events are
tiny); a consumer that goes away triggers disconnect→abort, which frees the
KV slot — capacity, not memory, is the resource worth reclaiming.

Failure policy: an exception in `step()` is batch-wide by construction (one
shared forward), so every in-flight request gets an error event, the engine
marks itself failed, and later submits fail immediately.
"""

from __future__ import annotations

import asyncio
import logging
import queue
import threading
from collections.abc import AsyncIterator

from typing import TYPE_CHECKING

from cantollm.engine.batching.allocator import SlotAllocator
from cantollm.engine.batching.config import BatchingConfig
from cantollm.engine.batching.types import (
    Abort,
    AddRequest,
    Command,
    SchedulerLike,
    Shutdown,
)
from cantollm.engine.types import InferenceRequest, TokenEvent

if TYPE_CHECKING:
    from cantollm.runtime import ModelRuntime

logger = logging.getLogger(__name__)

_JOIN_TIMEOUT_S = 5.0


class ContinuousBatchingEngine:
    def __init__(self, scheduler: SchedulerLike):
        self.scheduler = scheduler
        self._commands: queue.Queue[Command] = queue.Queue()
        self._queues: dict[str, asyncio.Queue[TokenEvent | None]] = {}
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._failed: str | None = None

    @classmethod
    def from_runtime(
        cls, runtime: "ModelRuntime", config: BatchingConfig
    ) -> "ContinuousBatchingEngine":
        """The production composition: the runtime's batched-forward front,
        a freshly preallocated KV pool, and a fresh allocator behind the
        real scheduler. Tests inject a SchedulerLike directly instead."""
        from cantollm.engine.batching.scheduler import ContinuousBatchingScheduler

        scheduler = ContinuousBatchingScheduler(
            forward_fn=runtime.forward_batched,
            pool=runtime.new_kv_pool(config),
            allocator=SlotAllocator(config.max_batch),
            config=config,
        )
        return cls(scheduler)

    async def start(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._thread = threading.Thread(
            target=self._run, name="cb-scheduler", daemon=True
        )
        self._thread.start()

    async def shutdown(self) -> None:
        # Mark shut down first: a submit() arriving during or after shutdown
        # would otherwise register a queue and block forever on a command
        # queue no thread will drain. submit() checks _failed and fails fast.
        if self._failed is None:
            self._failed = "engine is shut down"
        if self._thread is None:
            return
        self._commands.put(Shutdown())
        await asyncio.to_thread(self._thread.join, _JOIN_TIMEOUT_S)
        # Nothing will produce events anymore: close out every in-flight
        # iterator so no submit() hangs on a queue that can't fill.
        for rid, q in list(self._queues.items()):
            q.put_nowait(TokenEvent(finish_reason="abort", request_id=rid))
            q.put_nowait(None)
        self._queues.clear()

    def abort(self, request_id: str) -> None:
        self._commands.put(Abort(request_id))

    async def submit(self, req: InferenceRequest) -> AsyncIterator[TokenEvent]:
        rid = req.request_id
        if self._failed is not None:
            yield TokenEvent(error=self._failed, request_id=rid)
            return

        # Register the queue before the command goes in, so the request's
        # very first event can't race past an unregistered consumer.
        events: asyncio.Queue[TokenEvent | None] = asyncio.Queue()
        self._queues[rid] = events
        self._commands.put(AddRequest(req))

        finished = False
        try:
            while (evt := await events.get()) is not None:
                yield evt
            finished = True
        finally:
            self._queues.pop(rid, None)
            if not finished:
                # Consumer went away mid-stream (disconnect) — free the slot.
                self._commands.put(Abort(rid))

    # --- scheduler thread ---------------------------------------------

    def _run(self) -> None:
        while True:
            commands: list[Command] = []
            if self.scheduler.is_idle():
                # Nothing to step: block until the world says something.
                commands.append(self._commands.get())
            while True:
                try:
                    commands.append(self._commands.get_nowait())
                except queue.Empty:
                    break

            for cmd in commands:
                if isinstance(cmd, Shutdown):
                    return
                if isinstance(cmd, AddRequest):
                    self.scheduler.add_request(cmd.request)
                elif isinstance(cmd, Abort):
                    self.scheduler.abort(cmd.request_id)

            if self.scheduler.is_idle():
                continue

            try:
                events = self.scheduler.step()
            except Exception as exc:  # batch-wide by construction
                # Log with the traceback here, on the scheduler thread where
                # it happened — _fail only carries the message to clients, so
                # without this the stack of a batch-wide failure is lost.
                logger.exception("scheduler step failed; failing the engine")
                self._loop.call_soon_threadsafe(self._fail, str(exc))
                return

            if events:
                # One loop hop per step, not per token (IPC-shaped).
                self._loop.call_soon_threadsafe(self._dispatch, events)

    # --- event loop side ----------------------------------------------

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

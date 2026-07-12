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
                                                  drive_scheduler()
                                                          │ one
                                                          ▼ call_soon_threadsafe
                                                    _dispatch on the loop
                                                          │ put_nowait
                                                          ▼
                              unbounded per-request asyncio.Queues → clients

The API-facing half (submit/dispatch/fail, and the threading and
backpressure rules) lives in `EventMultiplexer` — shared with the
process-split client in `process.py`, which runs the same `drive_scheduler`
loop in a child process instead of a thread.

Failure policy: an exception in `step()` is batch-wide by construction (one
shared forward), so every in-flight request gets an error event, the engine
marks itself failed, and later submits fail immediately.
"""

from __future__ import annotations

import asyncio
import logging
import queue
import threading
from collections.abc import Callable

from typing import TYPE_CHECKING

from cantollm.engine.batching.allocator import SlotAllocator
from cantollm.engine.batching.config import BatchingConfig
from cantollm.engine.batching.mux import EventMultiplexer
from cantollm.engine.batching.types import (
    Abort,
    AddRequest,
    Command,
    SchedulerLike,
    Shutdown,
)
from cantollm.engine.types import TokenEvent

if TYPE_CHECKING:
    from cantollm.runtime import ModelRuntime

logger = logging.getLogger(__name__)

_JOIN_TIMEOUT_S = 5.0
_IDLE_POLL_S = 0.5


def scheduler_from_runtime(
    runtime: "ModelRuntime", config: BatchingConfig
) -> SchedulerLike:
    """The production composition: the runtime's batched-forward front, a
    freshly preallocated KV pool, and a fresh allocator behind the real
    scheduler. Used by `ContinuousBatchingEngine.from_runtime` in-process
    and by the engine-process factory after the split."""
    from cantollm.engine.batching.scheduler import ContinuousBatchingScheduler

    return ContinuousBatchingScheduler(
        forward_fn=runtime.forward_batched,
        pool=runtime.new_kv_pool(config),
        allocator=SlotAllocator(config.max_batch),
        config=config,
    )


def drive_scheduler(
    scheduler: SchedulerLike,
    commands,
    emit: Callable[[list[TokenEvent]], None],
    should_stop: Callable[[], bool] | None = None,
) -> None:
    """The engine's steady-state loop: drain commands, apply, step, emit —
    until a Shutdown command arrives (returns) or the scheduler raises
    (propagates; the caller owns failure policy).

    Runs on the in-process scheduler thread and, after the process split,
    as the body of the engine process — `commands` only needs the stdlib
    get/get_nowait surface, which `queue.Queue` and `multiprocessing.Queue`
    both provide.

    `should_stop` is the process split's orphan guard: when set, the idle
    block becomes a poll and the loop re-checks it each iteration, so an
    engine whose API process died stops stepping instead of generating into
    a pipe nobody drains. When None (in-process), idle blocks indefinitely.
    """
    while True:
        if should_stop is not None and should_stop():
            return
        batch: list[Command] = []
        if scheduler.is_idle():
            # Nothing to step: block until the world says something.
            if should_stop is None:
                batch.append(commands.get())
            else:
                try:
                    batch.append(commands.get(timeout=_IDLE_POLL_S))
                except queue.Empty:
                    continue  # loop around to re-check should_stop
        while True:
            try:
                batch.append(commands.get_nowait())
            except queue.Empty:
                break

        for cmd in batch:
            if isinstance(cmd, Shutdown):
                return
            if isinstance(cmd, AddRequest):
                scheduler.add_request(cmd.request)
            elif isinstance(cmd, Abort):
                scheduler.abort(cmd.request_id)

        if scheduler.is_idle():
            continue

        events = scheduler.step()
        if events:
            # One emission per step, not per token (IPC-shaped).
            emit(events)


class ContinuousBatchingEngine(EventMultiplexer):
    def __init__(self, scheduler: SchedulerLike):
        super().__init__()
        self.scheduler = scheduler
        self._commands: queue.Queue[Command] = queue.Queue()
        self._thread: threading.Thread | None = None

    @classmethod
    def from_runtime(
        cls, runtime: "ModelRuntime", config: BatchingConfig
    ) -> "ContinuousBatchingEngine":
        """Compose the production scheduler in-process. Tests inject a
        SchedulerLike directly instead."""
        return cls(scheduler_from_runtime(runtime, config))

    def _send_command(self, command: Command) -> None:
        self._commands.put(command)

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
        self._close_all_streams()

    # --- scheduler thread ---------------------------------------------

    def _run(self) -> None:
        try:
            drive_scheduler(
                self.scheduler,
                self._commands,
                emit=lambda events: self._loop.call_soon_threadsafe(
                    self._dispatch, events
                ),
            )
        except Exception as exc:  # batch-wide by construction
            # Log with the traceback here, on the scheduler thread where
            # it happened — _fail only carries the message to clients, so
            # without this the stack of a batch-wide failure is lost.
            logger.exception("scheduler step failed; failing the engine")
            self._loop.call_soon_threadsafe(self._fail, str(exc))

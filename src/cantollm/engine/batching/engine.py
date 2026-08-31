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

from cantollm import progress
from cantollm.engine.batching.allocator import BlockAllocator, SlotAllocator
from cantollm.engine.batching.config import BatchingConfig
from cantollm.engine.batching.mux import EventMultiplexer
from cantollm.engine.batching.stats import StepStatsCollector, StepUpdate
from cantollm.engine.batching.types import (
    Abort,
    AddRequest,
    Command,
    SchedulerLike,
    Shutdown,
)

if TYPE_CHECKING:
    from cantollm.engine.batching.trace import TraceStepObserver
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

    if (
        config.paged_kv
        and runtime.device.type == "cuda"
        and not config.torch_compile
    ):
        # Device-coupled rule the device-blind config cannot enforce:
        # FlexAttention only performs compiled (flex-spike-results.md §7),
        # so a paged CUDA engine without compile would silently serve the
        # slow eager kernel. Fail the build loudly instead, like the
        # graphs-off-CUDA check in capture_decode_shapes.
        raise RuntimeError(
            "paged_kv on CUDA requires torch_compile "
            "(paged-kv-plan.md §2.8)"
        )
    if config.paged_kv and runtime.device.type == "cuda":
        from cantollm.models.attention.flex import MIN_CUDA_KV_BLOCK

        if config.block_size < MIN_CUDA_KV_BLOCK:
            # Same device-coupled shape as the compile guard above: the
            # compiled Flex kernels prune every template below this KV
            # block size (5090 probe, 2026-08-30), which otherwise
            # surfaces as a cryptic NoValidChoicesError mid-warm-up.
            raise RuntimeError(
                f"paged_kv on CUDA requires block_size >= "
                f"{MIN_CUDA_KV_BLOCK}, got {config.block_size} "
                "(paged-kv-plan.md §2.13)"
            )
    from cantollm.models.attention.flex import FlexAttentionMethod

    method = getattr(getattr(runtime, "model", None), "attention_method", None)
    if config.paged_kv != isinstance(method, FlexAttentionMethod):
        # The layout and the read path select each other: flex reads
        # block tables the padded pool never has, and the padded/sdpa
        # methods read slot geometry the paged pool never has. A mismatch
        # would otherwise surface as a mid-traffic "no paged tables" (or
        # shape) error; fail the build instead.
        raise RuntimeError(
            f"attention method {type(method).__name__} does not match "
            f"paged_kv={config.paged_kv}: the paged pool requires flex "
            "attention and vice versa (--attention flex implies the "
            "paged stack)"
        )
    pool = runtime.new_kv_pool(config)
    forward_fn = runtime.forward_batched
    block_allocator = None
    paged_state = None
    if config.paged_kv:
        # The paged trio lives with the scheduler (paged-kv-plan.md §2.5):
        # the allocator owns which blocks are in use, the step state owns
        # the persistent device tables the per-step metas reference, plus
        # the per-family mask cache. Built BEFORE the warm-up sweep, whose
        # paged branch fills these same buffers (and populates the mask
        # cache) so no artifact guards on tensors traffic never presents.
        # The mask builder is the attention method's, injected here
        # because only assembly sees both sides; a runtime without one
        # (padded-attention tests) skips mask caching and the method
        # falls back to per-step construction.
        from cantollm.engine.batching.paging import PagedStepState

        method = getattr(runtime.model, "attention_method", None)
        block_allocator = BlockAllocator(config.resolved_kv_blocks)
        paged_state = PagedStepState(
            max_rows=config.max_batch,
            max_blocks_per_seq=config.max_seq_len // config.block_size,
            num_kv_blocks=config.resolved_kv_blocks,
            device=runtime.device,
            mask_builder=getattr(method, "build_family_mask", None),
        )
    if config.torch_compile:
        # Before the warm-up sweep, so the sweep's per-shape forwards are
        # what build the compiled artifacts (and, with cuda_graphs on,
        # capture then records the fused kernels). Ordering per
        # torch-compile-design.md §3.4.
        runtime.enable_torch_compile(strategy=config.torch_compile_strategy)
        logger.info("torch.compile enabled on the batched forward "
                    "(fullgraph, strategy=%s, artifacts built during "
                    "warm-up)", config.torch_compile_strategy)
        # One tick, honestly named: this stage only *enables* compile; the
        # artifacts get built by the sweep's forwards.
        progress.report("compile", 1, 1, "torch.compile enabled")
    if config.warmup_shapes:
        # Behind Ready in the process split (the factory runs before the
        # Ready handshake) and before from_runtime returns in-process: no
        # request can reach a shape the kernel hasn't already seen.
        from cantollm.engine.batching.warmup import warmup_shape_vocabulary

        warmup_shape_vocabulary(
            runtime.forward_batched, pool, config, paged_state=paged_state
        )
    if config.cuda_graphs:
        # Strictly after the eager warm-up (config validation enforces the
        # pairing): capture must record warm kernel choices, not one-time
        # setup. Also behind Ready, so instantiation cost joins the same
        # startup bill as the plan compiles (cuda-graphs-design.md §3).
        import time as _time

        from cantollm.engine.batching.graphs import GraphedBatchedForward

        graphed = GraphedBatchedForward(
            runtime.forward_batched, config, paged_state=paged_state
        )
        t0 = _time.perf_counter()
        captured = graphed.capture_decode_shapes(pool)
        logger.info(
            "CUDA graphs: captured %d decode shapes in %.1f s",
            captured, _time.perf_counter() - t0,
        )
        forward_fn = graphed
    return ContinuousBatchingScheduler(
        forward_fn=forward_fn,
        pool=pool,
        allocator=SlotAllocator(config.max_batch),
        config=config,
        block_allocator=block_allocator,
        paged_state=paged_state,
    )


def drive_scheduler(
    scheduler: SchedulerLike,
    commands,
    emit: Callable[[StepUpdate], None],
    should_stop: Callable[[], bool] | None = None,
    collector: StepStatsCollector | None = None,
    tracer: "TraceStepObserver | None" = None,
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

    `collector` (bench instrumentation, see batching/stats.py) snapshots
    scheduler state around each step; with one, every step emits — a
    prefill-only step carries no events but its stats still matter. With
    None the emission rule is unchanged: only steps with events emit.
    """
    # Test-only fault injection (chaos suite): wedge the loop after N steps
    # (the hung-but-alive engine only the watchdog can catch), or pace every
    # step (the tiny model otherwise outruns any outside-observer timing).
    import os as _os
    import time as _time

    wedge_after = int(_os.environ.get("CANTOLLM_TEST_WEDGE_AFTER_STEPS") or 0)
    step_delay = float(_os.environ.get("CANTOLLM_TEST_STEP_DELAY_S") or 0)
    steps_done = 0
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
                if tracer is not None:
                    tracer.on_request(cmd.request)
                scheduler.add_request(cmd.request)
            elif isinstance(cmd, Abort):
                scheduler.abort(cmd.request_id)

        if scheduler.is_idle():
            continue

        if collector is not None:
            collector.before_step(scheduler)
        if tracer is not None:
            tracer.before_step(scheduler)
        events = scheduler.step()
        stats = collector.after_step(scheduler, events) if collector is not None else None
        if tracer is not None:
            tracer.after_step(scheduler, events)
        steps_done += 1
        if wedge_after and steps_done >= wedge_after:
            logger.error("test wedge engaged after %d steps; sleeping", steps_done)
            _time.sleep(3600)
        if step_delay:
            _time.sleep(step_delay)
        if events or stats is not None:
            # One emission per step, not per token (IPC-shaped).
            emit(StepUpdate(events=events, stats=stats))


class ContinuousBatchingEngine(EventMultiplexer):
    def __init__(self, scheduler: SchedulerLike):
        super().__init__()
        self.scheduler = scheduler
        self._commands: queue.Queue[Command] = queue.Queue()
        self._thread: threading.Thread | None = None
        self.engine_stats.engine_kind = "batched-inprocess"
        config = getattr(scheduler, "config", None)
        if config is not None:
            self.engine_stats.max_batch = config.max_batch
            self.engine_stats.max_seq_len = config.max_seq_len

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
        from cantollm.engine.batching.trace import TraceStepObserver

        try:
            drive_scheduler(
                self.scheduler,
                self._commands,
                emit=lambda update: self._loop.call_soon_threadsafe(
                    self._dispatch_update, update
                ),
                collector=StepStatsCollector.for_scheduler(self.scheduler),
                tracer=TraceStepObserver.create(),
            )
        except Exception as exc:  # batch-wide by construction
            # Log with the traceback here, on the scheduler thread where
            # it happened — _fail only carries the message to clients, so
            # without this the stack of a batch-wide failure is lost.
            logger.exception("scheduler step failed; failing the engine")
            self._loop.call_soon_threadsafe(self._fail, str(exc))

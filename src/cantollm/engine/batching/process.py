"""The CB engine behind a process boundary: child main + API-side client.

Phase 2 item (2). The engine is a steady-state busy loop that wants its own
process: the scheduler thread becomes `engine_process_main` in a spawned
child, and the API keeps `EngineProcessClient`, which implements the
`InferenceEngine` Protocol — routers and registry can't tell it from the
in-process engine.

IPC is two multiprocessing queues (spawn context — fork is unsafe once CUDA
enters the picture, and macOS defaults to spawn anyway):

  submit()/abort() ──AddRequest/Abort/Shutdown──▶ command queue ──▶ child:
                                                  drive_scheduler()
                                                        │ one put per step:
                                                        ▼ StepUpdate(events, stats)
  per-request asyncio.Queues ◀─call_soon_threadsafe─ bridge thread

Wire protocol child → parent, in order: zero or more `Progress` ticks while
the factory builds (weight load, warm-up sweep, graph capture — forwarded
to the lifecycle handle so /ready can show them), then `Ready(load_seconds)`
once the scheduler is built (start() blocks on it), then per-step
`StepUpdate`s — the step's token events plus its bench stats record
(batching/stats.py); one pickle per step, never per token — then exactly
one farewell: `EngineFailed` (load or step failure; the child exits) or
`Stopped` (Shutdown acknowledged).

Nothing rich crosses the boundary: the child rebuilds spec/runtime from
primitives via a module-level factory (`ModelSpec` carries closures that
don't pickle, and weights shouldn't ship over a pipe anyway), and the
parent keeps a `TokenizerRuntime` — tokenization stays API-side per
Phase 1a, which is exactly what keeps prompt encoding off the scheduler's
core.

Liveness, both directions, without a heartbeat protocol: the bridge thread
polls the event queue with a timeout and fails all in-flight streams if the
child died without a farewell (segfault, OOM-kill); the child's drive loop
polls its command queue and exits if the parent died (`daemon=True` only
covers a clean parent exit, not a SIGKILLed one), so an orphaned engine
can't keep generating into a pipe nobody drains.
"""

from __future__ import annotations

import asyncio
import logging
import multiprocessing as mp
import queue
import signal
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from cantollm.engine.batching.config import BatchingConfig
from cantollm.engine.batching.engine import drive_scheduler, scheduler_from_runtime
from cantollm.engine.batching.mux import EventMultiplexer
from cantollm.engine.batching.stats import StepStatsCollector, StepUpdate
from cantollm.engine.batching.types import Command, SchedulerLike, Shutdown
from cantollm.progress import Progress, current_sink

logger = logging.getLogger(__name__)

_JOIN_TIMEOUT_S = 5.0
_STARTUP_POLL_S = 1.0
_BRIDGE_POLL_S = 0.5

SchedulerFactory = Callable[..., SchedulerLike]


# ── wire messages, child → parent ───────────────────────────────────


@dataclass(frozen=True)
class Ready:
    """Scheduler built, model loaded, commands welcome.

    Carries the bench harness's engine metadata (bench-spec.md §5): how
    long the factory (weights download + load) took, and the built
    scheduler's capacity — the child reports what it actually constructed
    rather than the parent guessing from factory kwargs.
    """

    load_seconds: float = 0.0
    max_batch: int | None = None
    max_seq_len: int | None = None


@dataclass(frozen=True)
class EngineFailed:
    """Terminal failure (scheduler build or step); the child exits after."""

    reason: str


@dataclass(frozen=True)
class Stopped:
    """Shutdown acknowledged; nothing follows."""


# ── engine process ──────────────────────────────────────────────────


def build_qwen3_batched_scheduler(
    size: str, device: str, config: BatchingConfig, attention: str = "padded"
) -> SchedulerLike:
    """Production factory — runs inside the engine process: download/load
    weights onto `device` and compose the real scheduler. `attention` picks
    the batched attention method ("padded" or "sdpa")."""
    # Imported here, not at module top: runtime.py imports this package for
    # BatchingConfig, so a module-level import would be circular.
    import torch

    from cantollm.runtime import build_runtime
    from cantollm.spec import qwen3_spec

    runtime = build_runtime(qwen3_spec(size), torch.device(device), attention=attention)
    return scheduler_from_runtime(runtime, config)


def engine_process_main(
    scheduler_factory: SchedulerFactory,
    factory_kwargs: dict[str, Any],
    commands: mp.Queue,
    events: mp.Queue,
) -> None:
    """Engine-process entry point: build the scheduler via the factory, then
    run the same drive loop the in-process engine runs on its thread."""
    # The parent owns Ctrl-C: uvicorn turns SIGINT into a clean lifespan
    # shutdown, which reaches us as a Shutdown command. The raw signal (the
    # terminal delivers it to the whole process group) would otherwise kill
    # the child mid-step.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    # spawn starts from a blank interpreter: the parent's logging config
    # didn't come along. Install the same JSON handler the API process
    # uses so both processes speak one format.
    from cantollm.obs.logging import configure_logging
    from cantollm.obs.tracing import configure_from_env, shutdown_tracing
    from cantollm.progress import bind_sink, unbind_sink

    configure_logging("engine")
    # The parent exports OTEL_EXPORTER_OTLP_ENDPOINT when --otlp-endpoint
    # is set; spawn inherits the environment, so the child builds its own
    # provider and exports engine-side spans directly.
    configure_from_env("cantollm-engine")
    parent = mp.parent_process()

    load_start = time.perf_counter()
    # Progress messages ride the event queue ahead of Ready: everything the
    # factory reports (weight load, sweep, capture) reaches the parent's
    # /ready while the API is already serving.
    sink_token = bind_sink(events.put)
    try:
        scheduler = scheduler_factory(**factory_kwargs)
    except Exception as exc:
        logger.exception("engine process failed to build its scheduler")
        events.put(EngineFailed(f"engine process failed to start: {exc}"))
        return
    finally:
        unbind_sink(sink_token)

    config = getattr(scheduler, "config", None)
    events.put(Ready(
        load_seconds=time.perf_counter() - load_start,
        max_batch=getattr(config, "max_batch", None),
        max_seq_len=getattr(config, "max_seq_len", None),
    ))
    from cantollm.engine.batching.trace import TraceStepObserver

    try:
        drive_scheduler(
            scheduler,
            commands,
            emit=events.put,
            should_stop=lambda: not parent.is_alive(),
            collector=StepStatsCollector.for_scheduler(scheduler),
            tracer=TraceStepObserver.create(),
        )
    except Exception as exc:  # batch-wide by construction
        logger.exception("scheduler step failed; engine process exiting")
        events.put(EngineFailed(str(exc)))
        return
    finally:
        shutdown_tracing()  # flush batched spans before the child exits
    events.put(Stopped())


# ── API-process side ────────────────────────────────────────────────


class EngineProcessClient(EventMultiplexer):
    """The API process's handle on a spawned engine process.

    submit/abort/dispatch semantics are shared with the in-process engine
    via `EventMultiplexer`; what differs is that commands cross a pipe and
    events come back through the bridge thread.
    """

    def __init__(
        self,
        scheduler_factory: SchedulerFactory,
        factory_kwargs: dict[str, Any] | None = None,
        on_progress: Callable[[Progress], None] | None = None,
    ):
        super().__init__()
        self._factory = scheduler_factory
        self._factory_kwargs = dict(factory_kwargs or {})
        self._ctx = mp.get_context("spawn")
        self._commands: mp.Queue | None = None
        self._events: mp.Queue | None = None
        self._proc: mp.Process | None = None
        self._bridge: threading.Thread | None = None
        # Called (from the waiter thread) with each pre-Ready Progress
        # message; must be thread-safe. When None, start() adopts the
        # ambient progress sink of the context it runs in, which is how
        # the lifecycle handle's sink reaches us without plumbing the
        # handle through the factory closure.
        self._on_progress = on_progress
        self.engine_stats.engine_kind = "batched-split"

    def _send_command(self, command: Command) -> None:
        try:
            self._commands.put(command)
        except ValueError:
            # Queue already closed by shutdown(): the engine is gone and
            # every stream has been closed out — a late disconnect's Abort
            # has nobody left to inform.
            pass

    async def start(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._commands = self._ctx.Queue()
        self._events = self._ctx.Queue()
        self._proc = self._ctx.Process(
            target=engine_process_main,
            args=(self._factory, self._factory_kwargs, self._commands, self._events),
            name="cantollm-engine",
            daemon=True,
        )
        self._proc.start()
        # No explicit on_progress: adopt the ambient sink (the lifecycle
        # handle binds one around this start()), so warm-up Progress from
        # the child reaches /ready without extra plumbing.
        if self._on_progress is None:
            self._on_progress = current_sink()
        # Model loading happens behind this wait — a first run can be a long
        # weights download, so no overall deadline; only child death (or an
        # EngineFailed farewell) breaks the wait.
        await asyncio.to_thread(self._wait_until_ready)
        self._bridge = threading.Thread(
            target=self._bridge_loop, name="engine-bridge", daemon=True
        )
        self._bridge.start()

    def _wait_until_ready(self) -> None:
        while True:
            try:
                msg = self._events.get(timeout=_STARTUP_POLL_S)
            except queue.Empty:
                if not self._proc.is_alive():
                    self._failed = (
                        "engine process died during startup "
                        f"(exit code {self._proc.exitcode})"
                    )
                    raise RuntimeError(self._failed)
                continue
            if isinstance(msg, Progress):
                # Runs on this waiter thread; the sink contract requires
                # thread-safety, so forward directly.
                if self._on_progress is not None:
                    self._on_progress(msg)
                continue
            if isinstance(msg, Ready):
                self.engine_stats.load_seconds = msg.load_seconds
                self.engine_stats.max_batch = msg.max_batch
                self.engine_stats.max_seq_len = msg.max_seq_len
                return
            if isinstance(msg, EngineFailed):
                self._failed = msg.reason
                raise RuntimeError(msg.reason)

    def kill(self) -> None:
        """Watchdog escalation: SIGKILL the child — a hung child may ignore
        SIGTERM, and the point is that it stopped listening. The bridge
        notices the death, fails in-flight streams, and the supervisor
        rebuilds. Presence of this method is also the watchdog's arming
        condition (a thread cannot be killed, so the in-process engine has
        no equivalent)."""
        if self._proc is not None:
            self._proc.kill()

    async def shutdown(self) -> None:
        # Same latch-first rule as in-process: a submit() arriving during or
        # after shutdown must fail fast, not hang on a dead engine.
        if self._failed is None:
            self._failed = "engine is shut down"
        if self._proc is None:
            return
        self._commands.put(Shutdown())
        await asyncio.to_thread(self._proc.join, _JOIN_TIMEOUT_S)
        if self._proc.is_alive():
            logger.warning("engine process ignored Shutdown; terminating it")
            self._proc.terminate()
            await asyncio.to_thread(self._proc.join, _JOIN_TIMEOUT_S)
        if self._bridge is not None:
            # Exits on the child's farewell (Stopped/EngineFailed) or on
            # noticing the process died; both are guaranteed by this point.
            await asyncio.to_thread(self._bridge.join, _JOIN_TIMEOUT_S)
        self._close_all_streams()
        for q in (self._commands, self._events):
            q.close()
            q.cancel_join_thread()

    # --- bridge thread ------------------------------------------------

    def _bridge_loop(self) -> None:
        """Forward child messages onto the event loop until a farewell or
        child death. Mirrors the in-process scheduler thread's dispatch: one
        loop hop per step batch."""
        while True:
            try:
                msg = self._events.get(timeout=_BRIDGE_POLL_S)
            except queue.Empty:
                if not self._proc.is_alive():
                    self._loop.call_soon_threadsafe(
                        self._fail,
                        f"engine process died (exit code {self._proc.exitcode})",
                    )
                    return
                continue
            if isinstance(msg, StepUpdate):
                self._loop.call_soon_threadsafe(self._dispatch_update, msg)
            elif isinstance(msg, Progress):
                # Defensive: all Progress precedes Ready by construction,
                # so the bridge should never see one. Drop stragglers.
                continue
            elif isinstance(msg, EngineFailed):
                self._loop.call_soon_threadsafe(self._fail, msg.reason)
                return
            elif isinstance(msg, Stopped):
                return

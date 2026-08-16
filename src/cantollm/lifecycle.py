"""Engine lifecycle: the per-model state machine between the API and its engine.

An `EngineHandle` is the stable identity the registry hands out; the engine
object behind it is disposable. That split exists because the event
multiplexer's failure latch is one-way by design: once an engine has failed
(or shut down), the only way back is a fresh engine object, which the
supervisor builds from the handle's factory closure and swaps in. The handle
carries everything that must survive that swap: state, generation counter,
warm-up progress, failure accounting, and the API-side in-flight counter.

Threading contract: every mutation of handle state happens on the event
loop. The factory runs in a worker thread (`asyncio.to_thread`), and updates
originating there (warm-up progress, child death) hop over via
`call_soon_threadsafe`. Routers therefore read `state`/`engine` lock-free
and must capture the engine object once per request (`ensure_ready()`
returns it) so a mid-request generation swap never mixes generations.

This chunk (3.5/2a) implements first start only: build -> warm -> ready,
with any failure landing in CRASHED. Backoff, restart wakes, reload, drain,
and the watchdog arrive in later chunks and extend the supervisor loop.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Callable

from cantollm.progress import Progress, bind_sink, unbind_sink

logger = logging.getLogger(__name__)


class EngineState(str, Enum):
    STARTING = "starting"      # factory not yet run, or build in flight
    WARMING = "warming"        # engine object exists, pre-Ready
    READY = "ready"
    DRAINING = "draining"      # signal drain or reload drain
    RESTARTING = "restarting"  # supervisor rebuilding after a failure
    CRASHED = "crashed"        # gave up; manual restart only
    STOPPED = "stopped"


@dataclass
class BuiltEngine:
    """What a handle factory returns: one engine and the runtime it serves.

    For the subprocess engine the runtime is the eagerly built
    TokenizerRuntime (identical object every generation); for in-process
    engines the factory builds the full ModelRuntime fresh each time.
    """

    engine: Any
    runtime: Any


class NotReadyError(Exception):
    """The model exists but cannot take requests right now -> HTTP 503.

    Raised by `EngineHandle.ensure_ready()`; the API error handlers render
    it as the dialect-correct 503 envelope with a Retry-After header.
    """

    def __init__(self, model: str, state: EngineState, detail: str,
                 retry_after_s: int = 5):
        super().__init__(detail)
        self.model = model
        self.state = state
        self.detail = detail
        self.retry_after_s = retry_after_s


class RequestTicket:
    """One admitted request's claim on the in-flight counter.

    `close()` is idempotent: the stream wrapper closes it on exhaustion or
    disconnect, and the router closes it on pre-submit failures; whichever
    runs first wins, so the counter can never go negative.
    """

    __slots__ = ("_handle", "_closed")

    def __init__(self, handle: "EngineHandle | None"):
        self._handle = handle
        self._closed = False

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._handle is not None:
            self._handle._inflight -= 1


class EngineHandle:
    def __init__(
        self,
        name: str,
        factory: Callable[[], BuiltEngine],
        *,
        runtime: Any | None = None,
    ):
        self.name = name
        self.factory = factory
        self.state = EngineState.STARTING
        self.engine: Any | None = None
        self.runtime: Any | None = runtime
        self._eager_runtime = runtime is not None
        self.generation = 0
        self.progress: Progress | None = None
        self.last_error: str | None = None
        self.consecutive_failures = 0
        self.supervisor_task: asyncio.Task | None = None
        self._inflight = 0
        self._pending: BuiltEngine | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._drain_latched = False

    # --- request-path API (event loop only) ----------------------------

    @property
    def inflight(self) -> int:
        return self._inflight

    def ensure_ready(self) -> Any:
        if self.state is EngineState.READY and self.engine is not None:
            return self.engine
        raise NotReadyError(self.name, self.state, self._not_ready_detail())

    def begin_request(self) -> RequestTicket:
        self._inflight += 1
        return RequestTicket(self)

    def _not_ready_detail(self) -> str:
        # Progress can flow in STARTING too: the in-process factory does
        # the whole build (load + sweep + capture) before the handle ever
        # reaches WARMING.
        building = self.state in (EngineState.STARTING, EngineState.WARMING)
        if building and self.progress is not None:
            p = self.progress
            total = f"/{p.total}" if p.total is not None else ""
            return (f"model '{self.name}' is warming up "
                    f"({p.stage} {p.done}{total})")
        if self.state is EngineState.CRASHED:
            return (f"model '{self.name}' engine crashed: {self.last_error}; "
                    "POST /admin/restart to retry")
        if self.state is EngineState.DRAINING:
            return "server is draining"
        if self.state is EngineState.STOPPED:
            return "server is shutting down"
        return f"model '{self.name}' is {self.state.value}"

    def begin_drain(self) -> None:
        """Server is going away: stop admitting, permanently. Latched so a
        build finishing mid-drain cannot flip the handle back to READY and
        re-open admission behind the drain's back."""
        self._drain_latched = True
        self.state = EngineState.DRAINING

    def status(self) -> dict:
        s: dict[str, Any] = {
            "state": self.state.value,
            "generation": self.generation,
            "last_error": self.last_error,
            "consecutive_failures": self.consecutive_failures,
        }
        if self.progress is not None:
            s["progress"] = asdict(self.progress)
        if self.state is EngineState.CRASHED:
            s["hint"] = "POST /admin/restart"
        return s

    # --- progress -------------------------------------------------------

    def _progress_sink(self, p: Progress) -> None:
        """Sink callback; may run on any thread (factory worker, the IPC
        waiter). Hops onto the loop so handle state stays loop-confined."""
        if self._loop is not None:
            self._loop.call_soon_threadsafe(self._apply_progress, p)

    def _apply_progress(self, p: Progress) -> None:
        self.progress = p

    # --- lifecycle ------------------------------------------------------

    def launch(self) -> None:
        """Spawn the supervisor task. Call from a running event loop."""
        self._loop = asyncio.get_running_loop()
        self.supervisor_task = self._loop.create_task(
            self._supervise(), name=f"engine-supervisor-{self.name}"
        )

    async def _supervise(self) -> None:
        try:
            await self._build_and_start()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.exception("engine build failed for %s", self.name)
            self.last_error = str(exc)
            self.consecutive_failures += 1
            self.state = EngineState.CRASHED
            return
        if self._drain_latched:
            self.state = EngineState.DRAINING
            logger.info(
                "engine for %s finished building mid-drain; not admitting",
                self.name,
            )
            return
        self.state = EngineState.READY
        logger.info("engine ready: %s (generation %d)", self.name, self.generation)

    async def _build_and_start(self) -> None:
        self.state = EngineState.STARTING
        t0 = time.perf_counter()
        # Bind the progress sink for the whole build: asyncio.to_thread
        # copies this context, so an in-process factory reports straight
        # through it, and the subprocess client's start() picks it up as
        # the ambient sink to forward child-side Progress into.
        sink_token = bind_sink(self._progress_sink)
        try:
            built = await asyncio.to_thread(self.factory)
            # Reachable from stop(): if we are cancelled during start()
            # below, the partially started engine still gets shut down
            # (which reaps a spawned child even mid-warm-up via
            # join-then-terminate).
            self._pending = built
            self.state = EngineState.WARMING
            await built.runtime.start()
            await built.engine.start()
        finally:
            unbind_sink(sink_token)
        self._pending = None
        self.progress = None
        self.engine = built.engine
        if not self._eager_runtime:
            self.runtime = built.runtime
        self.generation += 1
        logger.info(
            "engine for %s built and warm in %.1f s",
            self.name, time.perf_counter() - t0,
        )

    async def stop(self) -> None:
        self.state = EngineState.STOPPED
        if self.supervisor_task is not None:
            self.supervisor_task.cancel()
            try:
                await self.supervisor_task
            except asyncio.CancelledError:
                pass
            self.supervisor_task = None
        if self._pending is not None:
            try:
                await self._pending.engine.shutdown()
            except Exception:
                logger.exception("shutdown of half-started engine failed")
            self._pending = None
        if self.engine is not None:
            await self.engine.shutdown()
        if self.runtime is not None:
            await self.runtime.shutdown()

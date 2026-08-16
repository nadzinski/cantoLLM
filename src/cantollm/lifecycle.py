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

The supervisor task per handle IS the state machine: build -> warm -> ready
-> await a wake (engine died | reload | restart | stop). First startup is
simply the loop's first iteration, which is why a bad model no longer kills
the process: build failures retry with capped exponential backoff and land
in CRASHED after enough consecutive failures, recoverable via
POST /admin/restart. Reload and restart-from-ready drain this handle's
in-flight requests, fully shut the old engine down (child joined, queues
closed — never two engines' VRAM at once), and rebuild through the same
factory.
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

# Supervisor policy defaults (instance attributes; tests shrink them).
BACKOFF_INITIAL_S = 1.0
BACKOFF_FACTOR = 2.0
BACKOFF_CAP_S = 30.0
GIVE_UP_AFTER = 5          # consecutive failures -> CRASHED
STABLE_RESET_S = 60.0      # ready at least this long -> death resets the count
DRAIN_POLL_S = 0.05
ABORT_GRACE_S = 2.0


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


class AdmissionBusyError(Exception):
    """The model is serving at capacity and the admission queue timed out
    -> HTTP 429 (rate_limit_error in both dialects) with Retry-After."""

    def __init__(self, model: str, detail: str, retry_after_s: int):
        super().__init__(detail)
        self.model = model
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
            if self._handle._sem is not None:
                self._handle._sem.release()


class EngineHandle:
    def __init__(
        self,
        name: str,
        factory: Callable[[], BuiltEngine],
        *,
        runtime: Any | None = None,
        drain_timeout_s: float = 30.0,
        watchdog_timeout_s: float = 60.0,
        max_inflight: int | None = None,
        admission_timeout_s: float = 30.0,
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
        self.drain_timeout_s = drain_timeout_s
        self.watchdog_timeout_s = watchdog_timeout_s  # 0 disables
        self.watchdog_poll_s = 5.0
        # Load-based admission: bound concurrent in-flight requests. None
        # (test fakes, direct-registry uses) means unbounded — the old
        # behavior; the CLI always passes a capacity-derived bound.
        self.max_inflight = max_inflight
        self.admission_timeout_s = admission_timeout_s
        self._sem = (
            asyncio.Semaphore(max_inflight) if max_inflight is not None else None
        )
        self._admission_waiters = 0
        # Persistent across generations (bench scrapes survive restarts);
        # None until the first engine with stats is adopted, so sequential
        # engines keep reporting available:false on /debug/engine-stats.
        self.engine_stats: Any | None = None
        # Observability hooks, wired by create_app's metrics layer:
        # stats_observer becomes the accumulator's on_record; the request
        # observer factory yields per-request TTFT/duration observers.
        self.stats_observer: Callable | None = None
        self.request_observer_factory: Callable[[], Any] | None = None
        # Policy knobs, per-instance so tests can shrink them.
        self.backoff_initial_s = BACKOFF_INITIAL_S
        self.backoff_cap_s = BACKOFF_CAP_S
        self.give_up_after = GIVE_UP_AFTER
        self.stable_reset_s = STABLE_RESET_S
        self._inflight = 0
        self._pending: BuiltEngine | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._drain_latched = False
        self._wake: asyncio.Event = asyncio.Event()
        self._wake_reason: str | None = None
        self._restart_requested: asyncio.Event = asyncio.Event()
        self._died_reason: str | None = None
        self._retry_at: float | None = None

    # --- request-path API (event loop only) ----------------------------

    @property
    def inflight(self) -> int:
        return self._inflight

    def ensure_ready(self) -> Any:
        if self.state is EngineState.READY and self.engine is not None:
            return self.engine
        raise NotReadyError(self.name, self.state, self._not_ready_detail())

    async def begin_request(self) -> RequestTicket:
        """Claim an admission slot + the in-flight counter. Over capacity,
        waits up to admission_timeout_s for a slot (a burst that clears in
        200 ms should not eat rejections), then 429s. After a successful
        wait the state is re-checked: a drain that began while we queued
        must still win."""
        if self._sem is not None:
            self._admission_waiters += 1
            try:
                await asyncio.wait_for(
                    self._sem.acquire(), self.admission_timeout_s
                )
            except TimeoutError:
                raise AdmissionBusyError(
                    self.name,
                    f"model '{self.name}' is at capacity "
                    f"({self.max_inflight} in flight); retry later",
                    retry_after_s=self._retry_after_hint(),
                )
            finally:
                self._admission_waiters -= 1
            if self.state is not EngineState.READY:
                self._sem.release()
                raise NotReadyError(
                    self.name, self.state, self._not_ready_detail()
                )
        self._inflight += 1
        return RequestTicket(self)

    def _retry_after_hint(self) -> int:
        """Coarse Retry-After from live stats: how long until a slot
        plausibly frees, given the recent decode rate and the crowd ahead.
        A hint, not a promise; clamped to [1, 60]."""
        estimate = 2.0
        stats = self.engine_stats
        rate = stats.recent_decode_rate() if stats is not None else None
        if rate is not None and self.max_inflight:
            # Assume a typical short-chat remainder per in-flight request;
            # the crowd ahead of the caller is every waiter plus one.
            remaining_tokens = 64.0 * self.max_inflight
            slots_per_wave = max(1.0, float(self.max_inflight))
            estimate = (remaining_tokens / max(rate, 1.0)) \
                * (self._admission_waiters + 1) / slots_per_wave
        return int(max(1.0, min(60.0, estimate)))

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
        if self.state is EngineState.RESTARTING and self._retry_at is not None:
            s["retry_in_s"] = max(0.0, round(self._retry_at - time.monotonic(), 1))
        return s

    # --- admin requests (event loop only) -------------------------------

    def request_reload(self) -> bool:
        """Drain-then-rebuild of the same factory. Accepted only from READY
        (a build is already in flight in every other live state)."""
        if self.state is not EngineState.READY:
            return False
        self._wake_reason = "reload"
        self._wake.set()
        return True

    def request_restart(self) -> bool:
        """From CRASHED: wake the crashed-wait and try again. From READY:
        identical to reload. Anything else: a build is already in flight."""
        if self.state is EngineState.CRASHED:
            self._restart_requested.set()
            return True
        if self.state is EngineState.READY:
            self._wake_reason = "restart"
            self._wake.set()
            return True
        return False

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
        backoff = self.backoff_initial_s
        while True:
            # --- build (first start, rebuild after death, reload) -------
            try:
                await self._build_and_start()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.exception("engine build failed for %s", self.name)
                self.last_error = str(exc)
                self.consecutive_failures += 1
                backoff = await self._back_off_or_crash(backoff)
                continue
            backoff = self.backoff_initial_s

            if self._drain_latched:
                self.state = EngineState.DRAINING
                logger.info(
                    "engine for %s finished building mid-drain; not admitting",
                    self.name,
                )
                return
            self.state = EngineState.READY
            logger.info(
                "engine ready: %s (generation %d)", self.name, self.generation
            )
            ready_since = time.monotonic()

            # --- serve until something wakes us -------------------------
            self._wake.clear()
            self._wake_reason = None
            watchdog = self._arm_watchdog()
            try:
                await self._wake.wait()
            finally:
                if watchdog is not None:
                    watchdog.cancel()
                    try:
                        await watchdog
                    except asyncio.CancelledError:
                        pass
            reason = self._wake_reason

            if reason == "died":
                # In-flight streams already failed cleanly (_fail ran
                # before on_failed). A long-stable engine that dies gets a
                # fresh count; a crash loop escalates.
                if time.monotonic() - ready_since >= self.stable_reset_s:
                    self.consecutive_failures = 0
                self.consecutive_failures += 1
                self.last_error = self._died_reason or self.last_error
                self.state = EngineState.RESTARTING
                await self._retire_engine()
                backoff = await self._back_off_or_crash(backoff)
                continue

            if reason in ("reload", "restart"):
                await self._drain_own_requests()
                self.state = EngineState.RESTARTING
                self._retry_at = None
                await self._retire_engine()
                self.consecutive_failures = 0
                continue

    async def _back_off_or_crash(self, backoff: float) -> float:
        """Shared failure tail: either sleep the backoff (RESTARTING) or,
        past the give-up threshold, park in CRASHED until a manual restart.
        Returns the next backoff value."""
        if self.consecutive_failures >= self.give_up_after:
            self.state = EngineState.CRASHED
            self._retry_at = None
            logger.error(
                "engine for %s crashed %d times consecutively; giving up "
                "until POST /admin/restart", self.name,
                self.consecutive_failures,
            )
            self._restart_requested.clear()
            await self._restart_requested.wait()
            self.consecutive_failures = 0
            return self.backoff_initial_s
        self.state = EngineState.RESTARTING
        self._retry_at = time.monotonic() + backoff
        await asyncio.sleep(backoff)
        self._retry_at = None
        return min(backoff * BACKOFF_FACTOR, self.backoff_cap_s)

    async def _drain_own_requests(self) -> None:
        """Reload-scoped drain: stop admitting (state, not the permanent
        latch), let in-flight finish against the drain deadline, abort
        survivors through the normal event path."""
        self.state = EngineState.DRAINING
        deadline = time.monotonic() + self.drain_timeout_s
        while self._inflight > 0 and time.monotonic() < deadline:
            await asyncio.sleep(DRAIN_POLL_S)
        engine = self.engine
        pending = getattr(engine, "inflight_requests", None)
        if pending is not None:
            survivors = pending()
            for rid in survivors:
                engine.abort(rid)
            if survivors:
                logger.warning(
                    "reload drain deadline hit for %s; aborted %d requests",
                    self.name, len(survivors),
                )
                grace = time.monotonic() + ABORT_GRACE_S
                while self._inflight > 0 and time.monotonic() < grace:
                    await asyncio.sleep(DRAIN_POLL_S)

    async def _retire_engine(self) -> None:
        """Fully shut down the current generation before any rebuild: child
        joined and queues closed, so there is never a moment with two
        engines (and double VRAM) alive."""
        engine, self.engine = self.engine, None
        if not self._eager_runtime:
            self.runtime = None
        if engine is None:
            return
        try:
            await engine.shutdown()
        except Exception:
            logger.exception("shutdown of retired engine for %s failed", self.name)

    def _on_engine_died(self, reason: str) -> None:
        """Wired as EventMultiplexer.on_failed; runs on the event loop at
        the end of _fail(), after in-flight streams got their error events."""
        self._died_reason = reason
        self._wake_reason = "died"
        self._wake.set()

    # --- stats continuity ----------------------------------------------

    def _adopt_stats(self, engine: Any) -> None:
        """Keep one accumulator across generations so /debug/engine-stats
        (the bench scrape) survives restarts. First stats-bearing engine
        donates its accumulator; later generations get the persistent one
        injected, with per-generation metadata refreshed and the child's
        restarted step counter rebased (the `since` cursor stays valid)."""
        fresh = getattr(engine, "engine_stats", None)
        if fresh is None:
            return
        if self.engine_stats is None:
            self.engine_stats = fresh
            self.engine_stats.on_record = self.stats_observer
            return
        if fresh is self.engine_stats:
            return
        self.engine_stats.engine_kind = fresh.engine_kind
        self.engine_stats.load_seconds = fresh.load_seconds
        self.engine_stats.max_batch = fresh.max_batch
        self.engine_stats.max_seq_len = fresh.max_seq_len
        self.engine_stats.note_generation_start()
        self.engine_stats.on_record = self.stats_observer
        engine.engine_stats = self.engine_stats

    # --- watchdog -------------------------------------------------------

    def _arm_watchdog(self) -> asyncio.Task | None:
        """Armed only while READY, and only for engines that can be killed
        (the subprocess client); structurally cannot fire during warm-up.
        The staleness baseline resets on arming."""
        if self.watchdog_timeout_s <= 0:
            return None
        if not hasattr(self.engine, "kill"):
            if self.generation == 1:
                logger.info(
                    "watchdog unavailable for %s (engine not killable)",
                    self.name,
                )
            return None
        if self.engine_stats is None:
            return None
        self.engine_stats.last_update_mono = time.monotonic()
        return asyncio.get_running_loop().create_task(
            self._watchdog(), name=f"engine-watchdog-{self.name}"
        )

    async def _watchdog(self) -> None:
        while True:
            await asyncio.sleep(self.watchdog_poll_s)
            engine = self.engine
            if engine is None:
                return
            pending = engine.inflight_requests()
            if not pending:
                continue
            stale_s = time.monotonic() - self.engine_stats.last_update_mono
            if stale_s > self.watchdog_timeout_s:
                logger.error(
                    "watchdog: engine for %s has %d pending requests and no "
                    "step progress for %.0f s; killing the child",
                    self.name, len(pending), stale_s,
                )
                engine.kill()
                return

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
        # Death notification: the multiplexer's _fail runs on the loop and
        # calls this after in-flight streams got their error events. The
        # sequential engine has no failure path (no on_failed attribute).
        if hasattr(built.engine, "on_failed"):
            built.engine.on_failed = self._on_engine_died
        self._adopt_stats(built.engine)
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

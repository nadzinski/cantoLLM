"""CantoServer: uvicorn with drain-on-signal semantics (Phase 3.5).

Stock uvicorn's first signal stops accepting and waits (indefinitely,
unless timeout_graceful_shutdown) for open connections; open SSE streams
therefore used to hold shutdown open forever, and a second Ctrl-C was the
only escape. This subclass reroutes both SIGTERM and SIGINT through a
`DrainController`:

  first signal  -> handles flip to DRAINING (admission stops, /ready 503),
                   in-flight requests run to a deadline (--drain-timeout,
                   default 30 s), survivors are aborted through the normal
                   event path (clients see finish_reason="abort"), then
                   uvicorn's own graceful exit runs.
  second signal -> force: drain task cancelled, streams closed, uvicorn
                   force-exits. Works for SIGTERM too (stock uvicorn only
                   force-exits on a repeated SIGINT).

`handle_exit` is the single funnel uvicorn routes both signals through
(verified against uvicorn 0.44); overriding it leaves the rest of the
machinery — including uvloop via Config and the captured-signal re-raise
that makes a SIGTERM'd process exit with the conventional 143 — untouched.
Signal handlers run on the main thread between bytecodes, so the override
only sets flags and hops onto the loop with call_soon_threadsafe.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Callable

import uvicorn

from cantollm.registry import EngineRegistry

logger = logging.getLogger(__name__)

_POLL_S = 0.1
_ABORT_GRACE_S = 2.0


class DrainController:
    """Owns the drain choreography; loop-bound at serve() time."""

    def __init__(self, registry: EngineRegistry, drain_timeout_s: float = 30.0):
        self.registry = registry
        self.drain_timeout_s = drain_timeout_s
        self.draining = False
        self._loop: asyncio.AbstractEventLoop | None = None
        self._task: asyncio.Task | None = None

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    def _handles(self):
        return [
            entry.handle
            for _, entry in self.registry.items()
            if getattr(entry, "handle", None) is not None
        ]

    # --- signal-context entry points (flags + threadsafe hops only) ----

    def begin(self, on_done: Callable[[], None]) -> bool:
        """Start the drain; returns False if no loop is bound yet (caller
        falls back to stock shutdown)."""
        self.draining = True
        if self._loop is None:
            return False
        self._loop.call_soon_threadsafe(self._start_task, on_done)
        return True

    def force(self) -> None:
        if self._loop is not None:
            self._loop.call_soon_threadsafe(self._force_on_loop)

    # --- loop side -----------------------------------------------------

    def _start_task(self, on_done: Callable[[], None]) -> None:
        if self._task is None:
            self._task = asyncio.get_running_loop().create_task(
                self._drain(on_done), name="drain"
            )

    def _force_on_loop(self) -> None:
        if self._task is not None:
            self._task.cancel()
        # Close every stream so uvicorn's wait-for-connections clears.
        for _, entry in self.registry.items():
            engine = getattr(entry, "engine", None)
            close = getattr(engine, "_close_all_streams", None)
            if close is not None:
                close()

    async def _drain(self, on_done: Callable[[], None]) -> None:
        handles = self._handles()
        for h in handles:
            h.begin_drain()
        logger.info(
            "draining: %d in-flight, deadline %.0f s",
            sum(h.inflight for h in handles), self.drain_timeout_s,
        )
        try:
            deadline = time.monotonic() + self.drain_timeout_s
            while (any(h.inflight > 0 for h in handles)
                   and time.monotonic() < deadline):
                await asyncio.sleep(_POLL_S)
            survivors = 0
            for h in handles:
                engine = h.engine
                if engine is None:
                    continue
                pending = getattr(engine, "inflight_requests", None)
                if pending is None:
                    continue
                for rid in pending():
                    engine.abort(rid)
                    survivors += 1
            if survivors:
                logger.warning(
                    "drain deadline hit; aborted %d requests", survivors
                )
                grace = time.monotonic() + _ABORT_GRACE_S
                while (any(h.inflight > 0 for h in handles)
                       and time.monotonic() < grace):
                    await asyncio.sleep(0.05)
        finally:
            # Runs on cancellation too (force path): uvicorn must still be
            # told to exit.
            on_done()


class CantoServer(uvicorn.Server):
    def __init__(self, config: uvicorn.Config, drainer: DrainController):
        super().__init__(config)
        self._drainer = drainer

    async def serve(self, sockets=None) -> None:
        self._drainer.bind_loop(asyncio.get_running_loop())
        await super().serve(sockets)

    def handle_exit(self, sig, frame) -> None:
        # Signal-handler context: flags and threadsafe hops only.
        self._captured_signals.append(sig)
        if self._drainer.draining:
            self._drainer.force()
            self.force_exit = True
            self.should_exit = True
            return
        if not self._drainer.begin(on_done=self._exit_when_drained):
            # No loop bound yet (signal before serve started): stock path.
            self.should_exit = True

    def _exit_when_drained(self) -> None:
        self.should_exit = True

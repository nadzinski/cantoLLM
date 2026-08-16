"""Warm-up progress reporting, shared by every engine build path.

A leaf module (imports nothing from cantollm) so the model runtime, the
warm-up sweep, graph capture, the engine child process, and the lifecycle
handle can all speak the same vocabulary without import cycles.

`report(stage, done, total)` is called from inside the build (weight load,
compile enable, shape sweep, graph capture) and routes to whatever sink the
surrounding context bound with `bind_sink`:

- the engine child binds `events.put`, so Progress messages ride the IPC
  event queue ahead of Ready;
- the in-process supervisor binds a thread-safe hop onto the event loop, and
  `asyncio.to_thread` copies the context, so a factory running on a worker
  thread inherits the sink for free.

With no sink bound, report() is a no-op — library code never pays for it.

Throttling lives here so emitters stay dumb: a stage change and a stage
completion always emit; within a stage, at most one message per 0.5 s.
"""

from __future__ import annotations

import time
from contextvars import ContextVar, Token
from dataclasses import dataclass
from typing import Callable

_MIN_INTERVAL_S = 0.5

STAGES = ("load", "compile", "sweep", "capture")


@dataclass(frozen=True)
class Progress:
    """One warm-up progress tick. Crosses the IPC boundary (picklable)."""

    stage: str            # one of STAGES
    done: int
    total: int | None     # known up front for every current stage
    detail: str = ""
    elapsed_s: float = 0.0  # since this stage started


@dataclass
class _SinkState:
    cb: Callable[[Progress], None]
    stage: str | None = None
    stage_t0: float = 0.0
    last_emit: float = float("-inf")


_state: ContextVar[_SinkState | None] = ContextVar("progress_sink", default=None)


def bind_sink(cb: Callable[[Progress], None]) -> Token:
    """Route report() in this context (and contexts copied from it) to `cb`.
    The callback may be invoked from worker threads; it must be thread-safe
    (a queue put or a call_soon_threadsafe hop both qualify)."""
    return _state.set(_SinkState(cb))


def unbind_sink(token: Token) -> None:
    _state.reset(token)


def current_sink() -> Callable[[Progress], None] | None:
    """The raw callback bound in this context, if any. Used by the engine
    process client to forward child-side Progress to the ambient sink."""
    state = _state.get()
    return state.cb if state is not None else None


def report(stage: str, done: int, total: int | None, detail: str = "") -> None:
    state = _state.get()
    if state is None:
        return
    now = time.perf_counter()
    force = False
    if stage != state.stage:
        state.stage = stage
        state.stage_t0 = now
        force = True
    if total is not None and done == total:
        force = True
    if not force and now - state.last_emit < _MIN_INTERVAL_S:
        return
    state.last_emit = now
    state.cb(Progress(
        stage=stage, done=done, total=total, detail=detail,
        elapsed_s=now - state.stage_t0,
    ))

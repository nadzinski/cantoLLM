"""Drain + signals (3.5 chunk 2c): DrainController choreography, CantoServer
signal funnel, and one real SIGTERM through the tiny stub server.

The controller units use scripted handles (no engines); the signal tests
call handle_exit directly (no real signals; it is a plain method); the
integration test spawns tests.bench_server_stub in tiny mode and SIGTERMs
it mid-stream, asserting the in-flight stream ran to completion and the
process exited with conventional signal etiquette (-SIGTERM after a clean
drain, courtesy of uvicorn's captured-signal re-raise).
"""

from __future__ import annotations

import asyncio
import signal
import subprocess
import sys
import time

import httpx
import pytest
import uvicorn

from cantollm.server import CantoServer, DrainController
from tests.fakes import parse_sse

PORT = 8395  # fixed per-file port, matching the other server-spawning tests


# ── scripted doubles ─────────────────────────────────────────────────


class _Engine:
    def __init__(self, pending=()):
        self.pending = list(pending)
        self.aborts: list[str] = []
        self.closed = False

    def inflight_requests(self):
        return list(self.pending)

    def abort(self, rid):
        self.aborts.append(rid)

    def _close_all_streams(self):
        self.closed = True


class _Handle:
    def __init__(self, inflight=0, engine=None):
        self.inflight = inflight
        self.engine = engine
        self.drain_begun = False

    def begin_drain(self):
        self.drain_begun = True


class _Entry:
    def __init__(self, handle):
        self.handle = handle

    @property
    def engine(self):
        return self.handle.engine


class _Registry:
    def __init__(self, handles):
        self._entries = {f"m{i}": _Entry(h) for i, h in enumerate(handles)}

    def items(self):
        return self._entries.items()


# ── DrainController units ────────────────────────────────────────────


def test_drain_completes_when_inflight_clears():
    handle = _Handle(inflight=1, engine=_Engine())
    drainer = DrainController(_Registry([handle]), drain_timeout_s=5.0)
    done = asyncio.Event()

    async def main():
        drainer.bind_loop(asyncio.get_running_loop())
        assert drainer.begin(on_done=done.set) is True
        await asyncio.sleep(0.15)
        assert handle.drain_begun
        assert not done.is_set()          # still one in-flight
        handle.inflight = 0               # stream finished naturally
        await asyncio.wait_for(done.wait(), timeout=2)
        assert handle.engine.aborts == [] # nobody needed aborting

    asyncio.run(main())
    assert drainer.draining


def test_drain_deadline_aborts_survivors():
    engine = _Engine(pending=["r1", "r2"])
    handle = _Handle(inflight=2, engine=engine)
    drainer = DrainController(_Registry([handle]), drain_timeout_s=0.2)
    done = asyncio.Event()

    async def main():
        drainer.bind_loop(asyncio.get_running_loop())
        drainer.begin(on_done=done.set)
        await asyncio.wait_for(done.wait(), timeout=5)

    asyncio.run(main())
    assert engine.aborts == ["r1", "r2"]


def test_begin_without_loop_reports_failure():
    drainer = DrainController(_Registry([]), drain_timeout_s=1.0)
    assert drainer.begin(on_done=lambda: None) is False
    assert drainer.draining  # flag still set: second signal forces


def test_force_cancels_and_closes_streams():
    engine = _Engine(pending=["r1"])
    handle = _Handle(inflight=1, engine=engine)
    drainer = DrainController(_Registry([handle]), drain_timeout_s=30.0)
    done = asyncio.Event()

    async def main():
        drainer.bind_loop(asyncio.get_running_loop())
        drainer.begin(on_done=done.set)
        await asyncio.sleep(0.05)         # drain task is now waiting
        drainer.force()
        # finally in the cancelled task must still fire on_done.
        await asyncio.wait_for(done.wait(), timeout=2)
        assert engine.closed

    asyncio.run(main())


# ── CantoServer.handle_exit ──────────────────────────────────────────


async def _noop_app(scope, receive, send):  # pragma: no cover - never called
    pass


def _server(drainer) -> CantoServer:
    return CantoServer(uvicorn.Config(_noop_app), drainer)


def test_first_signal_drains_second_forces():
    drainer = DrainController(_Registry([]), drain_timeout_s=1.0)
    server = _server(drainer)

    async def main():
        drainer.bind_loop(asyncio.get_running_loop())
        server.handle_exit(signal.SIGTERM, None)
        assert drainer.draining
        assert not server.force_exit
        server.handle_exit(signal.SIGINT, None)   # either signal forces
        assert server.force_exit
        assert server.should_exit
        await asyncio.sleep(0.05)  # let the (cancelled) drain settle

    asyncio.run(main())


def test_signal_before_loop_falls_back_to_stock_exit():
    drainer = DrainController(_Registry([]), drain_timeout_s=1.0)
    server = _server(drainer)
    server.handle_exit(signal.SIGTERM, None)
    assert server.should_exit          # stock path: exit without drain
    assert not server.force_exit


# ── real SIGTERM through the tiny stub ───────────────────────────────


@pytest.mark.slow
def test_sigterm_mid_stream_drains_to_completion():
    proc = subprocess.Popen(
        [sys.executable, "-m", "tests.bench_server_stub",
         "--port", str(PORT), "--mode", "tiny", "--drain-timeout", "20"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    base = f"http://127.0.0.1:{PORT}"
    try:
        deadline = time.monotonic() + 60
        with httpx.Client(timeout=5.0) as client:
            while True:
                try:
                    if client.get(f"{base}/ready").status_code == 200:
                        break
                except httpx.HTTPError:
                    pass
                assert time.monotonic() < deadline, "stub never became ready"
                time.sleep(0.1)

            body = {
                "model": "tiny", "max_tokens": 56, "stream": True,
                "temperature": 0.0,
                "messages": [{"role": "user", "content": "hi"}],
            }
            with client.stream(
                "POST", f"{base}/v1/messages", json=body, timeout=30.0
            ) as resp:
                assert resp.status_code == 200
                chunks: list[str] = []
                sent = False
                for chunk in resp.iter_text():
                    chunks.append(chunk)
                    if not sent:
                        proc.send_signal(signal.SIGTERM)
                        sent = True
                text = "".join(chunks)

        events = parse_sse(text)
        kinds = [e.event for e in events if e.event != "ping"]
        # The drain let the stream finish: proper close, no truncation.
        assert kinds[-1] == "message_stop"
        assert "error" not in kinds
        # Clean exit with signal etiquette: uvicorn re-raises the captured
        # SIGTERM after graceful shutdown.
        assert proc.wait(timeout=30) == -signal.SIGTERM
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()

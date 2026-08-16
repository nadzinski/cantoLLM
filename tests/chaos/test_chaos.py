"""Phase 3.5 chaos suite — the load/failure exit criterion, runnable.

Excluded from the default run (`addopts = -m "not chaos"`); invoke with

    python -m pytest -m chaos tests/chaos/ -v

Every scenario spawns the REAL `canto serve` (subprocess engine included)
on the tiny chaos model (tests/chaos/tiny_serve.py, via the
CANTOLLM_TEST_SPEC hook), drives load through the bench loadgen used as a
library, injects one fault, and asserts client-visible behavior:

  1. 50 concurrent mixed-size clients run clean: zero truncation.
  2. Overload yields clean 429s with Retry-After; nothing else breaks.
  3. SIGTERM mid-load drains: in-flight streams complete, exit is clean.
  4. kill -9 the engine child: clean failures, /ready 503, supervisor
     recovery, traffic resumes.
  5. A wedged engine (env fault hook) trips the watchdog and recovers.

On the 5090 the same scenarios can be pointed at the real 0.6B stack by
swapping the spawn args; this CPU-tiny form is the regression net.
"""

from __future__ import annotations

import asyncio
import itertools
import signal
import subprocess
import sys
import time

import httpx
import psutil
import pytest

from cantollm.bench.loadgen import run_closed_loop
from cantollm.bench.sse_clients import SendOptions, build_sender
from cantollm.bench.workloads import Prompt

pytestmark = pytest.mark.chaos

READY_TIMEOUT_S = 90.0
SPEC_HOOK = "tests.chaos.tiny_serve:chaos_tiny_spec"


# ── server harness ───────────────────────────────────────────────────


class ChaosServer:
    def __init__(self, port: int, *extra_args: str, env_extra: dict | None = None):
        import os

        self.port = port
        self.base = f"http://127.0.0.1:{port}"
        cmd = [
            sys.executable, "-m", "cantollm.main", "serve",
            "--model", "tiny", "--engine", "batched", "--device", "cpu",
            "--host", "127.0.0.1", "--port", str(port),
            "--max-batch", "8", "--batch-max-seq-len", "96",
            "--max-tokens-per-step", "32",
            *extra_args,
        ]
        env = {**os.environ, "CANTOLLM_TEST_SPEC": SPEC_HOOK}
        env.update(env_extra or {})
        self.proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env
        )

    def wait_ready(self, timeout_s: float = READY_TIMEOUT_S) -> None:
        deadline = time.monotonic() + timeout_s
        with httpx.Client(timeout=5.0) as client:
            while True:
                if self.proc.poll() is not None:
                    raise AssertionError(
                        f"server died during startup (exit {self.proc.poll()}):\n"
                        + self.log_tail()
                    )
                try:
                    if client.get(f"{self.base}/ready").status_code == 200:
                        return
                except httpx.HTTPError:
                    pass
                if time.monotonic() > deadline:
                    raise AssertionError(
                        f"server not ready in {timeout_s}s:\n" + self.log_tail()
                    )
                time.sleep(0.2)

    def ready_body(self) -> dict:
        with httpx.Client(timeout=5.0) as client:
            return client.get(f"{self.base}/ready").json()

    def engine_child(self) -> psutil.Process:
        # The serve process has two children under spawn: the engine child
        # (cmdline mentions spawn_main) and multiprocessing's resource
        # tracker — killing the tracker would be a very quiet no-op.
        parent = psutil.Process(self.proc.pid)
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            for kid in parent.children(recursive=True):
                try:
                    if "spawn_main" in " ".join(kid.cmdline()):
                        return kid
                except psutil.NoSuchProcess:
                    continue
            time.sleep(0.1)
        raise AssertionError("no engine child process found")

    def log_tail(self, n: int = 40) -> str:
        # Only safe once the process exited (pipe fully readable).
        if self.proc.poll() is None:
            return "<server still running>"
        out = self.proc.stdout.read() or b""
        return b"\n".join(out.splitlines()[-n:]).decode(errors="replace")

    def stop(self) -> None:
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()


# ── load helpers (bench loadgen as a library) ────────────────────────


def _prompts():
    sizes = (8, 40, 120, 400)  # chars -> 2..48 tokens via the chaos tokenizer
    prompts = [
        Prompt(id=f"p{n}", messages=({"role": "user", "content": "x" * n},),
               system=None, input_tokens=None)
        for n in sizes
    ]
    return itertools.cycle(prompts)


async def _run_load(base: str, *, concurrency: int, total: int,
                    max_tokens: int):
    async with httpx.AsyncClient(base_url=base, timeout=30.0) as client:
        send = build_sender(client, SendOptions(
            model="qwen3-tiny", dialect="openai",
            max_tokens=max_tokens, ignore_eos=True,
        ))
        result = await run_closed_loop(
            send, _prompts(), concurrency=concurrency, total_requests=total,
            cell_id="chaos", repeat=0,
        )
    return result.records


# ── 1. fifty concurrent mixed-size clients, zero truncation ──────────


def test_fifty_clients_run_clean():
    server = ChaosServer(8420, "--max-inflight", "64")
    try:
        server.wait_ready()
        records = asyncio.run(_run_load(
            server.base, concurrency=50, total=100, max_tokens=16,
        ))
        assert len(records) == 100
        bad = [r for r in records if not r.ok]
        assert not bad, f"{len(bad)} failed, first: {bad[0].error}"
        # Fixed-length mode: anything but "length" is a truncation.
        assert {r.finish_reason for r in records} == {"length"}
        assert all(r.output_tokens == 16 for r in records)
    finally:
        server.stop()


# ── 2. overload -> clean 429s with Retry-After ───────────────────────


def test_overload_yields_clean_429s():
    server = ChaosServer(
        8421, "--max-inflight", "2", "--admission-timeout", "0.05",
    )
    try:
        server.wait_ready()

        async def burst():
            async with httpx.AsyncClient(
                base_url=server.base, timeout=30.0
            ) as client:
                async def one():
                    return await client.post("/v1/messages", json={
                        "model": "qwen3-tiny", "max_tokens": 48,
                        "ignore_eos": True, "stream": False,
                        "messages": [{"role": "user", "content": "x" * 100}],
                    })
                return await asyncio.gather(*(one() for _ in range(12)))

        responses = asyncio.run(burst())
        statuses = sorted({r.status_code for r in responses})
        assert set(statuses) <= {200, 429}, statuses
        rejected = [r for r in responses if r.status_code == 429]
        succeeded = [r for r in responses if r.status_code == 200]
        assert rejected, "overload never produced a 429"
        assert succeeded, "everything was rejected"
        for r in rejected:
            assert int(r.headers["Retry-After"]) >= 1
            assert r.json()["error"]["type"] == "rate_limit_error"
        for r in succeeded:
            assert r.json()["usage"]["output_tokens"] == 48
    finally:
        server.stop()


# ── 3. SIGTERM mid-load drains without truncation ────────────────────


def test_sigterm_drains_under_load():
    server = ChaosServer(8422, "--drain-timeout", "30")
    try:
        server.wait_ready()

        async def main():
            load = asyncio.create_task(_run_load(
                server.base, concurrency=8, total=24, max_tokens=32,
            ))
            await asyncio.sleep(0.4)  # load is genuinely in flight
            server.proc.send_signal(signal.SIGTERM)
            return await load

        records = asyncio.run(main())
        completed = [r for r in records if r.ok]
        refused = [r for r in records if r.error == "http 503"]
        # In flight at the signal -> drained to completion, full length.
        assert completed and all(
            r.finish_reason == "length" and r.output_tokens == 32
            for r in completed
        )
        # Submitted after the drain flip -> clean 503, never truncation.
        other = [r for r in records if not r.ok and r.error != "http 503"]
        # Connection errors are possible only after uvicorn closes; those
        # requests never started a stream, so they are clean refusals too.
        conn_errors = [r for r in other if "Connect" in (r.error or "")]
        assert len(other) == len(conn_errors), \
            f"unclean failures: {[r.error for r in other if r not in conn_errors][:3]}"
        assert len(completed) + len(refused) + len(conn_errors) == 24
        assert server.proc.wait(timeout=30) == -signal.SIGTERM
    finally:
        server.stop()


# ── 4. kill -9 the engine child -> 503 -> supervisor recovery ────────


def test_kill9_engine_recovers():
    # The tiny model outruns any client-side timing (a full 84-token
    # generation finishes before httpx parses the first SSE line), so the
    # step-delay fault hook paces the engine: 20 ms/step makes the stream
    # genuinely in flight when the kill lands after the first data line.
    server = ChaosServer(
        8423, env_extra={"CANTOLLM_TEST_STEP_DELAY_S": "0.02"},
    )
    try:
        server.wait_ready()
        child = server.engine_child()

        async def main():
            async with httpx.AsyncClient(
                base_url=server.base, timeout=30.0
            ) as client:
                # A stream in flight when the child dies must fail cleanly
                # (SSE error event on an HTTP 200 stream, not a hang or a
                # connection drop).
                async with client.stream("POST", "/v1/messages", json={
                    "model": "qwen3-tiny", "max_tokens": 84,
                    "ignore_eos": True, "stream": True,
                    "messages": [{"role": "user", "content": "x" * 40}],
                }) as resp:
                    assert resp.status_code == 200
                    tail = []
                    killed = False
                    async for line in resp.aiter_lines():
                        if not killed and line.startswith("data:"):
                            child.kill()  # SIGKILL: no farewell; the
                            killed = True  # bridge must notice on its own
                        elif killed:
                            tail.append(line)
                    assert any("engine failed" in line for line in tail), \
                        f"no clean error event; tail: {tail[-6:]}"

                # /ready flips away from serving, then the supervisor
                # rebuilds and traffic resumes.
                deadline = time.monotonic() + 60
                while time.monotonic() < deadline:
                    body = (await client.get("/ready")).json()
                    if (body["models"]["qwen3-tiny"]["generation"] >= 2
                            and body["status"] == "ready"):
                        break
                    await asyncio.sleep(0.2)
                else:
                    raise AssertionError(f"no recovery: {body}")
                r = await client.post("/v1/messages", json={
                    "model": "qwen3-tiny", "max_tokens": 8,
                    "ignore_eos": True, "stream": False,
                    "messages": [{"role": "user", "content": "hi"}],
                })
                assert r.status_code == 200
                assert r.json()["usage"]["output_tokens"] == 8

        asyncio.run(main())
    finally:
        server.stop()


# ── 5. wedged engine -> watchdog kill -> recovery ────────────────────


def test_watchdog_catches_wedged_engine():
    server = ChaosServer(
        8424, "--watchdog-timeout", "2",
        env_extra={"CANTOLLM_TEST_WEDGE_AFTER_STEPS": "2"},
    )
    try:
        server.wait_ready()

        async def main():
            async with httpx.AsyncClient(
                base_url=server.base, timeout=60.0
            ) as client:
                # This request wedges the child mid-generation: alive, work
                # pending, zero step progress. Only the watchdog can act.
                r = await client.post("/v1/messages", json={
                    "model": "qwen3-tiny", "max_tokens": 32,
                    "ignore_eos": True, "stream": True,
                    "messages": [{"role": "user", "content": "x" * 40}],
                })
                assert r.status_code == 200
                assert "engine failed" in r.text  # watchdog killed the child

                deadline = time.monotonic() + 60
                while time.monotonic() < deadline:
                    body = (await client.get("/ready")).json()
                    if (body["models"]["qwen3-tiny"]["generation"] >= 2
                            and body["status"] == "ready"):
                        return
                    await asyncio.sleep(0.2)
                raise AssertionError(f"no watchdog recovery: {body}")

        asyncio.run(main())
    finally:
        server.stop()

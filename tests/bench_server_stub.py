"""Runnable server-under-test stub for bench-executor tests.

The executor's `server_command` seam points here instead of `canto serve`:

    python -m tests.bench_server_stub --port 8399 --mode fake
    python -m tests.bench_server_stub --port 8399 --mode crash --crash-after 3
    python -m tests.bench_server_stub --port 8399 --mode tiny

`fake` serves the real FastAPI app over a StubEngine (honors max_tokens,
per-token sleep — realistic timings, no torch). `crash` hard-exits the
process mid-stream after N requests (server-death-mid-cell test). `tiny`
composes the real in-process ContinuousBatchingEngine on the tiny-Qwen3
runtime — the E2E path with real engine stats.
"""

from __future__ import annotations

import argparse
import asyncio
import os
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

from cantollm.engine.types import InferenceRequest, TokenEvent
from tests.fakes import FakeRegistry, FakeRuntime, FakeTokenizer


@dataclass
class StubEngine:
    """Honors max_tokens (fixed-length friendly), sleeps per token, and can
    hard-exit the whole process mid-stream after `crash_after` submits."""

    token_sleep: float = 0.005
    crash_after: int | None = None
    submits: int = 0
    engine_stats = None  # explicit: /debug/engine-stats → available:false

    async def start(self) -> None:  # pragma: no cover - lifecycle noop
        pass

    async def shutdown(self) -> None:  # pragma: no cover
        pass

    def abort(self, request_id: str) -> None:  # pragma: no cover
        pass

    async def submit(self, req: InferenceRequest) -> AsyncIterator[TokenEvent]:
        self.submits += 1
        crash_now = self.crash_after is not None and self.submits > self.crash_after
        for i in range(req.max_tokens):
            if self.token_sleep:
                await asyncio.sleep(self.token_sleep)
            if crash_now and i == 1:
                os._exit(17)   # simulated segfault: no farewell, no cleanup
            yield TokenEvent(token_id=(i % 200) + 1, request_id=req.request_id)
        yield TokenEvent(finish_reason="max_tokens", request_id=req.request_id)


@dataclass
class StatsStubEngine(StubEngine):
    """StubEngine plus a canned engine_stats accumulator, so executor tests
    exercise the scrape path without torch."""

    engine_stats: object = field(default=None)

    def __post_init__(self):
        from cantollm.engine.batching.stats import (
            EngineStatsAccumulator,
            StepStats,
            StepUpdate,
        )

        acc = EngineStatsAccumulator(
            engine_kind="stub", max_batch=4, max_seq_len=64,
        )
        acc.load_seconds = 0.123
        for seq in range(3):
            acc.record(StepUpdate(events=[], stats=StepStats(
                seq=seq, t_wall=seq * 1.0, t_perf=seq * 1.0, dur_s=0.01,
                rows=1, occupied_slots=0, queue_depth=0, kv_tokens=0,
                prefill_tokens=4, decode_tokens=1,
            )))
        self.engine_stats = acc


def build_app(mode: str, crash_after: int | None):
    from cantollm.api import create_app

    if mode == "tiny":
        import torch

        from cantollm.engine import ContinuousBatchingEngine
        from cantollm.engine.batching import BatchingConfig
        from cantollm.registry import EngineRegistry
        from cantollm.runtime import build_runtime
        from tests.tiny_model import tiny_qwen3_spec

        runtime = build_runtime(
            tiny_qwen3_spec(), torch.device("cpu"), attention="padded"
        )
        config = BatchingConfig(max_batch=3, max_seq_len=64, max_tokens_per_step=16)
        engine = ContinuousBatchingEngine.from_runtime(runtime, config)
        registry = EngineRegistry()
        registry.register("tiny", engine, runtime, max_request_tokens=64)
        return create_app(registry)

    if mode == "crash":
        engine = StubEngine(crash_after=crash_after)
    elif mode == "stats":
        engine = StatsStubEngine()
    else:
        engine = StubEngine()
    registry = FakeRegistry(
        entries={"stub-model": (engine, FakeRuntime(FakeTokenizer()))}
    )
    return create_app(registry)


def main() -> None:
    import uvicorn

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--mode", choices=("fake", "crash", "stats", "tiny"),
                        default="fake")
    parser.add_argument("--crash-after", type=int, default=None)
    args = parser.parse_args()

    uvicorn.run(
        build_app(args.mode, args.crash_after),
        host="127.0.0.1", port=args.port, log_level="warning",
    )


if __name__ == "__main__":
    main()

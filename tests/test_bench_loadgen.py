"""Load-generator tests against a scripted async sender — no HTTP."""

import asyncio
import itertools

import pytest

from cantollm.bench.loadgen import run_closed_loop, run_open_loop
from cantollm.bench.records import RequestRecord
from cantollm.bench.workloads import Prompt


def prompts():
    counter = itertools.count()
    while True:
        i = next(counter)
        yield Prompt(id=f"p{i % 5}", messages=({"role": "user", "content": "q"},),
                     system=None, input_tokens=10)


class ScriptedSender:
    """Sender double: fixed latency, tracks live concurrency."""

    def __init__(self, latency: float = 0.01):
        self.latency = latency
        self.live = 0
        self.max_live = 0
        self.calls = 0

    async def __call__(self, prompt, *, cell_id, repeat, request_index,
                       t_scheduled=None, excluded=False):
        import time
        self.calls += 1
        self.live += 1
        self.max_live = max(self.max_live, self.live)
        t_send = time.perf_counter()   # real clock: dispatch lag must be honest
        try:
            await asyncio.sleep(self.latency)
        finally:
            self.live -= 1
        record = RequestRecord(
            cell_id=cell_id, repeat=repeat, request_index=request_index,
            prompt_id=prompt.id, dialect="openai",
            t_scheduled=t_scheduled, t_send=t_send, t_done=time.perf_counter(),
            output_tokens=4, finish_reason="length",
        )
        return record.finalize(), None


def test_closed_loop_exact_count_and_concurrency_ceiling():
    sender = ScriptedSender()

    async def main():
        return await run_closed_loop(
            sender, prompts(), concurrency=4, total_requests=19,
            cell_id="c", repeat=0,
        )

    result = asyncio.run(main())
    assert len(result.records) == 19
    assert sender.max_live <= 4
    assert sorted(r.request_index for r in result.records) == list(range(19))
    assert not result.aborted


def test_closed_loop_abort_stops_between_requests():
    sender = ScriptedSender(latency=0.02)

    async def main():
        abort = asyncio.Event()
        task = asyncio.create_task(run_closed_loop(
            sender, prompts(), concurrency=2, total_requests=1000,
            cell_id="c", repeat=0, abort=abort,
        ))
        await asyncio.sleep(0.1)
        abort.set()
        return await task

    result = asyncio.run(main())
    assert result.aborted
    assert 0 < len(result.records) < 1000


def test_texts_keyed_by_cell_repeat_index_prompt():
    # The key must carry cell_id: the whole run appends to one text file,
    # and (repeat, index, prompt) repeats across cells.
    class TextSender(ScriptedSender):
        async def __call__(self, prompt, **kw):
            record, _ = await super().__call__(prompt, **kw)
            return record, f"reply to {prompt.id}"

    async def main():
        return await run_closed_loop(
            TextSender(latency=0.0), prompts(), concurrency=2, total_requests=4,
            cell_id="s0-p0-c2", repeat=1,
        )

    result = asyncio.run(main())
    assert len(result.texts) == 4
    for key, text in result.texts.items():
        cell, rep, _index, pid = key.split(":")
        assert cell == "s0-p0-c2" and rep == "1"
        assert text == f"reply to {pid}"


def test_open_loop_seeded_schedule_is_deterministic():
    async def run(seed):
        sends = []

        async def sender(prompt, *, t_scheduled=None, **kw):
            sends.append(t_scheduled)
            record = RequestRecord(
                cell_id="c", repeat=0, request_index=kw["request_index"],
                prompt_id=prompt.id, dialect="openai",
                t_scheduled=t_scheduled, t_send=t_scheduled or 0.0,
            )
            return record.finalize(), None

        await run_open_loop(
            sender, prompts(), rate_rps=200.0, arrivals="poisson",
            total_requests=10, max_inflight=64, seed=seed,
            cell_id="c", repeat=0,
        )
        return [s - sends[0] for s in sends]

    a = asyncio.run(run(seed=3))
    b = asyncio.run(run(seed=3))
    c = asyncio.run(run(seed=4))
    assert a == pytest.approx(b)
    assert a != pytest.approx(c)
    assert all(dt >= 0 for dt in a) and a == sorted(a)


def test_open_loop_fires_despite_slow_completions_until_cap():
    # 20 rps against a 0.5 s server: arrivals must not wait for completions
    # (that would be closed-loop); the tiny cap must trip instead.
    sender = ScriptedSender(latency=0.5)

    async def main():
        return await run_open_loop(
            sender, prompts(), rate_rps=200.0, arrivals="fixed",
            total_requests=8, max_inflight=4, seed=0,
            cell_id="c", repeat=0, drain_timeout_s=5.0,
        )

    result = asyncio.run(main())
    assert result.hit_inflight_cap
    assert sender.max_live == 4
    assert len(result.records) == 8
    # Requests past the cap show real dispatch lag.
    lags = [r.dispatch_lag_s for r in result.records if r.dispatch_lag_s is not None]
    assert max(lags) > 0.1


def test_open_loop_abort_cancels_inflight():
    sender = ScriptedSender(latency=10.0)   # would hang without cancellation

    async def main():
        abort = asyncio.Event()
        task = asyncio.create_task(run_open_loop(
            sender, prompts(), rate_rps=100.0, arrivals="fixed",
            total_requests=50, max_inflight=8, seed=0,
            cell_id="c", repeat=0, abort=abort,
        ))
        await asyncio.sleep(0.1)
        abort.set()
        return await asyncio.wait_for(task, timeout=5.0)

    result = asyncio.run(main())
    assert result.aborted
    assert len(result.records) < 50

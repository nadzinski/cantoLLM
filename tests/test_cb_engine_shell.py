"""ContinuousBatchingEngine shell tests — torch-free, against a scripted
scheduler double.

The shell's job: multiplex one scheduler thread into N async iterators,
carry commands in on one queue, never spin while idle, and fail loudly
batch-wide. Every test here pins one of those behaviors; none of them know
anything about models, pools, or the real scheduler.
"""

import asyncio
import threading

from cantollm.engine.batching import ContinuousBatchingEngine
from cantollm.engine.types import InferenceRequest, SamplingParams, TokenEvent


def make_request(rid: str) -> InferenceRequest:
    return InferenceRequest(
        request_id=rid,
        prompt_token_ids=[1, 2, 3],
        sampling_params=SamplingParams(greedy=True),
        max_tokens=8,
        stop_token_ids={999},
    )


def tok(rid: str, token_id: int) -> TokenEvent:
    return TokenEvent(token_id=token_id, request_id=rid)


def fin(rid: str, reason: str = "end_turn") -> TokenEvent:
    return TokenEvent(finish_reason=reason, request_id=rid)


class ScriptedScheduler:
    """SchedulerLike double: each active request emits its next scripted
    event per step (mirroring one-decode-per-step), abort acks become
    pending events, and is_idle honors the 'False whenever step() would
    produce events' contract."""

    def __init__(self, scripts: dict[str, list[TokenEvent]]):
        self.scripts = {rid: list(evts) for rid, evts in scripts.items()}
        self.active: list[str] = []
        self.pending: list[TokenEvent] = []
        self.added: list[str] = []
        self.aborted: list[str] = []
        self.step_calls = 0

    def add_request(self, request: InferenceRequest) -> None:
        self.added.append(request.request_id)
        self.active.append(request.request_id)

    def abort(self, request_id: str) -> None:
        self.aborted.append(request_id)
        if request_id in self.active:
            self.active.remove(request_id)
            self.pending.append(fin(request_id, "abort"))

    def is_idle(self) -> bool:
        return not self.active and not self.pending

    def step(self) -> list[TokenEvent]:
        self.step_calls += 1
        events = self.pending
        self.pending = []
        for rid in list(self.active):
            script = self.scripts.get(rid)
            evt = script.pop(0) if script else fin(rid)
            events.append(evt)
            if evt.finish_reason is not None or evt.error is not None:
                self.active.remove(rid)
        return events


class FailingScheduler(ScriptedScheduler):
    def step(self) -> list[TokenEvent]:
        super().step()
        raise RuntimeError("forward exploded")


async def start_engine(scheduler) -> ContinuousBatchingEngine:
    engine = ContinuousBatchingEngine(scheduler)
    await engine.start()
    return engine


async def collect(engine, req) -> list[TokenEvent]:
    return [evt async for evt in engine.submit(req)]


async def wait_for(predicate, timeout: float = 2.0) -> None:
    deadline = asyncio.get_event_loop().time() + timeout
    while not predicate():
        assert asyncio.get_event_loop().time() < deadline, "timed out waiting"
        await asyncio.sleep(0.005)


def assert_one_field_each(events: list[TokenEvent]) -> None:
    for evt in events:
        populated = [
            f for f in (evt.token_id, evt.finish_reason, evt.error) if f is not None
        ]
        assert len(populated) == 1, f"event populates {len(populated)} fields: {evt}"


class TestStreaming:
    def test_tokens_stream_in_order_then_finish_terminates(self):
        async def main():
            sched = ScriptedScheduler(
                {"r1": [tok("r1", 10), tok("r1", 11), tok("r1", 12), fin("r1")]}
            )
            engine = await start_engine(sched)
            events = await collect(engine, make_request("r1"))
            await engine.shutdown()
            return events

        events = asyncio.run(main())
        assert [e.token_id for e in events[:-1]] == [10, 11, 12]
        assert events[-1].finish_reason == "end_turn"
        assert_one_field_each(events)

    def test_interleaved_requests_route_by_request_id(self):
        async def main():
            sched = ScriptedScheduler({
                "r1": [tok("r1", 10), tok("r1", 11), fin("r1")],
                "r2": [tok("r2", 20), tok("r2", 21), tok("r2", 22), fin("r2")],
            })
            engine = await start_engine(sched)
            s1, s2 = await asyncio.gather(
                collect(engine, make_request("r1")),
                collect(engine, make_request("r2")),
            )
            await engine.shutdown()
            return s1, s2

        s1, s2 = asyncio.run(main())
        assert all(e.request_id == "r1" for e in s1)
        assert all(e.request_id == "r2" for e in s2)
        assert [e.token_id for e in s1[:-1]] == [10, 11]
        assert [e.token_id for e in s2[:-1]] == [20, 21, 22]


class TestAbort:
    def test_early_consumer_break_sends_abort_command(self):
        async def main():
            sched = ScriptedScheduler(
                {"r1": [tok("r1", i) for i in range(50)] + [fin("r1")]}
            )
            engine = await start_engine(sched)
            async for evt in engine.submit(make_request("r1")):
                break  # client disconnects after the first token
            await wait_for(lambda: "r1" in sched.aborted)
            await engine.shutdown()
            return sched

        sched = asyncio.run(main())
        assert sched.aborted == ["r1"]

    def test_explicit_abort_closes_stream_with_abort_reason(self):
        async def main():
            sched = ScriptedScheduler(
                {"r1": [tok("r1", i) for i in range(50)] + [fin("r1")]}
            )
            engine = await start_engine(sched)
            events = []
            async for evt in engine.submit(make_request("r1")):
                events.append(evt)
                if len(events) == 2:
                    engine.abort("r1")
            await engine.shutdown()
            return events

        events = asyncio.run(main())
        assert events[-1].finish_reason == "abort"
        assert_one_field_each(events)

    def test_abort_of_unknown_request_is_harmless(self):
        async def main():
            sched = ScriptedScheduler({"r1": [tok("r1", 1), fin("r1")]})
            engine = await start_engine(sched)
            engine.abort("ghost")
            events = await collect(engine, make_request("r1"))
            await engine.shutdown()
            return events

        events = asyncio.run(main())
        assert events[-1].finish_reason == "end_turn"


class TestIdle:
    def test_idle_engine_does_not_spin_and_late_submit_wakes_it(self):
        async def main():
            sched = ScriptedScheduler({
                "r1": [tok("r1", 1), fin("r1")],
                "r2": [tok("r2", 2), fin("r2")],
            })
            engine = await start_engine(sched)
            await collect(engine, make_request("r1"))

            await asyncio.sleep(0.05)
            calls_after_r1 = sched.step_calls
            await asyncio.sleep(0.1)
            calls_after_wait = sched.step_calls

            late = await collect(engine, make_request("r2"))
            await engine.shutdown()
            return calls_after_r1, calls_after_wait, late, sched.step_calls

        calls_after_r1, calls_after_wait, late, final_calls = asyncio.run(main())
        assert calls_after_wait == calls_after_r1, "idle engine kept stepping"
        assert late[-1].finish_reason == "end_turn", "late submit was not served"
        assert final_calls > calls_after_wait


class TestShutdown:
    def test_shutdown_with_inflight_closes_iterator_and_joins_thread(self):
        async def main():
            sched = ScriptedScheduler(
                {"r1": [tok("r1", i) for i in range(10_000)]}
            )
            engine = await start_engine(sched)

            events = []

            async def consume():
                async for evt in engine.submit(make_request("r1")):
                    events.append(evt)

            task = asyncio.create_task(consume())
            await wait_for(lambda: len(events) >= 3)
            await engine.shutdown()
            await asyncio.wait_for(task, timeout=2.0)
            return events, engine._thread.is_alive()

        events, thread_alive = asyncio.run(main())
        assert not thread_alive
        assert events[-1].finish_reason == "abort"

    def test_shutdown_before_start_is_a_noop(self):
        async def main():
            engine = ContinuousBatchingEngine(ScriptedScheduler({}))
            await engine.shutdown()

        asyncio.run(main())

    def test_submit_after_shutdown_fails_fast(self):
        """A submit() arriving after shutdown must not hang: with no scheduler
        thread to drain the command queue, awaiting the first event would
        block forever. shutdown() latches _failed so submit() errors out."""
        async def main():
            engine = await start_engine(ScriptedScheduler({}))
            await engine.shutdown()
            return await asyncio.wait_for(
                collect(engine, make_request("late")), timeout=2.0
            )

        stream = asyncio.run(main())
        assert len(stream) == 1
        assert stream[0].error is not None
        assert "shut down" in stream[0].error


class TestStepFailure:
    def test_step_exception_errors_all_inflight_and_fails_engine(self):
        async def main():
            sched = FailingScheduler({
                "r1": [tok("r1", i) for i in range(50)],
                "r2": [tok("r2", i) for i in range(50)],
            })
            engine = await start_engine(sched)
            s1, s2 = await asyncio.gather(
                collect(engine, make_request("r1")),
                collect(engine, make_request("r2")),
            )
            late = await collect(engine, make_request("r3"))
            await engine.shutdown()
            return s1, s2, late

        s1, s2, late = asyncio.run(main())
        for stream in (s1, s2):
            assert len(stream) == 1
            assert "forward exploded" in stream[0].error
        assert len(late) == 1 and "forward exploded" in late[0].error

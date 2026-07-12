"""EngineProcessClient tests: the CB engine behind a real spawned process.

Same layering as the shell tests: scripted doubles pin the boundary
behaviors (ready handshake, batch-wide failure, crash detection, shutdown),
and the toy-oracle equivalence pins that a real scheduler behind IPC still
streams the right tokens. Factories and scheduler doubles are module-level
so they pickle across the spawn boundary — the child rebuilds everything on
its side; nothing rich crosses.

Each test that talks to a child spawns a fresh interpreter (torch import
included), so this file is seconds-per-test where the shell tests are
milliseconds. That's the cost of testing the real boundary; keep new tests
here to behaviors that genuinely need a process.
"""

import asyncio
import os
import pickle

import httpx
import pytest

from cantollm.api import create_app
from cantollm.engine.batching import BatchingConfig, EngineProcessClient
from cantollm.engine.batching.allocator import SlotAllocator
from cantollm.engine.batching.scheduler import ContinuousBatchingScheduler
from cantollm.engine.batching.types import AddRequest
from cantollm.engine.logits_processors import TemperatureProcessor, TopPProcessor
from cantollm.engine.types import InferenceRequest, SamplingParams, TokenEvent
from cantollm.registry import EngineRegistry
from tests.fakes import FakeRuntime, FakeTokenizer, parse_sse
from tests.toy_stepper import ToyStepper, make_toy_pool, toy_oracle

GREEDY = SamplingParams.from_temperature_top_p(temperature=0.0, top_p=1.0)


# ── module-level factories (pickled by reference across spawn) ────────


def toy_scheduler_factory(max_batch=3, max_seq_len=64, max_tokens_per_step=8):
    config = BatchingConfig(
        max_batch=max_batch,
        max_seq_len=max_seq_len,
        max_tokens_per_step=max_tokens_per_step,
    )
    return ContinuousBatchingScheduler(
        forward_fn=ToyStepper(),
        pool=make_toy_pool(config),
        allocator=SlotAllocator(config.max_batch),
        config=config,
    )


class NeverFinishingScheduler:
    """SchedulerLike double: one token per active request per step; a stream
    only ever terminates via abort."""

    def __init__(self):
        self.active: list[str] = []
        self.pending: list[TokenEvent] = []
        self.steps = 0

    def add_request(self, request: InferenceRequest) -> None:
        self.active.append(request.request_id)

    def abort(self, request_id: str) -> None:
        if request_id in self.active:
            self.active.remove(request_id)
            self.pending.append(
                TokenEvent(finish_reason="abort", request_id=request_id)
            )

    def is_idle(self) -> bool:
        return not self.active and not self.pending

    def step(self) -> list[TokenEvent]:
        self.steps += 1
        events = self.pending
        self.pending = []
        for rid in list(self.active):
            events.append(TokenEvent(token_id=self.steps, request_id=rid))
        return events


class FailingScheduler(NeverFinishingScheduler):
    """One good step, then the forward 'explodes' — batch-wide failure."""

    def step(self) -> list[TokenEvent]:
        if self.steps >= 1:
            raise RuntimeError("forward exploded")
        return super().step()


class DyingScheduler(NeverFinishingScheduler):
    """Simulates a hard engine-process death (segfault, OOM-kill) mid-step:
    no farewell message ever reaches the bridge."""

    def step(self) -> list[TokenEvent]:
        os._exit(3)


def never_finishing_factory():
    return NeverFinishingScheduler()


def failing_factory():
    return FailingScheduler()


def dying_factory():
    return DyingScheduler()


def broken_factory():
    raise RuntimeError("weights fell off the truck")


# ── helpers ───────────────────────────────────────────────────────────


def make_request(rid, prompt, max_tokens=6, sampling=GREEDY, stop=None):
    return InferenceRequest(
        request_id=rid,
        prompt_token_ids=list(prompt),
        sampling_params=sampling,
        max_tokens=max_tokens,
        stop_token_ids=stop or set(),
    )


async def collect(engine, req) -> list[TokenEvent]:
    return [evt async for evt in engine.submit(req)]


async def start_client(factory, **factory_kwargs) -> EngineProcessClient:
    client = EngineProcessClient(factory, factory_kwargs)
    await client.start()
    return client


async def wait_for(predicate, timeout: float = 5.0) -> None:
    deadline = asyncio.get_event_loop().time() + timeout
    while not predicate():
        assert asyncio.get_event_loop().time() < deadline, "timed out waiting"
        await asyncio.sleep(0.01)


# ── tests ─────────────────────────────────────────────────────────────


class TestWireFormat:
    def test_inference_request_round_trips_through_pickle(self):
        """Commands cross the boundary via mp.Queue's pickler; the full
        SamplingParams pipeline must survive it."""
        req = make_request(
            "r1", [1, 2, 3], max_tokens=7,
            sampling=SamplingParams.from_temperature_top_p(0.7, 0.9),
            stop={5, 6},
        )
        clone = pickle.loads(pickle.dumps(AddRequest(req))).request
        assert clone.request_id == req.request_id
        assert clone.prompt_token_ids == req.prompt_token_ids
        assert clone.stop_token_ids == req.stop_token_ids
        assert clone.max_tokens == req.max_tokens
        assert [type(p) for p in clone.sampling_params.processors] == [
            TemperatureProcessor, TopPProcessor,
        ]
        assert clone.sampling_params.processors[0].temperature == 0.7
        assert clone.sampling_params.processors[1].top_p == 0.9


class TestEquivalence:
    def test_concurrent_streams_match_toy_oracle(self):
        """The real scheduler behind IPC produces the same tokens, finish
        reasons, and routing as a per-request sequential reference — and a
        sampled (non-greedy) request's processors survive the boundary."""
        sampled = make_request(
            "sampled", [20, 21], max_tokens=5,
            sampling=SamplingParams.from_temperature_top_p(0.8, 0.9),
        )

        async def main():
            client = await start_client(toy_scheduler_factory)
            try:
                greedy_requests = [
                    make_request("r0", [1, 2, 3, 4, 5], max_tokens=6),
                    make_request("r1", [7, 8], max_tokens=4),
                    make_request("r2", [9, 10, 11], max_tokens=8),
                ]
                streams = await asyncio.gather(
                    *(collect(client, r) for r in greedy_requests),
                    collect(client, sampled),
                )
                return greedy_requests, streams[:-1], streams[-1]
            finally:
                await client.shutdown()

        greedy_requests, greedy_streams, sampled_stream = asyncio.run(main())
        for req, events in zip(greedy_requests, greedy_streams):
            want_tokens, want_reason = toy_oracle(req)
            assert [e.token_id for e in events[:-1]] == want_tokens
            assert events[-1].finish_reason == want_reason
            assert all(e.request_id == req.request_id for e in events)
        assert sampled_stream[-1].finish_reason == "max_tokens"
        assert len(sampled_stream) == 6  # 5 tokens + finish


class TestStartup:
    def test_factory_failure_surfaces_in_start_and_latches(self):
        async def main():
            client = EngineProcessClient(broken_factory)
            with pytest.raises(RuntimeError, match="weights fell off the truck"):
                await client.start()
            # The failure latches: submits error out instead of hanging.
            return await asyncio.wait_for(
                collect(client, make_request("late", [1])), timeout=5
            )

        events = asyncio.run(main())
        assert len(events) == 1
        assert "weights fell off the truck" in events[0].error


class TestFailure:
    def test_step_exception_errors_inflight_and_fails_fast_after(self):
        async def main():
            client = await start_client(failing_factory)
            try:
                events = await asyncio.wait_for(
                    collect(client, make_request("r1", [1, 2])), timeout=10
                )
                late = await asyncio.wait_for(
                    collect(client, make_request("r2", [3])), timeout=10
                )
                return events, late
            finally:
                await client.shutdown()

        events, late = asyncio.run(main())
        assert events[-1].error is not None
        assert "forward exploded" in events[-1].error
        assert len(late) == 1 and "forward exploded" in late[0].error

    def test_process_death_fails_inflight_streams(self):
        """No farewell message — the bridge must notice the corpse."""
        async def main():
            client = await start_client(dying_factory)
            try:
                return await asyncio.wait_for(
                    collect(client, make_request("r1", [1])), timeout=10
                )
            finally:
                await client.shutdown()

        events = asyncio.run(main())
        assert len(events) == 1
        assert "engine process died" in events[0].error
        assert "exit code 3" in events[0].error


class TestAbort:
    def test_explicit_abort_terminates_stream(self):
        async def main():
            client = await start_client(never_finishing_factory)
            try:
                events = []
                async for evt in client.submit(
                    make_request("r1", [1], max_tokens=10_000)
                ):
                    events.append(evt)
                    if len(events) == 2:
                        client.abort("r1")
                return events
            finally:
                await client.shutdown()

        events = asyncio.run(main())
        assert events[-1].finish_reason == "abort"

    def test_disconnect_frees_the_slot_for_the_next_request(self):
        """A consumer that walks away mid-stream must translate into an
        Abort command in the child; with one slot, the follow-up request can
        only complete if the abort actually freed it."""
        async def main():
            client = await start_client(
                toy_scheduler_factory,
                max_batch=1, max_seq_len=64, max_tokens_per_step=8,
            )
            try:
                async for _ in client.submit(
                    make_request("r1", [1, 2], max_tokens=50)
                ):
                    break  # disconnect after the first token
                return await asyncio.wait_for(
                    collect(client, make_request("r2", [3, 4], max_tokens=4)),
                    timeout=10,
                )
            finally:
                await client.shutdown()

        events = asyncio.run(main())
        assert events[-1].finish_reason == "max_tokens"
        assert len(events) == 5  # 4 tokens + finish


class TestShutdown:
    def test_shutdown_closes_streams_and_reaps_the_child(self):
        async def main():
            client = await start_client(never_finishing_factory)
            events = []

            async def consume():
                async for evt in client.submit(
                    make_request("r1", [1], max_tokens=10_000)
                ):
                    events.append(evt)

            task = asyncio.create_task(consume())
            await wait_for(lambda: len(events) >= 2)
            await client.shutdown()
            await asyncio.wait_for(task, timeout=5)
            late = await asyncio.wait_for(
                collect(client, make_request("r2", [1])), timeout=5
            )
            return events, late, client

        events, late, client = asyncio.run(main())
        assert events[-1].finish_reason == "abort"
        assert client._proc.exitcode == 0
        assert not client._bridge.is_alive()
        assert len(late) == 1 and "shut down" in late[0].error

    def test_shutdown_before_start_is_a_noop(self):
        async def main():
            await EngineProcessClient(toy_scheduler_factory).shutdown()

        asyncio.run(main())


class TestApiSmoke:
    def test_sse_stream_over_the_process_engine(self):
        """The full path: HTTP → tokenize (API side) → IPC → toy scheduler
        in a child → IPC → SSE. The registry entry holds a tokenizer-only
        runtime; no model exists in this process."""
        async def main():
            registry = EngineRegistry()
            registry.register(
                "toy",
                EngineProcessClient(toy_scheduler_factory),
                FakeRuntime(FakeTokenizer()),
                max_request_tokens=64,
            )
            app = create_app(registry)
            transport = httpx.ASGITransport(app=app)
            async with app.router.lifespan_context(app):  # start/stop engine
                async with httpx.AsyncClient(
                    transport=transport, base_url="http://test"
                ) as http:
                    resp = await http.post(
                        "/v1/messages",
                        json={
                            "model": "toy",
                            "max_tokens": 5,
                            "temperature": 0.0,
                            "stream": True,
                            "messages": [
                                {"role": "user", "content": "hi"}
                            ],
                        },
                    )
                    return resp.status_code, parse_sse(resp.text)

        status, events = asyncio.run(main())
        assert status == 200
        kinds = [e.event for e in events if e.event != "ping"]
        assert kinds[0] == "message_start"
        assert kinds[-1] == "message_stop"
        deltas = [e for e in events if e.event == "content_block_delta"]
        assert deltas, "no content streamed over the process boundary"

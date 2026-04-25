"""Smoke tests for the tiny Qwen3 fixture.

Validates that the fixture wires a real Qwen3 forward pass through
build_runtime -> StandardBackend -> SequentialEngine -> TokenEvent stream.
Phase 2's scheduler/batching tests will lean on this seam working.
"""

import asyncio

import torch

from cantollm.engine.sequential import SequentialEngine
from cantollm.engine.types import InferenceRequest, SamplingParams
from cantollm.runtime import build_runtime
from tests.tiny_model import TINY_ARCH, tiny_qwen3_spec


def _run(coro):
    return asyncio.run(coro)


def test_tiny_runtime_builds_and_forward_runs():
    runtime = build_runtime(tiny_qwen3_spec(), torch.device("cpu"))

    assert runtime.spec.name == "qwen3-tiny"
    assert runtime.tokenizer is not None
    assert len(runtime.new_cache()) == TINY_ARCH["num_transformers"]

    cache = runtime.new_cache()
    tokens = torch.tensor([[1, 2, 3, 4]])
    with torch.no_grad():
        logits = runtime.model(tokens, start_pos=0, kv_cache=cache)

    assert logits.shape == (1, 4, TINY_ARCH["token_count"])
    # Cache populated by the forward pass
    assert cache.position == 4


def test_tiny_engine_generates_max_tokens():
    runtime = build_runtime(tiny_qwen3_spec(), torch.device("cpu"))
    engine = SequentialEngine(runtime)

    req = InferenceRequest(
        request_id="r1",
        prompt_token_ids=[1, 2, 3, 4],
        sampling_params=SamplingParams(greedy=True),
        max_tokens=4,
        stop_token_ids=set(),
    )

    async def drain():
        events = []
        async for evt in engine.submit(req):
            events.append(evt)
        return events

    events = _run(drain())

    token_events = [e for e in events if e.token_id is not None]
    finish_events = [e for e in events if e.finish_reason is not None]

    assert len(token_events) == 4
    assert len(finish_events) == 1
    assert finish_events[0].finish_reason == "max_tokens"
    assert all(e.request_id == "r1" for e in events)


def test_tiny_engine_stops_on_stop_token():
    runtime = build_runtime(tiny_qwen3_spec(), torch.device("cpu"))
    engine = SequentialEngine(runtime)
    prompt = [1, 2, 3, 4]

    # Greedy on a fixed model + prompt is deterministic — probe to find the
    # first token, then submit a real request with that token as the stop.
    probe = InferenceRequest(
        request_id="probe",
        prompt_token_ids=prompt,
        sampling_params=SamplingParams(greedy=True),
        max_tokens=1,
        stop_token_ids=set(),
    )

    async def first_token():
        async for evt in engine.submit(probe):
            if evt.token_id is not None:
                return evt.token_id
        return None

    first = _run(first_token())
    assert first is not None

    req = InferenceRequest(
        request_id="r2",
        prompt_token_ids=prompt,
        sampling_params=SamplingParams(greedy=True),
        max_tokens=8,
        stop_token_ids={first},
    )

    async def drain():
        events = []
        async for evt in engine.submit(req):
            events.append(evt)
        return events

    events = _run(drain())
    token_events = [e for e in events if e.token_id is not None]
    finish_events = [e for e in events if e.finish_reason is not None]

    # The backend short-circuits when the sampled token hits stop_token_ids
    # *before* yielding, so we expect zero token events and an end_turn finish.
    assert len(token_events) == 0
    assert len(finish_events) == 1
    assert finish_events[0].finish_reason == "end_turn"

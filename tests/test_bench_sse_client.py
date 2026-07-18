"""SSE client tests over ASGITransport against the real app + FakeEngine.

Pins the §3 timestamp taxonomy: ordering, TTFT-counts-thinking, usage-based
token counts, finish-reason mapping per dialect, and error capture.
"""

import asyncio

import httpx

from cantollm.api import create_app
from cantollm.bench.sse_clients import SendOptions, build_sender
from cantollm.bench.workloads import Prompt
from tests.fakes import (
    THINKING_END_ID,
    THINKING_START_ID,
    FakeEngine,
    FakeRegistry,
    FakeRuntime,
    FakeTokenizer,
    ScriptStep,
)


def _char_ids(text: str) -> list[int]:
    return [2000 + (ord(c) - ord("a")) for c in text]


def _script(text: str, *, sleep_first: float = 0.0) -> list[ScriptStep]:
    steps = [ScriptStep(token_id=tid) for tid in _char_ids(text)]
    if steps and sleep_first:
        steps[0] = ScriptStep(token_id=steps[0].token_id, sleep=sleep_first)
    return steps


PROMPT = Prompt(id="p0", messages=({"role": "user", "content": "hi"},),
                system=None, input_tokens=3)


def run_send(engine: FakeEngine, options: SendOptions):
    tokenizer = FakeTokenizer(
        id_to_text={2000 + i: chr(ord("a") + i) for i in range(26)}
    )
    registry = FakeRegistry(entries={"test-model": (engine, FakeRuntime(tokenizer))})
    app = create_app(registry)

    async def main():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            send = build_sender(client, options)
            return await send(PROMPT, cell_id="c", repeat=0, request_index=0)

    return asyncio.run(main())


def test_openai_happy_path_taxonomy_and_usage():
    engine = FakeEngine(script=_script("hello", sleep_first=0.05))
    record, text = run_send(
        engine,
        SendOptions(model="test-model", max_tokens=5, ignore_eos=True,
                    capture_text=True),
    )
    assert record.ok and record.error is None
    assert record.http_status == 200
    assert record.t_send < record.t_headers <= record.t_first_token <= record.t_done
    assert record.ttft_s >= 0.05                  # the scripted pre-token sleep
    assert record.input_tokens == 3               # from usage, not counted
    assert record.output_tokens == 5
    assert record.finish_reason == "length"       # max_tokens exit
    assert record.client_itl_mean_s is not None
    assert text == "hello"


def test_openai_thinking_tokens_open_ttft():
    script = [
        ScriptStep(token_id=THINKING_START_ID),
        ScriptStep(token_id=_char_ids("t")[0], sleep=0.03),
        ScriptStep(token_id=THINKING_END_ID),
        *_script("ok"),
    ]
    engine = FakeEngine(script=script)
    record, _ = run_send(
        engine, SendOptions(model="test-model", max_tokens=50, ignore_eos=False),
    )
    assert record.ok
    assert record.reasoning_tokens > 0
    # TTFT fired on the reasoning_content delta, not the later text.
    assert record.ttft_s < record.completion_s


def test_openai_http_400_is_captured_not_raised():
    engine = FakeEngine(script=_script("x"))
    record, _ = run_send(
        engine,
        SendOptions(model="no-such-model", max_tokens=5),
    )
    assert not record.ok
    assert record.http_status == 404
    assert record.error == "http 404"
    assert record.finish_reason is None


def test_openai_midstream_error_event_recorded():
    engine = FakeEngine(script=[
        ScriptStep(token_id=_char_ids("a")[0]),
        ScriptStep(raise_error=RuntimeError("engine exploded")),
    ])
    record, _ = run_send(
        engine, SendOptions(model="test-model", max_tokens=50, ignore_eos=False),
    )
    # OpenAI mid-stream errors surface as an error chunk then [DONE]; the
    # stream ends without a finish_reason and token counts stay partial.
    assert record.finish_reason is None
    assert record.t_first_token is not None


def test_anthropic_dialect_taxonomy():
    engine = FakeEngine(script=_script("hey", sleep_first=0.03))
    record, text = run_send(
        engine,
        SendOptions(model="test-model", dialect="anthropic", max_tokens=3,
                    ignore_eos=True, capture_text=True),
    )
    assert record.ok
    assert record.dialect == "anthropic"
    assert record.t_send < record.t_headers <= record.t_first_token <= record.t_done
    assert record.ttft_s >= 0.03
    assert record.input_tokens == 3
    assert record.output_tokens == 3
    assert record.finish_reason == "max_tokens"
    assert text == "hey"

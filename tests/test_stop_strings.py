"""Stop strings (tail step 13): text-level stop sequences on both dialects.

Stop strings can span token boundaries and start mid-token, so matching
lives at the decoded-text layer (`StopStringWatcher` in decoder.py) with
holdback: no partial stop string ever reaches the client. On a match the
adapter closes the engine stream — the disconnect→abort path — so
generation actually stops and (on the CB engine) the slot frees.
"""

import asyncio

import httpx

from cantollm.api import create_app
from cantollm.decoder import StopStringWatcher
from tests.fakes import (
    FakeEngine,
    FakeRegistry,
    FakeRuntime,
    FakeTokenizer,
    ScriptStep,
    parse_openai_sse,
    parse_sse,
)


class TestStopStringWatcher:
    def test_passthrough_without_stops(self):
        w = StopStringWatcher([])
        assert w.feed("hello") == "hello"
        assert w.matched is None

    def test_match_within_one_chunk(self):
        w = StopStringWatcher(["STOP"])
        assert w.feed("abcSTOPxyz") == "abc"
        assert w.matched == "STOP"
        assert w.feed("more") == ""  # nothing escapes after a match

    def test_match_spanning_chunks_never_leaks_partial(self):
        w = StopStringWatcher(["STOP"])
        out = w.feed("abcST")
        assert out == "abc"  # "ST" held back — could be a match forming
        assert w.matched is None
        assert w.feed("OPxyz") == ""
        assert w.matched == "STOP"

    def test_holdback_released_when_match_falls_through(self):
        w = StopStringWatcher(["STOP"])
        assert w.feed("abcST") == "abc"
        assert w.feed("ART") == "START"  # "ST" wasn't a match after all
        assert w.matched is None

    def test_earliest_match_wins(self):
        w = StopStringWatcher(["yy", "x"])
        assert w.feed("abxcyy") == "ab"
        assert w.matched == "x"

    def test_char_by_char_feed(self):
        w = StopStringWatcher(["ll"])
        released = "".join(w.feed(c) for c in "hello")
        assert released == "he"
        assert w.matched == "ll"

    def test_flush_releases_tail(self):
        w = StopStringWatcher(["STOP"])
        w.feed("endsWithS")
        assert w.flush() == "S"

    def test_repeated_prefix_overlap(self):
        w = StopStringWatcher(["aba"])
        out = w.feed("ab") + w.feed("ab")  # "abab" contains "aba"
        assert out == ""
        assert w.matched == "aba"


def _client(script_text: str):
    """FakeEngine emitting one token per character of `script_text`."""
    ids = {2000 + i: c for i, c in enumerate(script_text)}
    script = [ScriptStep(token_id=tid) for tid in ids]
    engine = FakeEngine(script=script)
    tokenizer = FakeTokenizer(id_to_text=ids)
    registry = FakeRegistry(
        entries={"m": (engine, FakeRuntime(tokenizer=tokenizer))}
    )
    app = create_app(registry)
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://t"), engine


class TestAnthropicStopSequences:
    def _body(self, stop_sequences, stream=False):
        return {
            "model": "m", "max_tokens": 50, "stream": stream,
            "stop_sequences": stop_sequences,
            "messages": [{"role": "user", "content": "hi"}],
        }

    def test_non_streaming_match(self):
        async def main():
            client, engine = _client("hello world")
            async with client:
                resp = await client.post(
                    "/v1/messages", json=self._body(["lo w"])
                )
            return resp, engine

        resp, engine = asyncio.run(main())
        body = resp.json()
        text = "".join(b.get("text", "") for b in body["content"])
        assert text == "hel"
        assert body["stop_reason"] == "stop_sequence"
        assert body["stop_sequence"] == "lo w"
        assert engine.aborted, "engine kept generating after the match"

    def test_streaming_match_never_leaks_stop_text(self):
        async def main():
            client, engine = _client("hello world")
            async with client:
                resp = await client.post(
                    "/v1/messages", json=self._body(["lo w"], stream=True)
                )
            return resp, engine

        resp, engine = asyncio.run(main())
        events = parse_sse(resp.text)
        deltas = "".join(
            e.data["delta"]["text"] for e in events
            if e.event == "content_block_delta" and e.data["delta"]["type"] == "text_delta"
        )
        assert deltas == "hel"
        message_delta = next(e for e in events if e.event == "message_delta")
        assert message_delta.data["delta"]["stop_reason"] == "stop_sequence"
        assert message_delta.data["delta"]["stop_sequence"] == "lo w"
        assert any(e.event == "message_stop" for e in events)
        assert engine.aborted

    def test_no_match_releases_heldback_tail(self):
        async def main():
            client, engine = _client("hello")
            async with client:
                resp = await client.post(
                    "/v1/messages", json=self._body(["loXYZ"])
                )
            return resp, engine

        resp, engine = asyncio.run(main())
        body = resp.json()
        text = "".join(b.get("text", "") for b in body["content"])
        assert text == "hello"  # the held "lo" came back on flush
        assert body["stop_reason"] == "end_turn"
        assert body["stop_sequence"] is None
        # (No `engine.aborted` assertion: the adapter always breaks on the
        # finish event, so the fake's finally marks aborted either way.)


class TestOpenAIStop:
    def _body(self, stop, stream=False):
        return {
            "model": "m", "max_tokens": 50, "stream": stream, "stop": stop,
            "messages": [{"role": "user", "content": "hi"}],
        }

    def test_non_streaming_string_stop(self):
        async def main():
            client, engine = _client("hello world")
            async with client:
                resp = await client.post(
                    "/v1/chat/completions", json=self._body("lo w")
                )
            return resp, engine

        resp, engine = asyncio.run(main())
        choice = resp.json()["choices"][0]
        assert choice["message"]["content"] == "hel"
        assert choice["finish_reason"] == "stop"
        assert engine.aborted

    def test_streaming_list_stop(self):
        async def main():
            client, engine = _client("hello world")
            async with client:
                resp = await client.post(
                    "/v1/chat/completions",
                    json=self._body(["zzz", "lo w"], stream=True),
                )
            return resp, engine

        resp, engine = asyncio.run(main())
        chunks, saw_done = parse_openai_sse(resp.text)
        assert saw_done
        text = "".join(
            c["choices"][0]["delta"].get("content") or ""
            for c in chunks if c["choices"]
        )
        assert text == "hel"
        finishes = [c["choices"][0]["finish_reason"]
                    for c in chunks if c["choices"]]
        assert finishes[-1] == "stop"
        assert engine.aborted

"""Test doubles for API contract tests.

`FakeTokenizer` exposes just the surface the adapter + decoder touch; token IDs
map 1:1 to single characters so tests can reason about output text directly.
`FakeEngine` scripts a token stream per request, with optional sleeps and a
mid-stream exception, and records when its async generator was closed early
(which is how the adapter signals a client disconnect today).
"""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

from cantollm.engine.types import InferenceRequest, TokenEvent

THINKING_START_ID = 1000
THINKING_END_ID = 1001
STOP_TOKEN_ID = 999


class _FakeIncrementalDecoder:
    def __init__(self, id_to_text: dict[int, str]):
        self._map = id_to_text

    def add(self, token_id: int) -> str:
        return self._map.get(token_id, "")

    def flush(self) -> str:
        return ""


class FakeTokenizer:
    """Minimal tokenizer double.

    `id_to_text` maps every non-special token id to the exact text it decodes
    to; `encode_conversation` is only used by the app to build an
    InferenceRequest, so it just returns a fixed prompt.
    """

    def __init__(self, id_to_text: dict[int, str] | None = None, prompt_len: int = 3):
        self.thinking_start_id = THINKING_START_ID
        self.thinking_end_id = THINKING_END_ID
        self.stop_token_ids = {STOP_TOKEN_ID}
        self._id_to_text = id_to_text or {i: chr(ord("a") + (i % 26)) for i in range(256)}
        self._prompt_len = prompt_len

    def encode_conversation(self, messages, system=None) -> list[int]:
        return [1] * self._prompt_len

    def incremental_decoder(self) -> _FakeIncrementalDecoder:
        return _FakeIncrementalDecoder(self._id_to_text)


@dataclass
class ScriptStep:
    """One step in a FakeEngine script — either a token or a pre-token sleep."""

    token_id: int | None = None
    sleep: float = 0.0
    raise_error: BaseException | None = None


@dataclass
class FakeRuntime:
    """Runtime double: exposes just `.tokenizer` (what the API layer reads)."""

    tokenizer: FakeTokenizer

    async def start(self) -> None:
        pass

    async def shutdown(self) -> None:
        pass


class FakeRegistry:
    """Minimal EngineRegistry stand-in for contract tests."""

    def __init__(self, entries: dict[str, tuple["FakeEngine", FakeRuntime]]):
        self._entries = {
            name: _FakeEntry(engine=eng, runtime=rt) for name, (eng, rt) in entries.items()
        }

    def get(self, name: str):
        return self._entries[name]

    def names(self) -> list[str]:
        return list(self._entries)

    def items(self):
        return self._entries.items()

    async def start_all(self) -> None:
        for entry in self._entries.values():
            await entry.runtime.start()
            await entry.engine.start()

    async def shutdown_all(self) -> None:
        for entry in self._entries.values():
            await entry.engine.shutdown()
            await entry.runtime.shutdown()


@dataclass
class _FakeEntry:
    engine: "FakeEngine"
    runtime: FakeRuntime
    registered_at: float = 0.0


@dataclass
class FakeEngine:
    """Engine double that replays a scripted list of ScriptSteps.

    `submit` is an async generator. If the consumer stops iterating early (the
    client-disconnect path), `aborted` flips True via the generator's finally
    block. `abort(request_id)` is recorded in `abort_calls` for completeness,
    though the current adapter doesn't call it.
    """

    script: list[ScriptStep] = field(default_factory=list)
    aborted: bool = False
    completed: bool = False
    abort_calls: list[str] = field(default_factory=list)
    started: bool = False
    shutdown_called: bool = False

    async def start(self) -> None:
        self.started = True

    async def shutdown(self) -> None:
        self.shutdown_called = True

    def abort(self, request_id: str) -> None:
        self.abort_calls.append(request_id)

    async def submit(self, req: InferenceRequest) -> AsyncIterator[TokenEvent]:
        tokens_emitted = 0
        try:
            for step in self.script:
                if step.sleep:
                    await asyncio.sleep(step.sleep)
                if step.raise_error is not None:
                    yield TokenEvent(
                        error=str(step.raise_error), request_id=req.request_id
                    )
                    self.completed = True
                    return
                if step.token_id is not None:
                    yield TokenEvent(token_id=step.token_id, request_id=req.request_id)
                    tokens_emitted += 1
            if tokens_emitted >= req.max_tokens:
                reason = "max_tokens"
            else:
                reason = "end_turn"
            yield TokenEvent(finish_reason=reason, request_id=req.request_id)
            self.completed = True
        finally:
            if not self.completed:
                self.aborted = True


# ── SSE parser ────────────────────────────────────────────────────────


_EVENT_LINE = re.compile(r"^event:\s*(.*)$", re.MULTILINE)
_DATA_LINE = re.compile(r"^data:\s*(.*)$", re.MULTILINE)


@dataclass
class SSEEvent:
    event: str
    data: dict | str


def parse_sse(body: str) -> list[SSEEvent]:
    """Parse an `event:/data:` SSE stream into typed records.

    Keeps the ping event (`event: ping\\ndata: {}`) as a dict too, so tests
    can filter them in or out explicitly.
    """
    out: list[SSEEvent] = []
    for block in body.split("\n\n"):
        block = block.strip("\n")
        if not block:
            continue
        ev_match = _EVENT_LINE.search(block)
        data_match = _DATA_LINE.search(block)
        if not ev_match or not data_match:
            continue
        raw = data_match.group(1)
        try:
            data: dict | str = json.loads(raw)
        except json.JSONDecodeError:
            data = raw
        out.append(SSEEvent(event=ev_match.group(1), data=data))
    return out

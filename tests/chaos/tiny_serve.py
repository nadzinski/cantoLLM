"""The chaos suite's model: tiny Qwen3 behind the real serve path.

`CANTOLLM_TEST_SPEC=tests.chaos.tiny_serve:chaos_tiny_spec` makes
`canto serve --model tiny` build this spec (see spec.qwen3_spec). It is
tests/tiny_model.py's 2-layer fixture with one change: the fake tokenizer
derives prompt length from message content, so mixed-size prompts are
actually mixed-size at the scheduler (the exit criterion's 50-client run
needs real length variety, and the fixture default is a constant 3)."""

from __future__ import annotations

from dataclasses import replace

from cantollm.spec import ModelSpec
from tests.fakes import FakeTokenizer
from tests.tiny_model import tiny_qwen3_spec

_MAX_PROMPT = 48


class ChaosTokenizer(FakeTokenizer):
    def encode_conversation(self, messages, system=None) -> list[int]:
        self.last_messages = messages
        self.last_system = system
        text = " ".join(str(m.get("content", "")) for m in messages)
        n = max(1, min(_MAX_PROMPT, len(text) // 4))
        return [1] * n


def chaos_tiny_spec() -> ModelSpec:
    return replace(
        tiny_qwen3_spec(),
        tokenizer_factory=lambda local_dir: ChaosTokenizer(),
    )

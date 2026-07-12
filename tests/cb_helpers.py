"""Shared helpers for tests that drive the real CB engine on tiny-Qwen3.

Home for the oracle-vs-batched harness used by test_cb_end_to_end.py and
test_logprobs.py — shared here rather than imported test-module-to-test-module
so renaming one test file can't break another suite.
"""

import torch

from cantollm.engine import ContinuousBatchingEngine, SequentialEngine
from cantollm.engine.batching import BatchingConfig
from cantollm.engine.types import InferenceRequest, SamplingParams
from cantollm.runtime import build_runtime
from tests.tiny_model import tiny_qwen3_spec

GREEDY = SamplingParams.from_temperature_top_p(temperature=0.0, top_p=1.0)

PROMPTS = [
    [11, 12, 13, 14, 15, 16, 17, 18],  # long-ish prefill
    [31, 32],                          # short
    [41, 42, 43, 44, 45],              # medium
]


def build_engines(**config_overrides):
    """(sequential oracle engine, CB engine) — identical weights, einsum vs
    padded attention."""
    cpu = torch.device("cpu")
    seq_runtime = build_runtime(tiny_qwen3_spec(), cpu)
    cb_runtime = build_runtime(tiny_qwen3_spec(), cpu, attention="padded")
    cb_runtime.model.load_state_dict(seq_runtime.model.state_dict())

    config = BatchingConfig(**{
        "max_batch": 3, "max_seq_len": 64, "max_tokens_per_step": 8,
        **config_overrides,
    })
    return SequentialEngine(seq_runtime), ContinuousBatchingEngine.from_runtime(
        cb_runtime, config
    )


def make_request(rid: str, prompt: list[int], max_tokens: int = 6,
                 stop_token_ids: set[int] | None = None) -> InferenceRequest:
    return InferenceRequest(
        request_id=rid,
        prompt_token_ids=list(prompt),
        sampling_params=GREEDY,
        max_tokens=max_tokens,
        stop_token_ids=stop_token_ids or set(),
    )


async def collect(engine, req: InferenceRequest) -> tuple[list[int], str | None]:
    tokens, finish = [], None
    async for evt in engine.submit(req):
        assert evt.error is None, f"unexpected error event: {evt.error}"
        if evt.token_id is not None:
            tokens.append(evt.token_id)
        if evt.finish_reason is not None:
            finish = evt.finish_reason
    return tokens, finish

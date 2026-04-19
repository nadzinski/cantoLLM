"""Shared request-building plumbing for API dialect adapters.

Message normalization (role shapes, system extraction) stays per-dialect —
this helper picks up after that, running tokenization on the executor and
wrapping the result into an `InferenceRequest`.
"""

import asyncio
import uuid
from concurrent.futures import ThreadPoolExecutor

from cantollm.engine.types import InferenceRequest, SamplingParams


def _build_sync(
    messages: list[dict],
    system: str | None,
    sampling_params: SamplingParams,
    max_tokens: int,
    tokenizer,
) -> InferenceRequest:
    prompt_token_ids = tokenizer.encode_conversation(messages, system=system)
    return InferenceRequest(
        request_id=uuid.uuid4().hex,
        prompt_token_ids=prompt_token_ids,
        sampling_params=sampling_params,
        max_tokens=max_tokens,
        stop_token_ids=tokenizer.stop_token_ids,
    )


async def tokenize_and_build_request(
    *,
    messages: list[dict],
    system: str | None,
    sampling_params: SamplingParams,
    max_tokens: int,
    tokenizer,
    executor: ThreadPoolExecutor,
) -> InferenceRequest:
    """Tokenize `messages` on the executor and wrap into an InferenceRequest.

    The HF Rust tokenizer releases the GIL so the thread-pool dispatch is a
    real CPU-parallel win for long prompts, and — once Phase 2 splits the
    engine into its own process — keeps tokenization off the API event loop
    and out of the scheduler's critical path.
    """
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        executor, _build_sync, messages, system, sampling_params, max_tokens, tokenizer,
    )

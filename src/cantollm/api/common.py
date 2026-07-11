"""Shared request-building plumbing for API dialect adapters.

Message normalization (role shapes, system extraction) stays per-dialect —
this helper picks up after that, running tokenization on the executor and
wrapping the result into an `InferenceRequest`.
"""

import asyncio
import uuid
from concurrent.futures import ThreadPoolExecutor

from cantollm.engine.types import InferenceRequest, SamplingParams


class AdmissionError(ValueError):
    """Request rejected at the door, before any engine sees it."""


def check_admission(req: InferenceRequest, max_request_tokens: int | None) -> None:
    """Reject requests that could never fit their engine's per-request cap.

    For a continuous-batching engine the cap is the per-slot KV capacity:
    admitting an over-cap request would hand it a slot it must eventually
    overflow — and a shared-batch failure takes every other request down
    with it. Rejecting here turns that into one client's clear 400.
    """
    if max_request_tokens is None:
        return
    prompt_tokens = len(req.prompt_token_ids)
    total = prompt_tokens + req.max_tokens
    if total > max_request_tokens:
        raise AdmissionError(
            f"prompt ({prompt_tokens} tokens) + max_tokens ({req.max_tokens}) "
            f"= {total} exceeds this model's limit of {max_request_tokens} tokens"
        )


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

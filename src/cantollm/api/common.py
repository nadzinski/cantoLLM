"""Shared request-building plumbing for API dialect adapters.

Message normalization (role shapes, system extraction) stays per-dialect —
this helper picks up after that, running tokenization on the executor and
wrapping the result into an `InferenceRequest`.
"""

import asyncio
import time
import uuid
from collections.abc import AsyncIterator
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from cantollm.engine.types import InferenceRequest, SamplingParams, TokenEvent
from cantollm.lifecycle import RequestTicket
from cantollm.obs import tracing
from cantollm.obs.logging import request_id_var


class AdmissionError(ValueError):
    """Request rejected at the door, before any engine sees it."""


def start_request_span(dialect: str, model: str) -> Any | None:
    """Root span for one inference request; None when tracing is off. The
    router owns it until tracked_events takes over and ends it at stream
    end (see end_request_span for the failure paths)."""
    tracer = tracing.get_tracer()
    if tracer is None:
        return None
    from opentelemetry.trace import SpanKind

    return tracer.start_span(
        f"{dialect} {model}",
        kind=SpanKind.SERVER,
        attributes={
            "model": model,
            "request_id": request_id_var.get() or "",
        },
    )


def end_request_span(span: Any | None, error: str | None = None) -> None:
    if span is None or not span.is_recording():
        return
    if error is not None:
        span.set_attribute("error", error)
    span.end()


def request_observer_for(entry: Any) -> Any | None:
    """The per-request metrics observer for this entry's model, or None
    when metrics are unwired (fake entries, direct engine tests)."""
    handle = getattr(entry, "handle", None)
    factory = getattr(handle, "request_observer_factory", None)
    return factory() if factory is not None else None


async def tracked_events(
    ticket: RequestTicket,
    events: AsyncIterator[TokenEvent],
    observe: Any | None = None,
    span: Any | None = None,
) -> AsyncIterator[TokenEvent]:
    """Wrap an engine event stream so the request's in-flight ticket closes
    exactly once — on exhaustion, on error, or on aclose (the adapter's
    disconnect path). Closing this wrapper also closes the inner stream,
    which is what triggers the engine-side disconnect abort. The optional
    observer gets TTFT at the first token and duration/tokens/reason at
    the end (reason None = client disconnect)."""
    t0 = time.perf_counter()
    tokens = 0
    first_seen = False
    reason: str | None = None
    try:
        async for evt in events:
            if evt.token_id is not None:
                tokens += 1
                if not first_seen:
                    first_seen = True
                    if observe is not None:
                        observe.first_token(time.perf_counter() - t0)
                    if span is not None and span.is_recording():
                        span.add_event("first_token")
            if evt.finish_reason is not None:
                reason = evt.finish_reason
            elif evt.error is not None:
                reason = "error"
            yield evt
    finally:
        ticket.close()
        if observe is not None:
            observe.finished(time.perf_counter() - t0, tokens, reason)
        if span is not None and span.is_recording():
            span.set_attribute("output_tokens", tokens)
            span.set_attribute("finish_reason", reason or "disconnect")
            span.end()
        await events.aclose()


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
    ignore_eos: bool,
    request_id: str,
    priority: int,
) -> InferenceRequest:
    prompt_token_ids = tokenizer.encode_conversation(messages, system=system)
    return InferenceRequest(
        request_id=request_id,
        prompt_token_ids=prompt_token_ids,
        sampling_params=sampling_params,
        max_tokens=max_tokens,
        # An empty stop set means only max_tokens ends generation — the
        # bench harness's fixed-length mode (ignore_eos).
        stop_token_ids=set() if ignore_eos else tokenizer.stop_token_ids,
        priority=priority,
    )


async def tokenize_and_build_request(
    *,
    messages: list[dict],
    system: str | None,
    sampling_params: SamplingParams,
    max_tokens: int,
    tokenizer,
    executor: ThreadPoolExecutor,
    ignore_eos: bool = False,
    priority: int = 0,
    parent_span: Any | None = None,
) -> InferenceRequest:
    """Tokenize `messages` on the executor and wrap into an InferenceRequest.

    The HF Rust tokenizer releases the GIL so the thread-pool dispatch is a
    real CPU-parallel win for long prompts, and — once Phase 2 splits the
    engine into its own process — keeps tokenization off the API event loop
    and out of the scheduler's critical path.
    """
    # Read the contextvar here, not in _build_sync: run_in_executor does not
    # propagate context into the worker thread. Falls back to a fresh id for
    # callers without the middleware (direct engine tests, clients).
    request_id = request_id_var.get() or uuid.uuid4().hex
    tok_span = None
    if parent_span is not None and (tracer := tracing.get_tracer()) is not None:
        from opentelemetry.trace import set_span_in_context

        tok_span = tracer.start_span(
            "tokenize", context=set_span_in_context(parent_span)
        )
    loop = asyncio.get_running_loop()
    try:
        req = await loop.run_in_executor(
            executor, _build_sync, messages, system, sampling_params,
            max_tokens, tokenizer, ignore_eos, request_id, priority,
        )
    finally:
        if tok_span is not None:
            tok_span.end()
    if parent_span is not None:
        # The engine child parents its queue/prefill/decode spans on this.
        req.trace_context = tracing.inject_span_context(parent_span)
    return req

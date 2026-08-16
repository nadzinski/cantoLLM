"""Engine-side request-phase spans (Phase 3.5 chunk 5).

`TraceStepObserver` rides the drive loop the same way StepStatsCollector
does — snapshots around each step, zero scheduler changes — and derives
per-request spans at step granularity:

  queue          add_request -> first appearance in the running set
  prefill chunk  a step that consumed prompt tokens (width + index attrs)
  decode         first decode token (first_token event) -> terminal event

All spans parent under the API-side root via the traceparent carrier on
`InferenceRequest.trace_context`, with explicit wall-clock timestamps
(time.time_ns; same host, coherent enough). Requests without a carrier are
ignored, and the observer is only constructed when tracing is configured,
so the untraced path costs nothing.

Step granularity is deliberate: a chunk span covers its whole step, not
the kernel time inside it. Kernel-level truth lives in profile_step and
the step-duration histograms; these spans answer "where did this request's
wall time go".
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from cantollm.engine.types import InferenceRequest, TokenEvent
from cantollm.obs import tracing


@dataclass
class _Traced:
    ctx: Any                    # extracted parent context (the API root)
    enqueue_ns: int
    queued: bool = True
    position: int = 0           # last seen position (prompt progress)
    prompt_len: int = 0
    chunk_index: int = 0
    decode_span: Any = None
    decode_tokens: int = 0
    attrs: dict = field(default_factory=dict)


class TraceStepObserver:
    """One per drive loop. Not thread-safe; lives on the scheduler thread
    (in-process) or in the engine child's main loop."""

    def __init__(self, tracer: Any):
        self._tracer = tracer
        self._traced: dict[str, _Traced] = {}
        self._t_start_ns = 0
        self._before_positions: dict[str, int] = {}

    @classmethod
    def create(cls) -> "TraceStepObserver | None":
        tracer = tracing.get_tracer()
        return cls(tracer) if tracer is not None else None

    # --- drive-loop hooks ----------------------------------------------

    def on_request(self, req: InferenceRequest) -> None:
        carrier = getattr(req, "trace_context", None)
        if not carrier:
            return
        self._traced[req.request_id] = _Traced(
            ctx=tracing.extract_context(carrier),
            enqueue_ns=time.time_ns(),
            attrs={"request_id": req.request_id},
        )

    def before_step(self, scheduler: Any) -> None:
        self._t_start_ns = time.time_ns()
        self._before_positions = {
            seq.request_id: seq.position for seq in scheduler.active
        }

    def after_step(self, scheduler: Any, events: list[TokenEvent]) -> None:
        t_end_ns = time.time_ns()
        active = {seq.request_id: seq for seq in scheduler.active}

        for rid, t in list(self._traced.items()):
            seq = active.get(rid)
            if seq is None:
                continue
            if t.queued:
                # Promoted during this step: the queue wait ended when the
                # step began.
                t.queued = False
                t.prompt_len = len(seq.prompt_token_ids)
                span = self._tracer.start_span(
                    "queue", context=t.ctx,
                    start_time=t.enqueue_ns, attributes=t.attrs,
                )
                span.end(end_time=self._t_start_ns)
            before = self._before_positions.get(rid, 0)
            consumed = seq.position - before
            if consumed > 0 and before < t.prompt_len:
                width = min(consumed, t.prompt_len - before)
                span = self._tracer.start_span(
                    "prefill chunk", context=t.ctx,
                    start_time=self._t_start_ns,
                    attributes={**t.attrs, "width": width,
                                "index": t.chunk_index},
                )
                span.end(end_time=t_end_ns)
                t.chunk_index += 1
            t.position = seq.position

        for evt in events:
            t = self._traced.get(evt.request_id)
            if t is None:
                continue
            if evt.token_id is not None:
                if t.decode_span is None:
                    t.decode_span = self._tracer.start_span(
                        "decode", context=t.ctx,
                        start_time=self._t_start_ns, attributes=t.attrs,
                    )
                    t.decode_span.add_event("first_token", timestamp=t_end_ns)
                t.decode_tokens += 1
            if evt.finish_reason is not None or evt.error is not None:
                self._finish(evt, t, t_end_ns)

    # --- teardown -------------------------------------------------------

    def _finish(self, evt: TokenEvent, t: _Traced, t_end_ns: int) -> None:
        reason = evt.finish_reason or "error"
        if t.queued:
            # Aborted before ever being scheduled: the queue span is the
            # whole story.
            span = self._tracer.start_span(
                "queue", context=t.ctx,
                start_time=t.enqueue_ns,
                attributes={**t.attrs, "finish_reason": reason},
            )
            span.end(end_time=t_end_ns)
        if t.decode_span is not None:
            t.decode_span.set_attribute("output_tokens", t.decode_tokens)
            t.decode_span.set_attribute("finish_reason", reason)
            t.decode_span.end(end_time=t_end_ns)
        self._traced.pop(evt.request_id, None)

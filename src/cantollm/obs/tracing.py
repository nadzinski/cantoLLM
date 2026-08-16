"""OTel tracing (Phase 3.5 chunk 5): request-phase spans, OTLP to Tempo.

Off unless configured: `get_tracer()` returns None and every call site
guards on that, so the default serve path and the whole test suite pay one
None-check. `canto serve --otlp-endpoint` (or OTEL_EXPORTER_OTLP_ENDPOINT
in the environment) turns it on; the engine child inherits the environment
variable and configures its own provider, so both processes export
directly to the same OTLP/HTTP endpoint. No collector.

Span model (decided in the planning session): one trace per request.
API side: a root span for the request plus a tokenize child; the root's
context crosses the IPC boundary as a W3C traceparent carrier on
`InferenceRequest.trace_context`. Engine side: queue / prefill-chunk /
decode spans, derived at step granularity by the drive loop's
TraceStepObserver (engine/batching/trace.py) with explicit wall-clock
timestamps — same-host processes keep those coherent. 100% sampling; it
is a lab, keep every trace.
"""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

_provider: Any | None = None
_tracer: Any | None = None

_PROPAGATOR = None


def _propagator():
    global _PROPAGATOR
    if _PROPAGATOR is None:
        from opentelemetry.trace.propagation.tracecontext import (
            TraceContextTextMapPropagator,
        )

        _PROPAGATOR = TraceContextTextMapPropagator()
    return _PROPAGATOR


def configure_tracing(
    service_name: str,
    endpoint: str | None = None,
    exporter: Any | None = None,
) -> bool:
    """Install a module-global provider. `endpoint` builds an OTLP/HTTP
    exporter (batched); tests pass an explicit `exporter` instead (wrapped
    in a SimpleSpanProcessor for determinism). Returns True if configured."""
    global _provider, _tracer
    if endpoint is None and exporter is None:
        return False
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import (
        BatchSpanProcessor,
        SimpleSpanProcessor,
    )

    provider = TracerProvider(
        resource=Resource.create({"service.name": service_name})
    )
    if exporter is not None:
        provider.add_span_processor(SimpleSpanProcessor(exporter))
    else:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
            OTLPSpanExporter,
        )

        provider.add_span_processor(
            BatchSpanProcessor(OTLPSpanExporter(
                endpoint=endpoint.rstrip("/") + "/v1/traces"
            ))
        )
        logger.info("tracing on: OTLP/HTTP -> %s (service %s)",
                    endpoint, service_name)
    _provider = provider
    _tracer = provider.get_tracer("cantollm")
    return True


def configure_from_env(service_name: str) -> bool:
    """The engine child's path: the parent exports the endpoint via the
    inherited environment."""
    return configure_tracing(
        service_name, endpoint=os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
    )


def get_tracer() -> Any | None:
    return _tracer


def shutdown_tracing() -> None:
    """Flush and drop the provider (process exit; tests)."""
    global _provider, _tracer
    if _provider is not None:
        _provider.shutdown()
    _provider = None
    _tracer = None


def inject_span_context(span: Any) -> dict[str, str]:
    """W3C traceparent carrier for `span`, for the IPC boundary."""
    from opentelemetry.trace import set_span_in_context

    carrier: dict[str, str] = {}
    _propagator().inject(carrier, context=set_span_in_context(span))
    return carrier


def extract_context(carrier: dict[str, str]) -> Any:
    """Parent context from a traceparent carrier (engine side)."""
    return _propagator().extract(carrier)

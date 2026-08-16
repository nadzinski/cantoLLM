"""FastAPI application factory.

Mounts three routers — common (`/health`, `/v1/models`), Anthropic
(`/v1/messages`), and OpenAI (`/v1/chat/completions`) — against a single
`EngineRegistry` and a shared tokenizer executor.
"""

import asyncio
import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response

from cantollm.api.admin_router import build_admin_router
from cantollm.api.anthropic_router import build_anthropic_router
from cantollm.api.common_router import build_common_router
from cantollm.api.debug_router import build_debug_router
from cantollm.api.errors import install_error_handlers
from cantollm.api.openai_router import build_openai_router
from cantollm.obs.logging import request_id_var
from cantollm.obs.metrics import CONTENT_TYPE as METRICS_CONTENT_TYPE
from cantollm.obs.metrics import Metrics
from cantollm.registry import EngineRegistry


def _default_tokenizer_workers() -> int:
    return min(8, os.cpu_count() or 4)


def create_app(
    registry: EngineRegistry, *, tokenizer_workers: int | None = None
) -> FastAPI:
    # Rust tokenizer releases the GIL, so threads give real parallelism; keep
    # the pool well under core count so it doesn't starve the event loop (and,
    # post Phase 2 split, the IPC bridge).
    workers = tokenizer_workers if tokenizer_workers is not None else _default_tokenizer_workers()
    tokenizer_executor = ThreadPoolExecutor(
        max_workers=workers, thread_name_prefix="tokenize"
    )

    metrics = Metrics(registry)
    metrics.wire_handles()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Non-blocking: supervisors build engines in the background while
        # uvicorn serves. /ready reports 503-with-progress until warm;
        # nothing heavy runs pre-yield, so the socket answers immediately.
        registry.launch_all()
        lag_task = asyncio.get_running_loop().create_task(
            metrics.run_loop_lag_sampler(), name="event-loop-lag"
        )
        try:
            yield
        finally:
            lag_task.cancel()
            try:
                await lag_task
            except asyncio.CancelledError:
                pass
            await registry.stop_all()
            tokenizer_executor.shutdown(wait=True, cancel_futures=True)

    app = FastAPI(title="CantoLLM", lifespan=lifespan)
    app.state.metrics = metrics

    @app.middleware("http")
    async def http_metrics_middleware(request: Request, call_next):
        t0 = time.perf_counter()
        response = await call_next(request)
        # Route template (never the raw path: cardinality); the router
        # mutates this scope on match, so it is visible post-dispatch.
        route = getattr(request.scope.get("route"), "path", "unmatched")
        metrics.observe_http(
            route, request.method, response.status_code,
            time.perf_counter() - t0,
        )
        return response

    @app.middleware("http")
    async def request_id_middleware(request: Request, call_next):
        # Honor an inbound X-Request-ID, else mint one. The contextvar makes
        # the id visible to JSON log records and to request tokenization
        # (common.py reads it into InferenceRequest.request_id); streaming
        # bodies inherit it because the downstream task copies this context.
        rid = request.headers.get("x-request-id") or uuid.uuid4().hex
        token = request_id_var.set(rid)
        try:
            response = await call_next(request)
            response.headers["x-request-id"] = rid
            return response
        finally:
            request_id_var.reset(token)

    install_error_handlers(app)

    @app.get("/metrics")
    async def metrics_route():
        return Response(content=metrics.render(), media_type=METRICS_CONTENT_TYPE)

    app.include_router(build_common_router(registry))
    app.include_router(build_anthropic_router(registry, tokenizer_executor))
    app.include_router(build_openai_router(registry, tokenizer_executor))
    app.include_router(build_debug_router(registry))
    app.include_router(build_admin_router(registry))
    return app

"""FastAPI application factory.

Mounts three routers — common (`/health`, `/v1/models`), Anthropic
(`/v1/messages`), and OpenAI (`/v1/chat/completions`) — against a single
`EngineRegistry` and a shared tokenizer executor.
"""

import os
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

from fastapi import FastAPI

from cantollm.api.anthropic_router import build_anthropic_router
from cantollm.api.common_router import build_common_router
from cantollm.api.openai_router import build_openai_router
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

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await registry.start_all()
        try:
            yield
        finally:
            await registry.shutdown_all()
            tokenizer_executor.shutdown(wait=True, cancel_futures=True)

    app = FastAPI(title="CantoLLM", lifespan=lifespan)
    app.include_router(build_common_router(registry))
    app.include_router(build_anthropic_router(registry, tokenizer_executor))
    app.include_router(build_openai_router(registry, tokenizer_executor))
    return app

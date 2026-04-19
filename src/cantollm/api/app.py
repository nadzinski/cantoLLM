"""FastAPI application for the CantoLLM Messages API."""

import asyncio
import os
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse

from cantollm.api.anthropic_adapter import render_message, render_sse
from cantollm.api.anthropic_types import (
    MessagesRequest,
    ModelInfo,
    ModelListResponse,
)
from cantollm.engine.types import InferenceRequest, SamplingParams
from cantollm.registry import EngineRegistry


def _build_inference_request(body: MessagesRequest, tokenizer) -> InferenceRequest:
    prompt_token_ids = tokenizer.encode_conversation(
        [m.model_dump() for m in body.messages],
        system=body.system,
    )
    return InferenceRequest(
        request_id=uuid.uuid4().hex,
        prompt_token_ids=prompt_token_ids,
        sampling_params=SamplingParams(temperature=body.temperature, top_p=body.top_p),
        max_tokens=body.max_tokens,
        stop_token_ids=tokenizer.stop_token_ids,
    )


async def _build_inference_request_async(
    body: MessagesRequest, tokenizer, executor: ThreadPoolExecutor
) -> InferenceRequest:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        executor, _build_inference_request, body, tokenizer
    )


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

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/v1/models", response_model=ModelListResponse)
    async def list_models():
        names = registry.names()
        data = [
            ModelInfo(
                id=name,
                display_name=name,
                created_at=datetime.fromtimestamp(
                    entry.registered_at, tz=timezone.utc
                ).isoformat().replace("+00:00", "Z"),
            )
            for name, entry in registry.items()
        ]
        return ModelListResponse(
            data=data,
            has_more=False,
            first_id=names[0] if names else None,
            last_id=names[-1] if names else None,
        )

    @app.post("/v1/messages")
    async def messages(body: MessagesRequest):
        try:
            entry = registry.get(body.model)
        except KeyError:
            raise HTTPException(
                status_code=404,
                detail=f"Model '{body.model}' is not registered. Available: {registry.names()}",
            )

        tokenizer = entry.runtime.tokenizer
        try:
            req = await _build_inference_request_async(
                body, tokenizer, tokenizer_executor
            )
        except (ValueError, TypeError, KeyError) as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        events = entry.engine.submit(req)
        input_tokens = len(req.prompt_token_ids)

        if body.stream:
            return StreamingResponse(
                render_sse(events, tokenizer, body.model, input_tokens),
                media_type="text/event-stream",
            )
        return await render_message(events, tokenizer, body.model, input_tokens)

    return app

"""FastAPI application for the CantoLLM Messages API."""

import uuid
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


def create_app(registry: EngineRegistry) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await registry.start_all()
        try:
            yield
        finally:
            await registry.shutdown_all()

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
        req = _build_inference_request(body, tokenizer)
        events = entry.engine.submit(req)
        input_tokens = len(req.prompt_token_ids)

        if body.stream:
            return StreamingResponse(
                render_sse(events, tokenizer, body.model, input_tokens),
                media_type="text/event-stream",
            )
        return await render_message(events, tokenizer, body.model, input_tokens)

    return app

"""FastAPI application for the CantoLLM Messages API."""

import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import StreamingResponse

from cantollm.api.anthropic_adapter import render_message, render_sse
from cantollm.api.anthropic_types import MessagesRequest
from cantollm.engine.engine import InferenceEngine
from cantollm.engine.types import InferenceRequest, SamplingParams


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


def create_app(engine: InferenceEngine, tokenizer, model_name: str) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await engine.start()
        try:
            yield
        finally:
            await engine.shutdown()

    app = FastAPI(title="CantoLLM", lifespan=lifespan)

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.post("/v1/messages")
    async def messages(body: MessagesRequest):
        req = _build_inference_request(body, tokenizer)
        events = engine.submit(req)
        input_tokens = len(req.prompt_token_ids)

        if body.stream:
            return StreamingResponse(
                render_sse(events, tokenizer, model_name, input_tokens),
                media_type="text/event-stream",
            )
        return await render_message(events, tokenizer, model_name, input_tokens)

    return app

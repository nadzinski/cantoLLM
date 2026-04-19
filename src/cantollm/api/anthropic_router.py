"""Router for the Anthropic-compatible Messages API (`POST /v1/messages`)."""

from concurrent.futures import ThreadPoolExecutor

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from cantollm.api.anthropic_adapter import render_message, render_sse
from cantollm.api.anthropic_types import MessagesRequest
from cantollm.api.common import tokenize_and_build_request
from cantollm.engine.types import SamplingParams
from cantollm.registry import EngineRegistry


def build_anthropic_router(
    registry: EngineRegistry,
    tokenizer_executor: ThreadPoolExecutor,
) -> APIRouter:
    router = APIRouter()

    @router.post("/v1/messages")
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
            req = await tokenize_and_build_request(
                messages=[m.model_dump() for m in body.messages],
                system=body.system,
                sampling_params=SamplingParams.from_temperature_top_p(
                    body.temperature, body.top_p,
                ),
                max_tokens=body.max_tokens,
                tokenizer=tokenizer,
                executor=tokenizer_executor,
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

    return router

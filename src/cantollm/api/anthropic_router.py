"""Router for the Anthropic-compatible Messages API (`POST /v1/messages`)."""

from concurrent.futures import ThreadPoolExecutor

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from cantollm.api.anthropic_adapter import render_message, render_sse
from cantollm.api.anthropic_types import MessagesRequest
from cantollm.api.common import (
    check_admission,
    request_observer_for,
    tokenize_and_build_request,
    tracked_events,
)
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

        # Raises NotReadyError (-> 503) while warming/draining/crashed.
        # Capture the engine now: a supervisor swap mid-request must not mix
        # generations. The ticket claims the in-flight counter in the same
        # handler tick, so a drain beginning on any later await still counts
        # this request; tracked_events (or the except below) closes it.
        engine = entry.ensure_ready()
        ticket = await entry.begin_request()
        try:
            if body.ignore_eos and body.stop_sequences:
                raise HTTPException(
                    status_code=400,
                    detail="ignore_eos and stop_sequences are mutually exclusive: "
                    "ignore_eos requests fixed-length output, stop sequences end it early.",
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
                    ignore_eos=body.ignore_eos,
                )
                check_admission(req, entry.max_request_tokens)
            except (ValueError, TypeError, KeyError) as exc:
                raise HTTPException(status_code=400, detail=str(exc))

            events = tracked_events(
                ticket, engine.submit(req),
                observe=request_observer_for(entry),
            )
            input_tokens = len(req.prompt_token_ids)

            if body.stream:
                return StreamingResponse(
                    render_sse(events, tokenizer, body.model, input_tokens,
                               stop_sequences=body.stop_sequences),
                    media_type="text/event-stream",
                )
            return await render_message(events, tokenizer, body.model, input_tokens,
                                        stop_sequences=body.stop_sequences)
        except BaseException:
            ticket.close()
            raise

    return router

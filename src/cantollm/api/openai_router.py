"""Router for the OpenAI-compatible Chat Completions API
(`POST /v1/chat/completions`).

Message normalization lives here: OpenAI encodes `system` / `developer`
context as role messages in the conversation, while our tokenizer expects a
top-level `system: str` plus an Anthropic-style `user` / `assistant`
sequence. This router folds the OpenAI shape into that one before handing
off to the shared request builder.
"""

import time
import uuid

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from cantollm.api.common import tokenize_and_build_request
from cantollm.api.openai_adapter import (
    render_chat_completion,
    render_chat_completion_sse,
)
from cantollm.api.openai_types import (
    ChatCompletionRequest,
    OpenAIAssistantMessage,
    OpenAIContentPartText,
    OpenAIMessage,
    OpenAISystemMessage,
    OpenAIUserMessage,
    StreamOptions,
)
from cantollm.engine.types import SamplingParams
from cantollm.registry import EngineRegistry

DEFAULT_MAX_TOKENS = 1024


def _flatten_text(content: str | list[OpenAIContentPartText] | None) -> str:
    """Collapse an OpenAI content field to a single string.

    Image / audio / file parts aren't in our Pydantic schema — they'd have
    been rejected at validation — so by the time we reach this helper the
    list-of-parts case contains only text parts.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    return "\n".join(part.text for part in content)


def _normalize_openai_messages(
    messages: list[OpenAIMessage],
) -> tuple[list[dict], str | None]:
    """Extract leading system/developer messages into the `system` kwarg and
    flatten the remaining user/assistant messages into the tokenizer's shape.

    Rules:
    - Any leading run of `system` / `developer` messages becomes the
      `system` string (joined with blank lines). `developer` is treated as
      `system` — per the spec, post-o1 models use `developer` to replace
      the old `system` role.
    - `system` / `developer` messages appearing after a user/assistant turn
      are a 400 (the tokenizer can't express a mid-conversation system
      switch; nor does the ChatML template we emit).
    - Assistant messages with no content (tool-call continuations) are a
      400 — we don't support tool semantics yet.
    """
    system_parts: list[str] = []
    conversation: list[dict] = []
    seen_non_system = False

    for msg in messages:
        if isinstance(msg, OpenAISystemMessage):
            if seen_non_system:
                raise HTTPException(
                    status_code=400,
                    detail="system/developer messages must appear before any user/assistant message.",
                )
            system_parts.append(_flatten_text(msg.content))
            continue

        seen_non_system = True
        if isinstance(msg, OpenAIUserMessage):
            conversation.append({"role": "user", "content": _flatten_text(msg.content)})
        elif isinstance(msg, OpenAIAssistantMessage):
            text = _flatten_text(msg.content)
            if not text:
                raise HTTPException(
                    status_code=400,
                    detail="Assistant messages must carry text content; tool-call continuations are not supported.",
                )
            conversation.append({"role": "assistant", "content": text})

    system = "\n\n".join(p for p in system_parts if p) or None
    return conversation, system


def build_openai_router(
    registry: EngineRegistry,
    tokenizer_executor,
) -> APIRouter:
    router = APIRouter()

    @router.post("/v1/chat/completions")
    async def chat_completions(body: ChatCompletionRequest):
        try:
            entry = registry.get(body.model)
        except KeyError:
            raise HTTPException(
                status_code=404,
                detail=f"Model '{body.model}' is not registered. Available: {registry.names()}",
            )

        messages, system = _normalize_openai_messages(body.messages)
        if not messages:
            raise HTTPException(
                status_code=400,
                detail="At least one user or assistant message is required.",
            )

        max_tokens = (
            body.max_completion_tokens
            or body.max_tokens
            or DEFAULT_MAX_TOKENS
        )

        tokenizer = entry.runtime.tokenizer
        try:
            req = await tokenize_and_build_request(
                messages=messages,
                system=system,
                sampling_params=SamplingParams.from_temperature_top_p(
                    body.temperature, body.top_p,
                ),
                max_tokens=max_tokens,
                tokenizer=tokenizer,
                executor=tokenizer_executor,
            )
        except (ValueError, TypeError, KeyError) as exc:
            raise HTTPException(status_code=400, detail=str(exc))

        events = entry.engine.submit(req)
        input_tokens = len(req.prompt_token_ids)
        completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        created = int(time.time())

        if body.stream:
            include_usage = (body.stream_options or StreamOptions()).include_usage
            return StreamingResponse(
                render_chat_completion_sse(
                    events, tokenizer, body.model, input_tokens,
                    completion_id, created, include_usage,
                ),
                media_type="text/event-stream",
            )
        return await render_chat_completion(
            events, tokenizer, body.model, input_tokens, completion_id, created,
        )

    return router

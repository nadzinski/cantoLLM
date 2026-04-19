"""Wire schema for the OpenAI-compatible Chat Completions API.

Covers the subset we actually support for Phase 1b: text-only messages,
`system` / `developer` / `user` / `assistant` roles, streaming with optional
usage, and the DeepSeek-R1 `reasoning_content` convention for the model's
thinking phase. Unsupported spec fields are rejected via ``extra="forbid"``
at the request boundary (see `_reject_unsupported_fields` in the router) so
we fail loudly on typos rather than silently dropping intent.
"""

from typing import Annotated, Literal, Union

from pydantic import BaseModel, Field

# ── Request: messages ────────────────────────────────────────────────


class OpenAIContentPartText(BaseModel):
    type: Literal["text"]
    text: str
    model_config = {"extra": "forbid"}


class OpenAISystemMessage(BaseModel):
    # 'developer' is the post-o1 replacement; we accept both and fold them.
    role: Literal["system", "developer"]
    content: str | list[OpenAIContentPartText]
    name: str | None = None
    model_config = {"extra": "forbid"}


class OpenAIUserMessage(BaseModel):
    role: Literal["user"]
    # Image / audio / file parts aren't in our schema for Phase 1b. If a
    # client sends them, Pydantic rejects at the discriminator boundary.
    content: str | list[OpenAIContentPartText]
    name: str | None = None
    model_config = {"extra": "forbid"}


class OpenAIAssistantMessage(BaseModel):
    role: Literal["assistant"]
    content: str | list[OpenAIContentPartText] | None = None
    name: str | None = None
    model_config = {"extra": "forbid"}


OpenAIMessage = Annotated[
    Union[OpenAISystemMessage, OpenAIUserMessage, OpenAIAssistantMessage],
    Field(discriminator="role"),
]


# ── Request: envelope ────────────────────────────────────────────────


class StreamOptions(BaseModel):
    include_usage: bool = False
    model_config = {"extra": "forbid"}


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[OpenAIMessage] = Field(min_length=1)
    temperature: float = 1.0
    top_p: float = 1.0
    stream: bool = False
    stream_options: StreamOptions | None = None
    # OpenAI accepts both; max_completion_tokens is preferred post-o1.
    max_tokens: int | None = Field(default=None, gt=0)
    max_completion_tokens: int | None = Field(default=None, gt=0)
    # Accepted but currently ignored — stop-string support lands in Phase 2.
    stop: str | list[str] | None = None
    # `user` is a free-form id OpenAI uses for abuse tracking; harmless pass-through.
    user: str | None = None
    # Everything else (tools, tool_choice, n, logprobs, response_format,
    # modalities, audio, seed, parallel_tool_calls, presence_penalty,
    # frequency_penalty, etc.) is rejected by extra="forbid".
    model_config = {"extra": "forbid"}


# ── Response: finish reasons ─────────────────────────────────────────

# OpenAI's full set is "stop" | "length" | "tool_calls" | "content_filter" |
# "function_call"; we emit only "stop" | "length" | None today.
FinishReason = Literal["stop", "length", "tool_calls", "content_filter", "function_call"]


# ── Response: usage ──────────────────────────────────────────────────


class CompletionTokensDetails(BaseModel):
    reasoning_tokens: int = 0
    accepted_prediction_tokens: int = 0
    rejected_prediction_tokens: int = 0


class CompletionUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    completion_tokens_details: CompletionTokensDetails | None = None


# ── Response: non-streaming ──────────────────────────────────────────


class ChatCompletionMessage(BaseModel):
    role: Literal["assistant"] = "assistant"
    content: str | None
    # DeepSeek-R1 convention — the OpenAI SDK ignores unknown fields, so
    # clients that care can read it, and strict clients aren't broken.
    reasoning_content: str | None = None


class ChatCompletionChoice(BaseModel):
    index: int = 0
    message: ChatCompletionMessage
    finish_reason: FinishReason | None
    logprobs: None = None


class ChatCompletion(BaseModel):
    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: list[ChatCompletionChoice]
    usage: CompletionUsage


# ── Response: streaming ──────────────────────────────────────────────


class ChatCompletionDelta(BaseModel):
    role: Literal["assistant"] | None = None
    content: str | None = None
    reasoning_content: str | None = None


class ChatCompletionChunkChoice(BaseModel):
    index: int = 0
    delta: ChatCompletionDelta
    finish_reason: FinishReason | None = None
    logprobs: None = None


class ChatCompletionChunk(BaseModel):
    id: str
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int
    model: str
    choices: list[ChatCompletionChunkChoice]
    # Only populated on the final usage-only chunk when
    # `stream_options.include_usage=true`; omitted otherwise.
    usage: CompletionUsage | None = None


# ── Errors ──────────────────────────────────────────────────────────


class OpenAIError(BaseModel):
    message: str
    type: str = "api_error"
    code: str | None = None
    param: str | None = None


class OpenAIErrorEnvelope(BaseModel):
    error: OpenAIError

"""Wire schema for the Anthropic-compatible Messages API.

Requests, responses, and SSE events are all Pydantic models. FastAPI uses them
for automatic validation, OpenAPI generation, and response serialization. The
adapter constructs typed SSE events so mis-shaped payloads fail at construction.
"""

from typing import Annotated, Literal, Union

from pydantic import BaseModel, Field, field_validator

# ── Request ────────────────────────────────────────────────────────────


class ContentBlockInput(BaseModel):
    type: Literal["text"]
    text: str


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str | list[ContentBlockInput]

    @field_validator("content", mode="after")
    @classmethod
    def _flatten_content(cls, v):
        if isinstance(v, list):
            return "\n".join(b.text for b in v)
        return v


class MessagesRequest(BaseModel):
    model: str
    max_tokens: int = Field(gt=0)
    messages: list[Message] = Field(min_length=1)
    system: str | None = None
    temperature: float = 0.7
    top_p: float = 0.9
    stream: bool = False


# ── Response: content blocks ──────────────────────────────────────────


class TextBlock(BaseModel):
    type: Literal["text"] = "text"
    text: str


class ThinkingBlock(BaseModel):
    type: Literal["thinking"] = "thinking"
    thinking: str


ContentBlock = Annotated[
    Union[TextBlock, ThinkingBlock],
    Field(discriminator="type"),
]


class Usage(BaseModel):
    input_tokens: int
    output_tokens: int


StopReason = Literal["end_turn", "max_tokens", "stop_sequence"]


class MessageResponse(BaseModel):
    id: str
    type: Literal["message"] = "message"
    role: Literal["assistant"] = "assistant"
    content: list[ContentBlock]
    model: str
    stop_reason: StopReason | None
    stop_sequence: str | None = None
    usage: Usage


# ── SSE events ────────────────────────────────────────────────────────


class MessageStartSnapshot(BaseModel):
    """The `message` field of a message_start event — a fresh, empty message."""

    id: str
    type: Literal["message"] = "message"
    role: Literal["assistant"] = "assistant"
    content: list[ContentBlock] = Field(default_factory=list)
    model: str
    stop_reason: StopReason | None = None
    stop_sequence: str | None = None
    usage: Usage


class MessageStartEvent(BaseModel):
    type: Literal["message_start"] = "message_start"
    message: MessageStartSnapshot


class ContentBlockStartEvent(BaseModel):
    type: Literal["content_block_start"] = "content_block_start"
    index: int
    content_block: ContentBlock


class TextDelta(BaseModel):
    type: Literal["text_delta"] = "text_delta"
    text: str


class ThinkingDelta(BaseModel):
    type: Literal["thinking_delta"] = "thinking_delta"
    thinking: str


DeltaBlock = Annotated[
    Union[TextDelta, ThinkingDelta],
    Field(discriminator="type"),
]


class ContentBlockDeltaEvent(BaseModel):
    type: Literal["content_block_delta"] = "content_block_delta"
    index: int
    delta: DeltaBlock


class ContentBlockStopEvent(BaseModel):
    type: Literal["content_block_stop"] = "content_block_stop"
    index: int


class MessageDeltaBody(BaseModel):
    stop_reason: StopReason | None
    stop_sequence: str | None = None


class StreamUsage(BaseModel):
    output_tokens: int
    thinking_tokens: int
    text_tokens: int


class MessageDeltaEvent(BaseModel):
    type: Literal["message_delta"] = "message_delta"
    delta: MessageDeltaBody
    usage: StreamUsage


class MessageStopEvent(BaseModel):
    type: Literal["message_stop"] = "message_stop"


SSEEvent = Annotated[
    Union[
        MessageStartEvent,
        ContentBlockStartEvent,
        ContentBlockDeltaEvent,
        ContentBlockStopEvent,
        MessageDeltaEvent,
        MessageStopEvent,
    ],
    Field(discriminator="type"),
]


def sse(evt: BaseModel) -> str:
    """Format a typed SSE event as a `event:/data:` pair."""
    return f"event: {evt.type}\ndata: {evt.model_dump_json()}\n\n"

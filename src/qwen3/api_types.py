"""Request/response types for the Anthropic-compatible Messages API."""

import json
from dataclasses import dataclass, field


@dataclass
class MessagesRequest:
    """Parsed and validated Messages API request."""

    model: str
    max_tokens: int
    messages: list[dict]
    system: str | None = None
    temperature: float = 0.7
    top_p: float = 0.9
    stream: bool = False

    @classmethod
    def from_dict(cls, body: dict) -> "MessagesRequest":
        """Parse and validate a request body dict.

        Raises ValueError with a human-readable message on validation failure.
        """
        # Required fields
        for field_name in ("model", "messages", "max_tokens"):
            if field_name not in body:
                raise ValueError(f"missing required field: {field_name}")

        messages = body["messages"]
        if not isinstance(messages, list) or not messages:
            raise ValueError("messages must be a non-empty array")

        max_tokens = body["max_tokens"]
        if not isinstance(max_tokens, int) or max_tokens < 1:
            raise ValueError("max_tokens must be a positive integer")

        # Normalize content: string -> keep as-is, content block array -> extract text
        normalized = []
        for msg in messages:
            if "role" not in msg or "content" not in msg:
                raise ValueError("each message must have 'role' and 'content'")
            content = msg["content"]
            if isinstance(content, list):
                # Extract text from content blocks
                text_parts = []
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        text_parts.append(block["text"])
                content = "\n".join(text_parts)
            normalized.append({"role": msg["role"], "content": content})

        return cls(
            model=body["model"],
            max_tokens=max_tokens,
            messages=normalized,
            system=body.get("system"),
            temperature=body.get("temperature", 0.7),
            top_p=body.get("top_p", 0.9),
            stream=body.get("stream", False),
        )


def make_message_response(
    msg_id: str,
    content_blocks: list[dict],
    model: str,
    stop_reason: str,
    input_tokens: int,
    output_tokens: int,
) -> dict:
    """Build an Anthropic-format Message response dict."""
    return {
        "id": msg_id,
        "type": "message",
        "role": "assistant",
        "content": content_blocks,
        "model": model,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        },
    }


def make_sse_event(event_type: str, data: dict) -> str:
    """Format a server-sent event string."""
    return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"

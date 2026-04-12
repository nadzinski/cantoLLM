"""Stream event types for the generation pipeline.

Uses a Rust-style enum pattern with pattern matching:

    match event:
        case ThinkingStartEvent():
            ...
        case ThinkingEndEvent():
            ...
        case TextChunk(text=t):
            ...
"""

from dataclasses import dataclass


class ThinkingStartEvent:
    """Emitted when model enters thinking mode."""

    pass


class ThinkingEndEvent:
    """Emitted when model exits thinking mode (or stream ends while thinking)."""

    pass


@dataclass
class TextChunk:
    """A piece of decoded text, suitable for streaming display."""

    text: str


# Type alias for pattern matching
StreamEvent = ThinkingStartEvent | ThinkingEndEvent | TextChunk

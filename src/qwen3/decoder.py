"""Token-to-event decoder."""

from collections.abc import Iterator

from qwen3.stream_events import StreamEvent, TextChunk, ThinkingEndEvent, ThinkingStartEvent


class StreamingDecoder:
    """Converts a token stream to a stream of structured events.

    Handles:
    - Thinking start/end token detection
    - Incremental text decoding (including multi-byte characters like emoji)
    - Synthetic ThinkingEndEvent on unexpected stream termination

    Usage:
        decoder = StreamingDecoder(tokenizer)
        for event in decoder.wrap(token_stream):
            match event:
                case ThinkingStartEvent(): ...
                case ThinkingEndEvent(): ...
                case TextChunk(text=t): ...
    """

    def __init__(self, tokenizer):
        """Initialize decoder with a tokenizer.

        Args:
            tokenizer: Qwen3Tokenizer instance (needs _special_to_id and
                       incremental_decoder())
        """
        self._tokenizer = tokenizer
        self._thinking_start_id = tokenizer._special_to_id.get("<think>")
        self._thinking_end_id = tokenizer._special_to_id.get("</think>")

    def wrap(self, token_stream: Iterator[int]) -> Iterator[StreamEvent]:
        """Convert a token stream to an event stream.

        Args:
            token_stream: Iterator yielding token IDs

        Yields:
            StreamEvent instances (ThinkingStartEvent, ThinkingEndEvent, TextChunk)
        """
        in_thinking = False
        incremental = self._tokenizer.incremental_decoder()

        for token_id in token_stream:
            # Check for thinking control tokens
            if token_id == self._thinking_start_id:
                in_thinking = True
                yield ThinkingStartEvent()
                continue

            if token_id == self._thinking_end_id:
                in_thinking = False
                yield ThinkingEndEvent()
                continue

            # Regular token - decode and emit text
            text = incremental.add(token_id)
            if text:
                yield TextChunk(text)

        # Finalize: handle unexpected end of stream
        if in_thinking:
            yield ThinkingEndEvent()  # Synthetic close

        remaining = incremental.flush()
        if remaining:
            yield TextChunk(remaining)

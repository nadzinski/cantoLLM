"""Token-to-event decoder."""

from collections.abc import Iterable, Iterator

from cantollm.stream_events import StreamEvent, TextChunk, ThinkingEndEvent, ThinkingStartEvent


class StreamingDecoder:
    """Converts a token stream to structured events.

    Stateful: call `process(token_id)` per token (yields zero or more events),
    then `flush()` at end-of-stream. `wrap(iterator)` is a convenience for the
    sync case; async callers can drive `process` themselves.
    """

    def __init__(self, tokenizer):
        self._tokenizer = tokenizer
        self._thinking_start_id = tokenizer.thinking_start_id
        self._thinking_end_id = tokenizer.thinking_end_id
        self._incremental = tokenizer.incremental_decoder()
        self._in_thinking = False

    def process(self, token_id: int) -> Iterable[StreamEvent]:
        if token_id == self._thinking_start_id:
            self._in_thinking = True
            yield ThinkingStartEvent()
            return
        if token_id == self._thinking_end_id:
            self._in_thinking = False
            yield ThinkingEndEvent()
            return
        text = self._incremental.add(token_id)
        if text:
            yield TextChunk(text)

    def flush(self) -> Iterable[StreamEvent]:
        if self._in_thinking:
            self._in_thinking = False
            yield ThinkingEndEvent()
        remaining = self._incremental.flush()
        if remaining:
            yield TextChunk(remaining)

    def wrap(self, token_stream: Iterator[int]) -> Iterator[StreamEvent]:
        for token_id in token_stream:
            yield from self.process(token_id)
        yield from self.flush()

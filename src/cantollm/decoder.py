"""Token-to-event decoder."""

from collections.abc import Iterable, Iterator

from cantollm.stream_events import StreamEvent, TextChunk, ThinkingEndEvent, ThinkingStartEvent


class StopStringWatcher:
    """Scans a text stream for stop strings, holding back exactly enough
    text that no partial match ever reaches the client.

    Stop strings are a text-level concept — they can span token boundaries
    and start mid-token — so this lives at the decoded-text layer, not in
    the engine. `feed(text)` returns the prefix that is now provably safe
    to emit: the held-back suffix is the longest tail of the buffer that
    could still grow into a match (the longest suffix that is a prefix of
    some stop string). On a match, `matched` is set, the text *before* the
    match is returned, and the stop string itself (plus anything after it)
    is never emitted. `flush()` releases the held tail when the stream ends
    without a match.
    """

    def __init__(self, stop_strings: list[str]):
        self._stops = [s for s in stop_strings if s]
        self._buffer = ""
        self.matched: str | None = None

    def feed(self, text: str) -> str:
        if self.matched is not None:
            return ""
        if not self._stops:
            return text
        self._buffer += text

        earliest: tuple[int, str] | None = None
        for stop in self._stops:
            idx = self._buffer.find(stop)
            if idx != -1 and (earliest is None or idx < earliest[0]):
                earliest = (idx, stop)
        if earliest is not None:
            idx, stop = earliest
            self.matched = stop
            released, self._buffer = self._buffer[:idx], ""
            return released

        hold = 0
        for stop in self._stops:
            longest = min(len(stop) - 1, len(self._buffer))
            for k in range(longest, hold, -1):
                if self._buffer.endswith(stop[:k]):
                    hold = k
                    break
        cut = len(self._buffer) - hold
        released, self._buffer = self._buffer[:cut], self._buffer[cut:]
        return released

    def flush(self) -> str:
        released, self._buffer = self._buffer, ""
        return released


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
            yield from self._release_held()
            self._in_thinking = True
            yield ThinkingStartEvent()
            return
        if token_id == self._thinking_end_id:
            yield from self._release_held()
            self._in_thinking = False
            yield ThinkingEndEvent()
            return
        text = self._incremental.add(token_id)
        if text:
            yield TextChunk(text)

    def _release_held(self) -> Iterable[StreamEvent]:
        """Flush and reset the incremental decoder at a phase boundary.

        A marker token separates two phases in the real token stream, so any
        UTF-8 bytes the incremental decoder is still holding belong to the
        *outgoing* phase — emit them now (as a TextChunk the caller tags with
        the phase it was in) instead of letting the next phase's first token
        release them, which would leak thinking bytes into visible text.
        """
        remaining = self._incremental.flush()
        self._incremental.reset()
        if remaining:
            yield TextChunk(remaining)

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

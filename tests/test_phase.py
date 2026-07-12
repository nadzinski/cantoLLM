"""phase_tagged_events lifecycle: it must close its source generator.

The renderers break out of phase_tagged_events on the terminal event; the
engine's submit() generator is then suspended with its cleanup (KV-slot
release on the batched engine) pending. phase_tagged_events closes it
deterministically in a finally instead of leaving it for GC.

Each test keeps a strong reference to the source generator AND snapshots the
close-flag *inside* the running loop, before `asyncio.run` teardown finalizes
lingering async generators — otherwise loop shutdown would close the source
for us and hide whether phase_tagged_events did it deterministically.
"""

import asyncio

from cantollm.api.phase import DecodeState, phase_tagged_events
from cantollm.engine.types import TokenEvent
from tests.fakes import FakeTokenizer


def test_source_generator_closed_after_finish():
    closed = []

    async def source():
        try:
            yield TokenEvent(token_id=104, request_id="r")  # 'h'
            yield TokenEvent(finish_reason="end_turn", request_id="r")
            # A real engine puts its terminator after this; the consumer breaks
            # on the finish event, so this must never be reached.
            yield TokenEvent(token_id=105, request_id="r")
        finally:
            closed.append(True)

    async def main():
        src = source()  # strong ref: not collectable via GC
        state = DecodeState()
        [evt async for evt in phase_tagged_events(src, FakeTokenizer(), state)]
        # Snapshot now, inside the loop and while src is still referenced: the
        # source can only be closed by an explicit aclose in phase_tagged_events.
        assert src  # keep the reference alive to here
        return list(closed), state

    closed_snapshot, state = asyncio.run(main())
    assert state.finish_reason == "end_turn"
    assert closed_snapshot == [True], "source was not closed deterministically"


def test_source_generator_closed_on_early_consumer_exit():
    """If the consumer stops early (client disconnect), the source is still
    closed — the finally runs on GeneratorExit."""
    closed = []

    async def source():
        try:
            while True:
                yield TokenEvent(token_id=104, request_id="r")
        finally:
            closed.append(True)

    async def main():
        src = source()  # strong ref
        state = DecodeState()
        gen = phase_tagged_events(src, FakeTokenizer(), state)
        await gen.__anext__()  # pull one event, then abandon
        await gen.aclose()
        assert src  # keep the reference alive
        return list(closed)

    assert asyncio.run(main()) == [True]

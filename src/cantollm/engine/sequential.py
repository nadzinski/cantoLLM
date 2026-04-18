"""Sequential inference engine: one request at a time on a worker thread."""

import asyncio
import threading
from collections.abc import AsyncIterator

from cantollm.engine.types import InferenceRequest, TokenEvent
from cantollm.kv_cache import KVCache

_QUEUE_MAXSIZE = 256


class SequentialEngine:
    """Runs one request at a time through the existing TokenGenerator.

    Generation is synchronous torch work, so we dispatch it to a worker thread
    and bridge its output into a bounded asyncio.Queue. Each active request
    has a threading.Event that the generator checks per step; abort() sets it.
    """

    def __init__(self, generator_factory, config):
        self.generator_factory = generator_factory
        self.config = config
        self._active: dict[str, threading.Event] = {}

    async def start(self) -> None:
        pass

    async def shutdown(self) -> None:
        for event in list(self._active.values()):
            event.set()

    def abort(self, request_id: str) -> None:
        event = self._active.get(request_id)
        if event is not None:
            event.set()

    async def submit(self, req: InferenceRequest) -> AsyncIterator[TokenEvent]:
        queue: asyncio.Queue[TokenEvent | BaseException | None] = asyncio.Queue(
            maxsize=_QUEUE_MAXSIZE
        )
        loop = asyncio.get_running_loop()
        stop_event = threading.Event()
        self._active[req.request_id] = stop_event

        def put_threadsafe(item):
            # Use call_soon_threadsafe + put_nowait. For a bounded queue this
            # can raise QueueFull on the loop thread — acceptable as a hard
            # backpressure signal, but in practice the consumer here drains
            # fast enough and the queue is sized generously.
            loop.call_soon_threadsafe(queue.put_nowait, item)

        def run():
            try:
                cache = KVCache(self.config["num_transformers"])
                gen = self.generator_factory(
                    req.sampling_params.temperature,
                    req.sampling_params.top_p,
                )
                # Let the generator (or its main generator, for speculative)
                # observe cancellation between steps.
                if hasattr(gen, "stop_event"):
                    gen.stop_event = stop_event
                elif hasattr(gen, "main") and hasattr(gen.main, "stop_event"):
                    gen.main.stop_event = stop_event

                for tok in gen.generate(
                    input_ids=req.prompt_token_ids,
                    cache=cache,
                    stop_token_ids=req.stop_token_ids,
                    max_tokens=req.max_tokens,
                ):
                    put_threadsafe(TokenEvent(token_id=tok))
                    if stop_event.is_set():
                        break
            except Exception as e:
                put_threadsafe(e)
            finally:
                put_threadsafe(None)

        worker_task = asyncio.create_task(asyncio.to_thread(run))

        try:
            while (evt := await queue.get()) is not None:
                if isinstance(evt, BaseException):
                    raise evt
                yield evt
        finally:
            stop_event.set()
            self._active.pop(req.request_id, None)
            # Keep a reference until the thread really exits so we don't get
            # "Task was destroyed" warnings on client disconnect.
            try:
                await worker_task
            except BaseException:
                pass

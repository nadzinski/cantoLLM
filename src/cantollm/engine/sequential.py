"""Sequential inference engine: one request at a time on a worker thread."""

import asyncio
import threading
from collections.abc import AsyncIterator

from cantollm.engine.types import FinishReason, InferenceRequest, TokenEvent
from cantollm.runtime import ModelRuntime

_QUEUE_MAXSIZE = 256


class SequentialEngine:
    """Runs one request at a time through an InferenceBackend.

    Generation is synchronous torch work, so we dispatch it to a worker thread
    and bridge its output into a bounded asyncio.Queue. Each active request
    has a threading.Event that the backend checks per step; abort() sets it.
    """

    def __init__(self, runtime: ModelRuntime):
        self.runtime = runtime
        self._active: dict[str, threading.Event] = {}

    @property
    def backend(self):
        return self.runtime.backend

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
        queue: asyncio.Queue[TokenEvent | None] = asyncio.Queue(maxsize=_QUEUE_MAXSIZE)
        loop = asyncio.get_running_loop()
        stop_event = threading.Event()
        self._active[req.request_id] = stop_event
        rid = req.request_id

        def put_threadsafe(item):
            # Real backpressure: schedule `queue.put(item)` on the loop and
            # block the worker thread until it completes. If the consumer
            # stops draining, the finally below drains the queue so this can
            # unblock and the backend can observe stop_event.
            future = asyncio.run_coroutine_threadsafe(queue.put(item), loop)
            try:
                future.result()
            except BaseException:
                # Loop is shutting down mid-put; worker will see stop_event
                # on its next iteration and exit.
                pass

        def run():
            tokens_emitted = 0
            try:
                cache = self.runtime.new_cache()
                self.runtime.backend.reset()
                for tok in self.runtime.backend.generate(
                    input_ids=req.prompt_token_ids,
                    cache=cache,
                    sampling=req.sampling_params,
                    stop_token_ids=req.stop_token_ids,
                    max_tokens=req.max_tokens,
                    stop_event=stop_event,
                ):
                    put_threadsafe(TokenEvent(token_id=tok, request_id=rid))
                    tokens_emitted += 1
                    if stop_event.is_set():
                        break
            except Exception as e:
                put_threadsafe(TokenEvent(error=str(e), request_id=rid))
                put_threadsafe(None)
                return

            reason: FinishReason
            if tokens_emitted >= req.max_tokens:
                reason = "max_tokens"
            elif stop_event.is_set():
                reason = "abort"
            else:
                # Backend returned without hitting max_tokens and without an
                # abort — it saw an EOS or configured stop token.
                reason = "end_turn"
            put_threadsafe(TokenEvent(finish_reason=reason, request_id=rid))
            put_threadsafe(None)

        worker_task = asyncio.create_task(asyncio.to_thread(run))

        try:
            while (evt := await queue.get()) is not None:
                yield evt
        finally:
            stop_event.set()
            self._active.pop(req.request_id, None)
            # Drain the queue so a producer blocked on put() can unblock;
            # the backend checks stop_event between steps and will then
            # return, after which run()'s finally pushes a terminator.
            while not worker_task.done():
                try:
                    await asyncio.wait_for(queue.get(), timeout=0.05)
                except asyncio.TimeoutError:
                    continue
            # Keep a reference until the thread really exits so we don't get
            # "Task was destroyed" warnings on client disconnect.
            try:
                await worker_task
            except BaseException:
                pass

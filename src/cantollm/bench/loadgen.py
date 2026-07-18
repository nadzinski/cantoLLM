"""Load generators: closed-loop worker pool + open-loop arrival scheduler
(bench-spec.md §2).

Both drive an abstract async `send(prompt, ...)` (see sse_clients.build_sender)
and return the records they produced — no HTTP knowledge here, which is what
lets the tests run against a scripted sender in milliseconds.

Closed loop: `concurrency` workers, each firing its next request as soon as
the previous completes, sharing one seeded prompt iterator, until the level's
request count is exhausted.

Open loop: one scheduler task computes seeded arrival times (exponential
gaps for poisson, constant for fixed) and fires a request task per arrival
regardless of completions, bounded by `max_inflight` — hitting that bound
means the *generator* saturated, recorded via `LoadResult.hit_inflight_cap`
and each record's dispatch lag.
"""

from __future__ import annotations

import asyncio
import random
import time
from dataclasses import dataclass, field

from cantollm.bench.records import RequestRecord


@dataclass
class LoadResult:
    records: list[RequestRecord] = field(default_factory=list)
    texts: dict[str, str] = field(default_factory=dict)   # prompt-run key → text
    hit_inflight_cap: bool = False
    aborted: bool = False


async def run_closed_loop(
    send,
    prompts,                      # iterator of Prompt (endless, seeded)
    *,
    concurrency: int,
    total_requests: int,
    cell_id: str,
    repeat: int,
    excluded: bool = False,
    abort: asyncio.Event | None = None,
    on_record=None,
) -> LoadResult:
    result = LoadResult()
    counter = iter(range(total_requests))
    lock = asyncio.Lock()

    async def next_index() -> int | None:
        async with lock:
            return next(counter, None)

    async def worker() -> None:
        while True:
            if abort is not None and abort.is_set():
                result.aborted = True
                return
            index = await next_index()
            if index is None:
                return
            record, text = await send(
                next(prompts),
                cell_id=cell_id, repeat=repeat, request_index=index,
                excluded=excluded,
            )
            _collect(result, record, text, on_record)

    await asyncio.gather(*(worker() for _ in range(int(concurrency))))
    return result


async def run_open_loop(
    send,
    prompts,
    *,
    rate_rps: float,
    arrivals: str,                # "poisson" | "fixed"
    total_requests: int,
    max_inflight: int,
    seed: int,
    cell_id: str,
    repeat: int,
    excluded: bool = False,
    abort: asyncio.Event | None = None,
    on_record=None,
    drain_timeout_s: float = 300.0,
) -> LoadResult:
    if rate_rps <= 0:
        raise ValueError("rate_rps must be > 0")
    result = LoadResult()
    rng = random.Random(seed)
    inflight = asyncio.Semaphore(int(max_inflight))
    tasks: set[asyncio.Task] = set()
    start = time.perf_counter()
    next_at = start

    async def fire(index: int, t_scheduled: float) -> None:
        try:
            record, text = await send(
                next(prompts),
                cell_id=cell_id, repeat=repeat, request_index=index,
                t_scheduled=t_scheduled, excluded=excluded,
            )
            _collect(result, record, text, on_record)
        finally:
            inflight.release()

    for index in range(total_requests):
        gap = rng.expovariate(rate_rps) if arrivals == "poisson" else 1.0 / rate_rps
        next_at += gap
        delay = next_at - time.perf_counter()
        if delay > 0:
            if abort is not None:
                try:
                    await asyncio.wait_for(abort.wait(), timeout=delay)
                except asyncio.TimeoutError:
                    pass
            else:
                await asyncio.sleep(delay)
        if abort is not None and abort.is_set():
            result.aborted = True
            break

        if inflight.locked():
            # The cap is the backstop between "measuring the server" and
            # "measuring our own queue" — record that we hit it.
            result.hit_inflight_cap = True
        while True:
            # Abort-aware acquire: the cap can block indefinitely against a
            # stalled server, and an abort must still get through.
            try:
                await asyncio.wait_for(inflight.acquire(), timeout=0.1)
                break
            except asyncio.TimeoutError:
                if abort is not None and abort.is_set():
                    result.aborted = True
                    break
        if result.aborted:
            break
        task = asyncio.create_task(fire(index, t_scheduled=next_at))
        tasks.add(task)
        task.add_done_callback(tasks.discard)

    if tasks:
        if result.aborted:
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
        else:
            _, pending = await asyncio.wait(tasks, timeout=drain_timeout_s)
            if pending:
                for t in pending:
                    t.cancel()
                await asyncio.gather(*pending, return_exceptions=True)
                result.aborted = True
    return result


def _collect(result: LoadResult, record: RequestRecord, text, on_record) -> None:
    result.records.append(record)
    if text is not None:
        # Key must be unique across the whole run's shared text file — a
        # (repeat, index, prompt) triple repeats across cells.
        result.texts[
            f"{record.cell_id}:{record.repeat}:{record.request_index}:{record.prompt_id}"
        ] = text
    if on_record is not None:
        on_record(record)

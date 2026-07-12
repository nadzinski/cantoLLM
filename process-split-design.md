# Design note: API/engine process split (Phase 2, item 2)

**Status: landed 2026-07-12.** The batched engine now defaults to a dedicated
engine process (`serve --engine batched`; `--in-process` keeps the old shape
for debugging). Sequential stays in-process — it's the debug path and the
speculative-decoding home, and it isn't a steady-state loop that wants a
process to itself.

## Shape

```
API process (FastAPI, async)                 engine process (sync busy loop)
────────────────────────────                 ───────────────────────────────
EngineProcessClient                          engine_process_main
  submit()/abort()  ──AddRequest/Abort/Shutdown──▶  command mp.Queue
  per-request asyncio.Queues                            │ drained per step
        ▲ call_soon_threadsafe                          ▼
  bridge thread  ◀──Ready | list[TokenEvent] | ──  drive_scheduler():
                    EngineFailed | Stopped         apply commands → step()
                    (events mp.Queue)              → one put per step
```

Both halves were already written for this: commands were message-shaped
dataclasses on a single thread-safe queue, and events were dispatched once
per step, not per token. The split replaces `queue.Queue` with
`multiprocessing.Queue` on both channels and moves the scheduler thread into
a child process. The loop itself is now a shared function
(`drive_scheduler` in `engine/batching/engine.py`) driven by the in-process
engine's thread and the engine process alike — the IPC-readiness claim made
literal. Likewise the API-side per-request multiplexing (`submit`/
`_dispatch`/`_fail`) is one base class (`engine/batching/mux.py`) shared by
`ContinuousBatchingEngine` and `EngineProcessClient`, so the two engines
can't drift apart behaviorally.

## Decisions

1. **IPC: stdlib `multiprocessing` queues, not ZMQ.** PLAN.md said "pick one
   and commit." ZMQ's advantages (cross-host sockets, language-agnostic
   framing, fan-out topologies) buy nothing for one engine process on the
   same box, and it adds a dependency plus hand-rolled serialization and
   process management. `mp.Queue` gives pickling of the existing command/
   event dataclasses unchanged, plus process lifecycle (spawn/join/exitcode)
   from the same package. vLLM's ZMQ usage is driven by multi-node needs we
   don't have; if Phase 8 ever wants cross-host IPC, the boundary is one
   message-shaped seam to swap.

2. **`spawn` context, explicitly.** Fork is unsafe once CUDA enters the
   picture (the child inherits a broken CUDA context) and macOS already
   defaults to spawn. Linux (the 5090 box) defaults to fork — hence explicit.
   Consequence: everything crossing the boundary must pickle, and the child
   re-imports modules from scratch.

3. **The child rebuilds the model; nothing rich crosses the boundary.**
   `ModelSpec` carries closures (weights loader, tokenizer factory) and
   can't pickle. The client is constructed with a module-level factory
   callable plus primitive kwargs (`size`, `device` string,
   `BatchingConfig`); the child calls it to load weights and compose the
   scheduler. Tests inject toy/scripted factories the same way.

4. **Wire protocol, child → parent, in order:** `Ready` once the scheduler
   is built (start() blocks on it — model load can be a long first-run
   download, so there's no overall deadline; only child death breaks the
   wait), then per-step `list[TokenEvent]` batches (one pickle per step,
   never per token), then exactly one farewell: `EngineFailed(reason)` (load
   or step failure; the batch-wide failure policy carries over verbatim) or
   `Stopped` (Shutdown acknowledged).

5. **Liveness in both directions, no heartbeat protocol.** The parent's
   bridge thread polls `events.get(timeout=0.5)` and, if the child died
   without a farewell (segfault, OOM-kill), fails all in-flight streams with
   the exit code. The child polls its command queue while idle and checks
   `parent_process().is_alive()` once per loop, so an orphaned engine
   (API SIGKILLed; `daemon=True` only covers clean parent exit) stops
   stepping instead of generating into a pipe nobody drains. Real /ready
   endpoints and restart supervision are Phase 3.5.

6. **Tokenization stays API-side; the API process never loads weights.**
   This is where Phase 1a's decision pays off: the parent holds a
   `TokenizerRuntime` (spec + tokenizer, nothing else) built via the new
   `ModelSpec.tokenizer_files_loader`, which downloads only tokenizer files.
   The engine process has something real to protect now — an 8k-prompt
   tokenization never competes with the scheduler loop for a core.

7. **Backpressure: keep the documented unbounded design at the new
   boundary.** Per-request event counts are bounded by admission
   (prompt + max_tokens ≤ slot capacity), events are tiny, and
   disconnect→abort reclaims the resource that matters (the KV slot). The
   sequential path's `put_nowait` edge got real backpressure earlier (see
   `sequential.py::put_threadsafe`); the CB path keeps unbounded queues with
   the same rationale as in-process, now plus the orphan guard in (5), which
   removes the one way "unbounded" could become "unbounded growth".

8. **uvicorn now runs uvloop + httptools explicitly.** `uvicorn[standard]`
   already shipped both and `loop="auto"` was likely picking them up;
   pinning them makes the end-of-phase baseline reproducible and fails loud
   if the extras go missing.

## Alternatives considered

- **ZMQ** — see (1). Rejected for scope, not on merits.
- **Shared memory / pipes for token tensors** — nothing tensor-shaped
  crosses the boundary; events are a few ints and floats. Pickle overhead
  measured in microseconds per step-batch is noise next to a forward pass.
- **Splitting the sequential engine too** — it's not a busy loop; it gains
  nothing from a process and would drag the speculative path through IPC for
  free complexity.
- **One process per model in the registry** — the registry API already
  permits it (each entry owns its engine client); deliberately not built
  until multi-model serving is real (Phase 5's routing experiment).

## Testing

`tests/test_process_engine.py`: toy-oracle token equivalence through a real
spawned child; sampling-params pickle round-trip; ready handshake and
factory-failure surfacing; batch-wide step failure; hard child death
(`os._exit`) detection; abort and disconnect-frees-slot through the real
scheduler; shutdown reaps the child and closes streams; API-level SSE smoke
over the process engine. Factories are module-level so they pickle across
spawn.

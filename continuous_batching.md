# Continuous batching: prototype → real-project integration notes

Investigation of how the `prototypes/continuous_batching/` design lands inside
`src/cantollm/`. Not a roadmap (PLAN.md owns that) — a design note covering
the seams that have to move and where the friction lives.

## Integration shape, briefly

The prototype lays down four artifacts that the real project needs analogues of:

| Prototype | Real-project analogue |
|---|---|
| `PaddedKVCache` (single-layer pool) | A multi-layer pool owned by `ModelRuntime`, replacing per-request `KVCache` for batched-eligible models |
| `ToyModel.forward(input_ids, slot_metas, kv_cache)` | A new `Qwen3.forward` path with per-row `start_pos`/`num_new`, plus a filled-in `PaddedAttentionMethod` |
| `ContinuousBatchingScheduler` | A new `ContinuousBatchingEngine` (sibling to `SequentialEngine`) that owns the scheduler loop and routes step events to per-request queues |
| `runner.run_to_completion` | Async multiplexer feeding many `submit()` async iterators from one scheduler thread |

`InferenceEngine` Protocol is unchanged externally — `submit/abort/start/shutdown`
stay the same. The registry can hold both engines side-by-side (e.g. speculative
on `SequentialEngine`, dense Qwen3 on `ContinuousBatchingEngine`). API
tokenization stays untouched.

## Tricky points

**1. The KV cache shape is a load-bearing decision.** Today `Sequence.cache` is
a per-request `KVCache` (list of grow-via-cat dicts). The prototype shifts to a
shared, preallocated, slot-indexed pool. These two can't unify cleanly — either
the runtime owns the pool and `Sequence.cache` becomes a slot handle, or you
get parallel cache types for sequential vs batched. Worth picking deliberately
because Phase 4 paged KV is going to inherit this shape.

**2. `InferenceBackend.generate(seq) -> Iterator[int]` is per-sequence by
construction.** That contract assumes one consumer drives one request through
prefill→decode. Continuous batching has one driver and many consumers — the
protocol doesn't fit. Cleanest break: the new engine doesn't go through
`InferenceBackend` at all; it calls the model directly via the `AttentionMethod`
seam. `StandardBackend` and `SpeculativeBackend` stay on the sequential path.

**3. `Qwen3.forward(tokens, start_pos: int, kv_cache)` has a scalar
`start_pos`.** RoPE (`apply_rotary_emb(..., offset=start_pos)`),
`_validate_cache`, mask construction, and the attention method all consume it
as a scalar. Continuous batching needs per-row positions and per-row `num_new`.
The `AttentionMethod` Protocol is the right seam to widen, but RoPE in
particular is on the hot path and will need either a Python-loop variant or a
`(B, num_new_max)` position gather.

**4. Mask shape changes from 2D to 3D.** `EinsumAttentionMethod.build_mask`
returns `(seq_len, start_pos+seq_len)`. Padded needs
`(B, num_new_max, max_history_len)` accounting for per-row history and ragged
`num_new`. The protocol signature `(start_pos, seq_len, device)` is the wrong
shape for batching — needs a per-row variant.

**5. Real model has GQA with groups + heads.** Prototype has 1 head, no GQA, no
q/k norm, no per-layer iteration. Real attention math is grouped, multi-head,
RMS-normed Q/K, and runs across N transformer layers. The prototype's per-row
Python loop inside `forward` is the simplest correctness path but devastating
perf on the real model — the padded path will want vectorized batched einsum
(which `EinsumAttentionMethod` mostly has already; the new bit is per-row
history slicing).

**6. Sampling is greedy-only in the prototype.** Real path has `SamplingParams`
with a per-request `LogitsProcessor` pipeline (Temperature, TopP, future
repetition penalty etc.). After a batched forward you get `(B, vocab)` logits;
the easiest correctness path is to loop rows post-forward, accept the cost,
optimize later with batched processors. Greedy is the only one that's trivially
`argmax(dim=-1)`.

**7. Multiplexing one scheduler loop into N async iterators is not just "submit
with a queue" again.** `SequentialEngine` gets away with one worker thread +
one queue per request because the engine *is* the worker. With batching there's
one scheduler thread producing events for many requests, and the `submit()`
async iterators each need their own queue keyed by `request_id`. The mid-stream
abort path is the trickiest sub-case — `stop_event` setting needs to (a) signal
the scheduler to free that slot, (b) emit a `finish_reason="abort"` event,
(c) close the per-request queue cleanly without leaving the worker thread
blocked on a `put`.

**8. `stop_event` polling is new state for the scheduler.** Each running
sequence has one already; the scheduler has to remember to poll them all
between steps, finalize aborted ones, free slots, and emit abort events.
Prototype doesn't model this.

**9. KV pool sizing is real memory budget.** For 0.6B Qwen3 (28 layers, 8
groups, head_dim 128, bf16): a shared pool at `max_batch=8, max_seq_len=4096`
is ~6 GB for KV alone. The model's existing `max_seq_len=40_960` knob is way
too large to use directly — there needs to be a separate per-sequence cap for
the batched cache, tunable via spec/CLI, smaller on Mac than 5090. Two
different "max seq lens" need to coexist in config.

**10. Tests need both a toy-model layer (fast scheduler correctness) and a
tiny-Qwen3 layer (integration).** `tests/tiny_model.py` already builds a
2-layer random-init Qwen3 — perfect for end-to-end batched-vs-sequential
equivalence tests, the same shape as the prototype's "scheduler matches
reference token-for-token." Don't drop the toy-model tests; they're 10× faster
and pin scheduler logic in isolation from model bugs.

**11. Process split is a separate piece.** PLAN.md is explicit: in-process
scheduler first, IPC boundary second. The prototype's design supports that
ordering — nothing in the engine refactor forces a process split. Easy to be
tempted to do both at once; the debugging cost is much lower if they're
sequenced.

**12. Speculative decoding is the explicit non-goal here.** PLAN.md disables
batched speculation — register speculative models on the sequential engine,
dense models on the batched engine, and don't try to unify the two engine
paths.

## Suggested ordering of prereq refactors before the scheduler drops in

1. Widen `AttentionMethod` (build_mask + forward_*) for per-row state. Padded
   variant still stub but signature settles.
2. Adapt `Qwen3.forward` and `apply_rotary_emb` to accept per-row positions.
3. Build the multi-layer `PaddedKVCache` and have `ModelRuntime` own it (with
   a config switch — sequential runtimes still hand out per-request `KVCache`).
4. Fill in `PaddedAttentionMethod` against the new shapes; validate against
   `EinsumAttentionMethod` on a single-row prefill.
5. Drop in the scheduler as a new `ContinuousBatchingEngine`; register it for
   dense models in `main.py`.

The scheduler/batched-forward/padded-KV pieces themselves are the author's per
PLAN.md's "Author note" in Phase 2 — these prereqs and the engine/queue
multiplexing in step 5 are fair game for assistants.

# Continuous batching: prototype → real-project integration notes

Investigation of how the `prototypes/continuous_batching/` design lands inside
`src/cantollm/`. Not a roadmap (PLAN.md owns that) — a design note covering
the seams that have to move and where the friction lives.

Revised 2026-07-04 after a code-level review pass: the open questions are now
committed decisions, the fable-review.md findings that bite this plan are
folded into the checklist, and the prereq ordering gained two steps (a shared
sampler, and an explicit engine↔model protocol).

## Integration shape, briefly

The prototype lays down five artifacts that the real project needs analogues of:

| Prototype | Real-project analogue |
|---|---|
| `PaddedKVCache` (single-layer pool) | A multi-layer pool: memory owned by `ModelRuntime`, slot allocator driven by the scheduler. Replaces per-request `KVCache` for batch-eligible models |
| `ToyModel.forward(input_ids, slot_metas, kv_cache)` | A narrow batched-forward protocol implemented by `Qwen3` — per-row `start_pos`/`num_new`, returning `(B, vocab)` — plus a filled-in `PaddedAttentionMethod` |
| `greedy_sample` | A shared sampler module extracted from `StandardBackend` (logits-processor pipeline + greedy/multinomial), consumed by both engine paths |
| `ContinuousBatchingScheduler` | A new `ContinuousBatchingEngine` (sibling to `SequentialEngine`) that owns the scheduler loop and routes step events to per-request queues |
| `runner.run_to_completion` | Async multiplexer feeding many `submit()` async iterators from one scheduler thread |

`InferenceEngine` Protocol is unchanged externally — `submit/abort/start/shutdown`
stay the same. The registry can hold both engines side-by-side (e.g. speculative
on `SequentialEngine`, dense Qwen3 on `ContinuousBatchingEngine`). API
tokenization stays untouched.

## Decisions

Formerly open questions, committed here so the prereq work below can assume
them.

1. **The runtime owns the pool memory; the scheduler owns the allocator
   state; a sequence carries a slot handle.** Phase 4's paged cache is the
   same shape — a block table is also a handle into a runtime-owned pool — so
   the seam gets designed once. Sequential runtimes keep handing out
   per-request `KVCache` via a config switch.
2. **The CB engine gets its own per-request state dataclass** (the prototype
   `Sequence` shape: slot, position, output tokens) rather than force-unifying
   with `engine/types.Sequence`, whose required `cache: KVCache` field and
   per-request `stop_event` are sequential-path artifacts. Unify later only if
   it earns it.
3. **Stop tokens are suppressed, not emitted.** The prototype emits the stop
   token then finishes (`scheduler.py:145`); `StandardBackend.generate`
   returns without yielding it (`standard.py:103`), which is the correct API
   behavior and wins. The stop check moves *before* emission during the port,
   and finish becomes its own event — the real `TokenEvent` contract
   (`engine/types.py:84`) populates exactly one field per event, so the
   prototype's combined token+finish event becomes two.
4. **The engine talks to the model through a narrow batched-forward protocol,
   not `Qwen3` directly.** The prototype already discovered the contract:
   `forward_batched(input_ids, slot_metas, pool) -> (B, vocab)` logits at each
   row's last real token. `ModelRuntime` fronts it, so the engine never
   imports a model class and Gemma (Phase 6) slots in without touching the
   scheduler.
5. **Per-request event queues are unbounded; disconnect→abort is the whole
   backpressure story.** `SequentialEngine`'s blocking-put backpressure
   (`sequential.py:48`) is unusable here — one stalled SSE consumer would
   freeze every request in the batch. Token events are tiny; if a client
   stops reading, the disconnect path aborts the request and frees its slot.
6. **`add_request` and `abort` are commands on one thread-safe command queue**
   drained at the top of each step — not N `stop_event`s the scheduler has to
   remember to poll. Cleaner, and it is already the shape the Phase 2 IPC
   split wants (commands crossing a boundary).
7. **`max_batch`, the batched per-sequence cap, and `max_tokens_per_step` are
   engine config, not `ModelSpec`.** They're deployment knobs (smaller on Mac
   than on the 5090), not model identity.

## Pre-work (land before any of this)

- **The `SequentialEngine` concurrency hole** (fable-review #1): `submit()`
  spawns a worker thread per request with no lock, and `SpeculativeBackend`
  carries shared mutable draft-cache state. Two engines behind one API make
  this *more* reachable, not less. Cheap lock fix; do it first.
- **The bench-harness spec** has been Open since Phase 0, and Phase 2's exit
  numbers are the "before" baseline Phase 3 measures against. Time-box the
  spec before the scheduler work finishes, not after.

## Tricky points

**1. `InferenceBackend.generate(seq) -> Iterator[int]` is per-sequence by
construction — and bypassing it leaves sampling homeless.** The contract
assumes one consumer drives one request through prefill→decode; continuous
batching has one driver and many consumers, so the new engine doesn't go
through `InferenceBackend` at all (`StandardBackend` and `SpeculativeBackend`
stay on the sequential path). But the logits-processor pipeline and sampling
live on `StandardBackend.get_probs`/`sample` (`standard.py:25-58`) — bypass
the backend and the new engine has nowhere to get sampling from except by
copying it. Extracting a shared sampler is therefore a prereq in its own
right, and the extraction is the moment to fix the greedy shortcut
(fable-review #4): `sample` skips the pipeline when `greedy=True`, which is
correct for temperature/top-p but silently wrong the day repetition penalty or
logit bias lands. Fix it once instead of inheriting it twice.

**2. `Qwen3.forward(tokens, start_pos: int, kv_cache)` has a scalar
`start_pos` — and the batched path should also change the return shape.**
RoPE (`apply_rotary_emb(..., offset=start_pos)`), `_validate_cache`, mask
construction, and the attention method all consume `start_pos` as a scalar;
batching needs per-row positions and per-row `num_new`. RoPE is on the hot
path and wants a `(B, num_new_max)` position gather over `freqs_cis` (a
Python-loop variant works as a first correctness pass). Separately: the toy
contract returns `(B, vocab)` at each row's last real token, while real
`Qwen3.forward` runs `output_layer` over *every* position. The batched path
must gather per-row last positions anyway (ragged `num_new`), and doing the
gather *before* the lm_head avoids materializing `(B, num_new_max, 151936)`
logits — ~150 MB per 512-wide prefill row in bf16. Rows still mid-prefill
need no logits at all and can be skipped entirely.

**3. The prefill/decode dichotomy dissolves in a mixed batch.** A continuous-
batching step mixes prefill-chunk rows and decode rows in one forward, so the
padded path can't be split across `forward_prefill`/`forward_decode` — it
needs a single mixed-batch entrypoint. Two more casualties of the same fact:
`GroupedQueryAttention.forward` dispatches on cache emptiness
(`model.py:169`), which is meaningless for a preallocated pool that is never
"empty"; and `_validate_cache` checks `cache["keys"].shape[1] == start_pos`,
a grow-via-cat assumption. Consequence for the protocol: **add new batched
methods to `AttentionMethod` rather than widening the existing signatures.**
Union types like `start_pos: int | Tensor` would contort
`EinsumAttentionMethod`, which is the frozen correctness reference — the
sequential signatures should not move.

**4. Mask shape changes from 2D to 3D.** `EinsumAttentionMethod.build_mask`
returns `(seq_len, start_pos+seq_len)`. Padded needs
`(B, num_new_max, max_history_len)` accounting for per-row history and ragged
`num_new` (broadcast over the group/head dims at use). Build it once per step
at the model level, as today — history lengths per row are the same across
layers.

**5. Real model has GQA with groups + heads — vectorize the math, loop the
writes.** The prototype has 1 head, no GQA, no q/k norm, one layer; its
per-row Python loop is the simplest correctness path but devastating perf on
the real model. The vectorized version is less scary than it looks: gather
`pool.k[slots, :max_hist]` to `(B, max_hist, groups, head_dim)` in one index
op and let the 3D mask do the ragged work — `EinsumAttentionMethod` already
has most of the einsum. The genuinely fiddly part is the ragged KV *write*
(`num_new` differs per row); a small per-row loop of slice-assignments is a
fine middle ground, since it's bookkeeping, not math. One trap if the write
is vectorized instead: pad columns writing K/V past a row's real tokens are
harmless (the next step overwrites them) *except* when `start_pos +
num_new_max` overruns the slot — bounds-check the write, not just the row.
Also note the padded-width cost: decode rows get padded to the widest prefill
chunk in the batch, so QKV/FF compute scales with `B × num_new_max` even when
519 of the 4096 token-slots are real. The prefill chunk size is a real knob;
tune it when Phase 3's measurement lands.

**6. Sampling is per-row after the batched forward.** Real requests carry
`SamplingParams` with a per-request `LogitsProcessor` pipeline. After a
batched forward you get `(B, vocab)` logits; the easiest correctness path is
to loop rows through the shared sampler (point 1), accept the cost, optimize
later with batched processors. Greedy is the only case that's trivially
`argmax(dim=-1)` — and greedy is also the only mode where batched-vs-
sequential equivalence tests are even *possible* (multinomial draws RNG in a
different order across the two paths by construction).

**7. Multiplexing one scheduler loop into N async iterators is not just
"submit with a queue" again.** `SequentialEngine` gets away with one worker
thread + one queue per request because the engine *is* the worker. With
batching there's one scheduler thread producing events for many requests;
each `submit()` async iterator gets its own queue keyed by `request_id`, fed
via `loop.call_soon_threadsafe` with non-blocking puts (decision 5 — the
scheduler thread must never block on a consumer). The abort path: an abort
command (decision 6) makes the scheduler free the slot, emit a
`finish_reason="abort"` event, and close that queue — with unbounded queues
there's no blocked-`put` case to untangle. The shell also needs the loop's
idle mechanics: when no sequences are waiting or active, the scheduler thread
should block on the command queue rather than spin.

**8. KV pool sizing is a real memory budget — and needs enforcement, not
just arithmetic.** For 0.6B Qwen3 (28 layers, 8 KV groups, head_dim 128,
bf16): 2 × 28 × 8 × 128 × 2 bytes ≈ 112 KB per token, so a pool at
`max_batch=8, max_seq_len=4096` is ~3.8 GB for KV alone (an earlier draft
said ~6 GB; fable-review's re-derivation is right). The model's existing
`max_seq_len=40_960` knob is far too large to preallocate — a separate
per-sequence cap enters the engine config (decision 7), so two different
"max seq lens" coexist. Enforcement (fable-review #2): nothing in the
prototype checks `position + num_new <= max_seq_len`, and one overlong
request crashes the shared forward for the whole batch. The integrated
version needs an admission check — reject `prompt_len + max_tokens > cap`
with a 400 at the API, which means the registry entry exposes the cap — plus
the per-step write bound from point 5.

**9. Tests need both a toy-model layer (fast scheduler correctness) and a
tiny-Qwen3 layer (integration) — and the equivalence promise is softer on
the real stack.** `tests/tiny_model.py` already builds a 2-layer random-init
Qwen3, perfect for batched-vs-sequential equivalence tests. Keep the
toy-model tests too; they're 10× faster and pin scheduler logic in isolation
from model bugs. The caveat: the prototype is bitwise-deterministic because
both paths run the same per-row float32 CPU loop. A vectorized batched
einsum is a different kernel shape with a different reduction order, so on
MPS/bf16 greedy token-for-token equality can flake on near-tie logits. Plan
for: tiny-Qwen3 equivalence on float32 CPU as the strict test, and
logit-tolerance comparisons as the assertion that runs on real hardware.

**10. Process split is a separate piece.** PLAN.md is explicit: in-process
scheduler first, IPC boundary second. The prototype's design supports that
ordering — nothing in the engine refactor forces a process split. Easy to be
tempted to do both at once; the debugging cost is much lower if they're
sequenced. (The command queue in decision 6 is deliberately IPC-shaped.)

**11. Speculative decoding is the explicit non-goal here.** PLAN.md disables
batched speculation — register speculative models on the sequential engine,
dense models on the batched engine, and don't try to unify the two engine
paths.

## Suggested ordering of prereq refactors before the scheduler drops in

0. Pre-work above: the `SequentialEngine` lock fix and the bench-spec
   time-box.
1. Extract the shared sampler from `StandardBackend` (processor pipeline +
   greedy/multinomial), fixing the greedy shortcut once (point 1).
2. One design pass sketching the batched signatures across the
   `AttentionMethod` additions, the pool layout, and the model entrypoint
   *together* — the protocol signature depends on the pool layout, so don't
   settle it in isolation. Deliverable: the stub signatures steps 3–5 fill in.
3. Build the multi-layer `PaddedKVCache`: `ModelRuntime` owns the memory
   (with a config switch — sequential runtimes still hand out per-request
   `KVCache`), scheduler-facing allocator API, deterministic free-slot order
   (fable-review's reproducibility nit).
4. Add the batched entrypoints: new `AttentionMethod` methods (sequential
   signatures untouched; einsum raises on the batched ones) and
   `Qwen3.forward_batched` — per-row RoPE gather, 3D mask built once per
   step, per-row last-token gather *before* the lm_head, `(B, vocab)` return
   — formalized as the protocol `ModelRuntime` fronts (decision 4).
5. Fill in `PaddedAttentionMethod` against the new shapes; validate against
   `EinsumAttentionMethod` on a single-row prefill, then on batches. ✍
6. Port the scheduler into a new `ContinuousBatchingEngine` ✍ and build the
   engine shell around it: command queue, scheduler thread + idle wake,
   unbounded per-request queues, async multiplexer, abort path. Port
   checklist from fable-review: stop check before emission (decision 3),
   `>=` on the max_tokens check, finish as its own event, don't sample rows
   still mid-prefill. Register for dense models in `main.py`.

## Division of labor

Per PLAN.md's Phase 2 author note, marked ✍ above: the scheduler port and the
`PaddedAttentionMethod` math (step 5 and the scheduler half of step 6) are the
author's, written by hand for learning. Assistants own the pre-work, steps
1–4, the engine shell/multiplexer half of step 6, and tests throughout. One
deliberate deviation from the author note's letter: PLAN.md lists "padded KV"
as author-written, but the prototype already proved that design, so the
multi-layer port (step 3) is treated as mechanical and assistant-fair-game —
flag if it should stay hand-written.

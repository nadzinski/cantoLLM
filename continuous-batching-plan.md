# Integrating the continuous-batching prototype into src/cantollm

## Context

Phase 2's headline work: port `prototypes/continuous_batching/` (scheduler, padded KV
pool, water-fill budget allocator — 29 tests green) into the real package as a
`ContinuousBatchingEngine` sibling to `SequentialEngine`. Baseline is
`old_research_continuous_batching.md`'s ordering 0–6 with its committed decisions 1–7 (the
original design note, which this plan supersedes as source of truth), plus the
fable-review fixes. This plan keeps that ordering but restructures step 6 (it was three
sessions pretending to be one) and folds in gaps found during planning.

**Division of labor (settled):** per the design note — Nadia hand-writes the
`PaddedAttentionMethod` batched attention math (step 5) and the scheduler port (step 8);
Claude writes everything else (pre-work, sampler, stubs, pool, model plumbing, engine
shell, admission, tests throughout). Steps are shaped so Nadia's sessions are pure
red→green against pre-landed test suites.

**Scope decisions (settled):** bench-harness spec consciously skipped for now (accepted
risk: Phase 2 exits with only rough numbers from the existing `bench.py`); viz/ gets one
refresh step at the end rather than per-step churn; logprobs + stop strings included as
optional tail steps; process split and batched speculation stay out of scope.

## Key deviations from old_research_continuous_batching.md (it said don't treat it as gospel)

1. **Step 6 split into three** — engine shell (vs a scripted fake scheduler), scheduler
   port, wire-up — and the shell pulled forward: it has zero model dependency, and the
   threading code deserves its own fast torch-free test suite instead of being debugged
   through an end-to-end tube.
2. **Author-facing test suites pre-land as xfail** one step before each hand-written
   chunk, so those sessions are red→green on core logic only.
3. **Gaps the note missed, now owned by specific steps:** step-failure policy (a raised
   exception in the shared forward is batch-wide — define it, don't discover it);
   engine shutdown path; `max_tokens<=0` short-circuit (the `>=` fix alone still emits
   one spurious token); abort of a still-queued request (no slot to free); `CBSequence`
   naming to avoid colliding with `engine/types.Sequence`.
4. **3D-mask insight worth pinning in tests:** pure per-row causality
   (`mask[b,i,j] = j > start_pos[b]+i`) also hides stale K/V from previous slot
   occupants and keeps pad query rows finite — no slot-zeroing on free, no NaN handling.
5. **Chunked prefill is not a separate feature** — it falls out of `water_fill` and the
   ported prototype tests. Don't schedule it twice.
6. **Mask ownership split:** `build_batched_mask` is geometry/bookkeeping → Claude,
   step 4; `forward_batched` attention math → Nadia, step 5.

## Step sequence

Each step is one focused session, independently landable, suite green at the end.
Order of steps 5–8 is flexible: the scheduler port (step 8) only needs steps 2–3, so
Nadia can take it right after step 3 if she wants hand-writing sooner.

### Step 0 — SequentialEngine serialization fix + housekeeping  [Claude]
Fable-review #1: `submit()` spawns a worker per request with no lock;
`SpeculativeBackend`'s shared `draft_cache` + per-request `reset()` corrupt concurrent
requests. Add a lock serializing the `reset()+generate()` region in
`src/cantollm/engine/sequential.py` (check `stop_event` after acquiring, so an aborted
waiter exits cleanly). New `tests/test_sequential_engine.py`: fake backend proves no
overlap; abort-while-waiting yields `finish_reason="abort"` with zero tokens.
Housekeeping: commit the pending design-note edits — the file is now
`old_research_continuous_batching.md` (renamed; this plan supersedes it as source of
truth) — along with the rename, this plan, and the reference updates.
**Done when:** overlap test fails on old code, passes on new; suite green.

### Step 1 — Shared sampler extraction + greedy-pipeline fix  [Claude]
Extract processor-pipeline + greedy/multinomial from `StandardBackend`
(`src/cantollm/standard.py:25-58`) into `src/cantollm/engine/sampler.py`
(module functions accepting `(vocab,)` and `(B, vocab)`); `StandardBackend` methods
become thin delegates (keep them — `speculative.py` calls them). Fix fable-review #4:
greedy runs the pipeline then argmax, no shortcut. New `tests/test_sampler.py` with a
ranking-inverting mock processor that fails against the old code.
**Done when:** inversion test green; no behavior change elsewhere in the suite.

### Step 2 — Design pass: land the seam stubs as code  [Claude writes, Nadia signs off]
Everything later steps compile against, committed as importable stubs:
- `models/attention/protocol.py`: `BatchMeta` frozen dataclass (`rows` list of
  `(slot_idx, start_pos, num_new)` for loop-the-writes; `slots/start_pos/num_new`
  `(B,)` tensors + `positions (B, num_new_max)` for vectorized gathers;
  `num_new_max`, `max_history_len`) and two protocol additions —
  `build_batched_mask(meta, device) -> (B, num_new_max, max_history_len) bool` and
  `forward_batched(queries, keys, values, mask, layer_k, layer_v, meta)` where
  `layer_k/v` are `(max_batch, max_seq_len, groups, head_dim)` pool views written in
  place. Sequential signatures untouched; `EinsumAttentionMethod` raises on batched.
- `src/cantollm/kv_pool.py`: `PaddedKVPool` — memory only, no allocator state
  (decision 1); `layer(i)` returns no-copy views.
- `src/cantollm/engine/batching/config.py`: frozen `BatchingConfig(max_batch,
  max_seq_len, max_tokens_per_step)` with the prototype's
  `max_tokens_per_step >= max_batch` guard in `__post_init__`.
- `models/qwen3/model.py`: stub `Qwen3.forward_batched(input_ids, meta, pool) ->
  (B, vocab)` — docstring commits to: 3D mask built once per step, per-row hidden-state
  gather at column `num_new[b]-1` **before** `output_RMSNorm` + lm_head (skipping
  lm_head for mid-prefill rows is an explicit Phase-3 TODO).
- `runtime.py`: stub `ModelRuntime.new_kv_pool(config)` and
  `ModelRuntime.forward_batched(...)` (decision 4 — the engine never imports `Qwen3`);
  plus a `BatchedForwardFn` protocol in `engine/batching/` (the scheduler's test seam).
Note: the existing `padded.py` stub carries old sequential-shaped signatures from
before the design note — reshape it here, per note point 3 (single mixed-batch
entrypoint, not a prefill/decode split).
**Done when:** stubs import, `BatchingConfig` validation tested, einsum raises on
batched methods, `test_attention.py` untouched-green, Nadia has signed off on signatures.

### Step 3 — Multi-layer PaddedKVPool + SlotAllocator + runtime wiring  [Claude]
Fill `kv_pool.py`; new `engine/batching/allocator.py` `SlotAllocator` with a
`deque` free list in ascending order (fixes the nondeterministic-`set` nit);
implement `ModelRuntime.new_kv_pool` from `spec.arch`. Port
`prototypes/.../padded_kv.py` semantics + its tests into `tests/test_kv_pool.py`, plus
view-aliasing and FIFO-determinism pins.
**Done when:** `build_runtime(tiny_qwen3_spec(), cpu).new_kv_pool(cfg)` gives correct
shapes; allocator tests green.

### Step 4 — Batched entrypoints (everything except attention math)  [Claude]
- `models/rope.py`: `apply_rotary_emb_batched(x, freqs_cis, positions (B,S))` — gather
  over `freqs_cis`, matching the existing complex-halves recombination convention
  exactly (pin by test against scalar `apply_rotary_emb` per row, exact equality).
- `models/attention/padded.py`: implement `build_batched_mask` (pure per-row causal).
- `models/qwen3/model.py`: `GroupedQueryAttention.forward_batched` (same
  projections/q-k-norm as sequential, batched RoPE, delegate to method — no
  cache-emptiness dispatch), `Transformer.forward_batched`, `Qwen3.forward_batched`
  (embed → one mask → per-layer with `pool.layer(i)` → last-token gather → lm_head).
- `runtime.py`: `forward_batched` body; `build_runtime` gains
  `attention: Literal["einsum","padded"]` switch replacing the hardcoded pin at
  `runtime.py:55`.
- Green tests (`tests/test_batched_model.py`): RoPE row-equivalence; each `mask[b]`
  equals the einsum 2D mask on its real region; plumbing shapes + gather-column
  verified via a recording `FakeAttentionMethod`.
- **Pre-land Nadia's step-5 suite as xfail(NotImplementedError)**
  (`tests/test_padded_equivalence.py`, tiny-Qwen3 float32 CPU, padded vs einsum):
  single-row prefill logits allclose; prefill-then-decode; mixed batch vs per-row
  sequential oracle; pad-invariance (row alone == same row in a ragged batch);
  pool-write pin; stale-slot-reuse (garbage slot freed and reused == fresh slot);
  bounds-assert fires on overlong write.
**Done when:** plumbing tests green; equivalence suite collected and xfailing on
exactly `NotImplementedError`; sequential suite untouched.

### Step 5 — PaddedAttentionMethod.forward_batched  [NADIA, hand-written]
The batched GQA attention math against frozen shapes: vectorized gather
`layer_k[meta.slots, :max_history_len]`, the two einsums from
`EinsumAttentionMethod._attend` (`einsum.py:50-91`) with the 3D mask unsqueezed over
group/head dims, per-row loop for the ragged KV writes with a hard bounds assert
(vectorize the math, loop the writes). `toy_model.py:71-113` is the 1-head reference.
Claude's role: interpreting test failures only.
**Done when:** all xfail markers removed from `test_padded_equivalence.py`, green on
CPU float32; a loose-tolerance MPS variant green locally.

### Step 6 — ContinuousBatchingEngine shell + async multiplexer  [Claude]
New `engine/batching/engine.py` + `engine/batching/types.py` (command dataclasses
`AddRequest/Abort/Shutdown`; `SchedulerLike` protocol: `add_request/abort/step/is_idle`).
Mechanics per decisions 5–6: one `queue.Queue` command queue drained at the top of each
step; scheduler thread blocks on it when idle (no spin); one
`loop.call_soon_threadsafe(dispatch, events)` hop **per step, not per token**
(deliberately IPC-shaped); dispatch routes by `request_id` into unbounded per-request
`asyncio.Queue`s with `put_nowait`, `None` terminator on finish/error; `submit`'s
`finally` → abort + unregister (disconnect→abort is the whole backpressure story);
`shutdown` joins the thread then abort-finishes anything in flight so no iterator
hangs. **Step-failure policy** (gap in the note): `step()` exception → error event to
every in-flight request, engine marked failed, later submits get an immediate error.
Tests (`tests/test_cb_engine_shell.py`, pytest-asyncio, torch-free, against a
`ScriptedScheduler`): per-request ordering, interleaved routing, finish terminates
iterator with one-field-per-event, early consumer break → Abort observed, explicit
abort closes stream, **idle no-spin + late submit wakes the thread** (pins fable-review
#5's idle-drop trap at the shell level), shutdown with in-flight, step-raise → error
events.
**Done when:** shell suite green in <1s, no "Task was destroyed" warnings.

### Step 7 — Admission control: `prompt_len + max_tokens <= cap` → 400  [Claude]
`RegistryEntry` gains `max_request_tokens: int | None`; check lives in
`api/common.py` beside `tokenize_and_build_request` (first place prompt length exists);
both routers map `AdmissionError` → 400 naming prompt tokens, max_tokens, and cap.
Contract tests both dialects: no cap → no check; `== cap` → 200; `+1` → 400 and
`FakeEngine.submit` never called.
**Done when:** both dialects reject over-cap; capless entries behave exactly as before.

### Step 8 — Scheduler port  [NADIA, hand-written; skeleton + toy tests pre-landed by Claude]
Claude pre-lands: `engine/batching/scheduler.py` skeleton (stubs + docstrings;
`water_fill` and `Row` ported verbatim — proven, not the learning target), `CBSequence`
dataclass (prototype `Sequence` shape + `sampling_params`; decision 2), a toy stepper
(`tests/toy_stepper.py` adapting the prototype `ToyModel` to the
`BatchedForwardFn`/allocator seams), and `tests/test_cb_scheduler.py` as xfail — the
ported prototype suite updated to the real contract, plus new abort/max_tokens=0/
over-cap cases. Scheduler constructor takes `(forward_fn, pool, allocator, config)` —
never sees `Qwen3` or `ModelRuntime`. Sampling: per-row loop through
`engine/sampler.py` on the `(B, vocab)` logits.
Nadia's port checklist (each item pinned by a pre-landed test):
1. Stop token suppressed — check moves before emission (decision 3).
2. Finish is its own `TokenEvent` (one field per event).
3. `>=` on the max_tokens check.
4. `max_tokens <= 0` short-circuits at `add_request` (the `>=` fix alone still emits
   one token — gap the note missed).
5. `abort`: running → free slot + abort event; still-queued → remove + abort event.
6. `add_request` re-validates the cap (defense behind step 7 — non-API callers exist).
7. No sampling/emission for mid-prefill rows (prototype already correct).
8. Deterministic slots via `SlotAllocator`.
**Done when:** toy scheduler suite fully green, milliseconds on CPU, zero real-model
dependency.

### Step 9 — Wire-up: real engine end-to-end + registration  [pair]
Default `scheduler_factory` composing `runtime.forward_batched` +
`runtime.new_kv_pool(config)` + `SlotAllocator`. `main.py`: `serve --engine
{sequential,batched}` (default sequential until soak-tested), batching-config flags,
error on `--engine batched --speculative`, `build_runtime(..., attention="padded")`,
`registry.register(..., max_request_tokens=...)`.
Tests (`tests/test_cb_end_to_end.py`): strict greedy token-for-token equivalence vs
`StandardBackend.generate` on tiny-Qwen3 float32 CPU for N concurrent submits (viable
only because step 8 fixed the stop contract), stop-mid-stream + max_tokens cases,
staggered arrivals, abort-under-load frees capacity, API-level SSE smoke over the CB
engine + over-cap 400, device-tolerance variant (logit allclose) for MPS.
**Done when:** suite green on CPU; manual smoke — `serve --engine batched` handles 3+
concurrent chat clients on the Mac; full repo suite green.

### Step 10 — Close-out: rough numbers + docs  [pair]
Rough sequential-vs-batched comparison with the existing `clients/bench.py` (formal
bench spec remains consciously deferred — note this in PLAN.md). Update PLAN.md Phase 2
Status line per CLAUDE.md's convention (done/open/deferrals, dated); add a short
postscript to `old_research_continuous_batching.md` recording where the integration deviated.
**Done when:** numbers recorded; Status line reflects reality.

### Step 11 — viz/ refresh  [Claude]
One pass at the end (as agreed): add the CB engine to the architecture diagram, fix/
extend trace harnesses broken by the refactors (`StandardBackend.forward/sample` moved
delegation, new engine subsystem), re-run trace generation per `viz/README.md`, refresh
the Roadmap tab (PLAN.md status lines) and re-check the CB-wiring tab (which mirrors
this plan and was updated when the plan landed).
**Done when:** harnesses run clean; tabs match the updated docs.

### Steps 12–13 (optional tail, post-integration) — logprobs, then stop strings
Logprobs: `TokenEvent` widening + both adapters; the shared sampler already returns
probs, so it lands on both engines. Stop strings: decoder-level backtracking in
`decoder.py` + adapters; zero scheduler coupling. Each is its own small session.

## Verification

- Every step ends with the full suite green: `python -m pytest tests/ -v` (plus the
  prototype's own suite staying green until the port is complete).
- The two-layer test strategy from the note: toy-stepper scheduler tests (fast, model-
  free) + tiny-Qwen3 equivalence (strict token equality only on float32 CPU greedy;
  logit-tolerance on MPS — different reduction order makes bitwise equality flaky).
- End-to-end: step 9's manual smoke with concurrent chat clients against
  `serve --engine batched`, and the API contract tests running over the real CB engine
  on the tiny runtime.

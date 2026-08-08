# Design note: CUDA graphs for steady-state decode (Phase 3)

**Status: implemented 2026-08-05, validated on the 5090 2026-08-07.**
Landed as designed (wrapper `engine/batching/graphs.py`, write-map padding
with the scratch position column, `BatchMeta.seed_kv_write_map`,
engine/CLI/bench wiring). The validation record is
`cuda-graphs-results.md`: every §7 gate cleared; §6's predictions stand
unedited below and are graded there (two missed, both conservatively).
First hardware contact also cashed in §4's cached-property hazard for
real — a device-compare `replace()` upstream dropped the seeded write map
and invalidated capture (fixed, `40fbcf9`). Background reading: the viz
CUDA-graphs chapter (`viz/index.html#/cudagraphs`) and its committed capture
record (`viz/captures/cudagraphs-2026-07-26/`); `step-profiling.md` for the
target.

## 1. The target

The decode floor is CPU dispatch, and it is the one Phase 3 sink still
standing. After the KV-write and finalize fixes, a 16-row decode step still
issues ~1900 CUDA API calls and holds at p50 ~10.5 ms with ~9 ms of fixed
per-step cost; at 1 row, 8.97 of 9.3 ms is the forward call returning before
any GPU wait. SDPA could not touch this (short_chat moved about a percent in
its A/B): fused kernels shrink the ride, and the ride was never the problem
in short-context decode. Graphs remove the toll.

The L4 capture session bounds the prize: the 2-layer test fixture's 110-kernel
decode step went 2066 → 134 µs under replay, with CPU submit flat at ~3.5 µs
regardless of kernel count. The engine's 28-layer step is the same picture at
~16× the node count.

## 2. The shape of the change

One new wrapper implementing `BatchedForwardFn`, sitting between the
scheduler and `ModelRuntime.forward_batched` (working name
`GraphedBatchedForward`; final name hers). The scheduler does not change at
all, the same altitude call as extracting `shaping.py::shape_step`:

```
scheduler.step()
  plan → shape_step → forward_fn(input_ids, meta, pool) → sample → finalize
                          │
              GraphedBatchedForward
                ├─ shape (B, W, KV) in the graph table?
                │    yes: validate host-side → copy_ prologue into the
                │         shape's static buffers → cudaGraphLaunch → logits
                │    no:  delegate to ModelRuntime.forward_batched (eager)
                └─ owns: graph table, per-shape static buffers,
                         one shared graph memory pool, hit/miss counters
```

Capture happens once, inside the existing warm-up sweep behind Ready
(`warmup.py`), which already visits every vocabulary shape. Serving never
captures; a shape outside the table runs eager forever, exactly as cold
shapes bypass warm cuDNN plans today.

## 3. Decisions

1. **Capture the model forward only; sampling stays eager (v1).** The
   sampler pipeline is per-row Python over `SamplingParams.processors`,
   dynamic control flow that capture cannot hold, and the batched finalize
   already collapsed its cost to one transfer per step. The flood is the
   28-layer forward; that is what the graph swallows. Pulling device-side
   sampling math into the graph is a v2 with its own RNG-capture homework,
   taken only if profiles later point at it.

2. **Decode-only capture set: the (B, 1, KV) shapes.** Batch buckets ×
   kv spans, ~80 graphs at the 0.6B baseline geometry (5 batch buckets ×
   16 kv spans; the exact count falls out of `BatchingConfig`). Prefill and
   mixed steps stay eager: they are compute-bound (the SDPA A/B is the
   evidence), and capturing all ~477 vocabulary shapes would multiply
   capture time and memory for steps graphs cannot help.

3. **The KV-scatter wrinkle resolves to write-map padding, and the code
   picked for us.** The chapter left (a) pad the write map vs (b) piecewise
   capture deliberately open. Reading the code closes it: the scatter runs
   per layer *inside* `forward_batched` (padded.py), so "leave the scatter
   eager" would mean restructuring the method template to stash 28 layers
   of K/V and defer their writes, more memory and more kernels in service
   of avoiding a padding trick. Instead: pad `kv_write_map` to the bucket's
   row count so its length is a constant of the shape (decode: exactly B
   entries). Filler entries need a harmless destination, which is decision 4.

4. **Scratch destination: one extra position column, not an extra slot.**
   The pool allocates `max_seq_len + 1` positions; padded map entries point
   at `pos = max_seq_len` (any slot). Gathers read `[:max_history_len]`,
   which is capped at `max_seq_len`, so the scratch column is never read.
   Cost at 0.6B geometry: ~2 MB. The alternative, a reserved scratch slot,
   costs a full slot (~470 MB at 4096 capacity) for the same guarantee.
   `FILLER_SPEC` semantics change from "writes nothing" to "writes garbage
   to the scratch column"; the mask story is untouched.

5. **One shared graph memory pool.** All captures pass the same
   `torch.cuda.graph(pool=...)` handle, largest batch first, so N graphs
   share one arena instead of owning N private pools (vLLM does the same,
   for the same reason). Decode activations are small; §6 predicts the bill.

6. **Capture rides the warm-up sweep, strictly after the eager warm.** Per
   decode shape: the sweep's existing dummy forward runs first (this is what
   compiles the cuDNN plan; capture must record the *warm* kernel choice,
   with the `sdpa_kernel` priority pin active inside the capture context),
   then a second all-filler forward is captured, instantiated against the
   shared pool, and verified with one replay. Prefill shapes keep their
   eager-warm only. Config: CUDA graphs require `shape_buckets` +
   `warmup_shapes`, validated the same way sdpa-without-buckets warns today.

7. **Static-buffer prologue replaces the per-step `.to()` moves.** Today the
   scheduler builds meta on CPU and the runtime `.to()`s fresh tensors each
   step: new addresses every step, which replay cannot follow. On a table
   hit the wrapper instead `copy_`s into the shape's static buffers:
   input_ids, slots, start_pos, num_new, positions, and the four padded
   write-map columns (~9 small H2D copies; packing them into one pinned
   staging buffer is a v2). On a miss, the eager path keeps its current
   behavior bit for bit.

8. **CLI: tri-state `--cuda-graphs` / `--no-cuda-graphs`, default on for
   CUDA when buckets + warm-up are on**, mirroring the attention/buckets
   flags. Bench gains a `[server] cuda_graphs` key; every pre-existing
   config gets pinned `false` so history re-runs reproduce what they
   measured. Mac/CPU/MPS never graph (CUDA-only API).

9. **Not `torch.compile(mode="reduce-overhead")`**, though it exists and
   would capture graphs for us. Hand capture is the point (PLAN.md: "the
   implementation detail is the lesson"), it keeps the write-map contract
   explicit, and it leaves `torch.compile` where PLAN.md has it: a separate
   Phase 3 experiment about kernel fusion. The two compose later if wanted
   (capture a compiled region); they are complements, not rivals.

## 4. Capture hazards the implementation will meet

Named now so they are recognized as expected, not debugged as surprises:

- **Hidden sync in validation.** `Qwen3._validate_batched` evaluates
  `(meta.num_new < 0).any()` as a host bool: a device sync every step today,
  and inside capture it is rule 1 (a question mid-flight) and invalidates
  the recording. Validation moves host-side into the wrapper, which holds
  the CPU meta anyway. Same for the per-row bounds loop at the top of
  `PaddedAttentionMethod.forward_batched`: pure host Python, so replay
  simply never runs it; the wrapper must do that check before replaying.
- **`kv_write_map` is a `cached_property` built from `meta.rows`.** At
  capture time it would bake capture-time addresses; at replay it never
  rebuilds. The capture-time meta must carry the padded map *as* the static
  buffers, pre-injected (a constructor/factory on `BatchMeta`, her call on
  the API; the frozen dataclass allows seeding the cache).
- **Values vs shapes.** Slots, positions, and map indices change every step
  and that is fine: graphs bake shapes and addresses, not values. Device-side
  gathers (`freqs_cis[positions]`, `layer_k[slots]`) read whatever the
  static buffers hold at replay. The discipline is only that every
  per-step tensor lives in a static buffer and every shape is bucket-fixed.
- **The pool must pre-exist capture** (it does: allocated at engine build),
  and the capture order must follow weights + pool so the shared graph pool
  sizes against the true free-memory watermark.
- **Tripwire, extended.** The SDPA lesson: silent fallbacks need tests that
  assert the fast path actually ran. The wrapper keeps hit/miss counters
  surfaced through StepStats, and a test asserts a decode-shaped step
  replays (and that replay logits match eager bit for bit; same kernels,
  same order, so exact equality is the expectation, not a tolerance).

## 5. Phase 4 interaction

Paged KV (Phase 4) reshapes the read path into block tables, which
invalidates the captured recordings but not the design: block-table indices
are also just values in static buffers, and the wrapper's table/dispatch/
prologue survive a re-capture on the new geometry. Keeping the wrapper thin
(no scheduler knowledge, no model knowledge beyond `BatchedForwardFn`) is
what makes that a re-capture, not a rewrite.

## 6. Predictions on record

Graded after the 5090 runs; wrong predictions get kept, not edited.

- `profile_step` decode recheck: ~1900 API calls/step at 16 rows drops
  below ~100 (sampling + prologue copies); the 1-row step falls from
  9.3 ms to ≤ 2 ms; the 16-row step p50 from ~10.5 ms to ~6-7 ms
  (GPU-busy-bound, per step-profiling's ~5.9 ms busy figure).
- Bench, 0.6B: short_chat c=16 aggregate 1469 → 1900-2300 tok/s
  (+30-55%). long_context within noise of the current default (compute-
  bound; SDPA already owns it). The wide-slots fixed term collapses; the
  per-row slope (already ~0.04 ms/row) barely moves.
- Hit rate: ≥ 95% of decode steps land on captured shapes under short_chat
  (the buckets already quantize arrivals); reported by the new counters.
- Bills: capture + instantiate adds roughly 40-90 s on top of the 102 s
  warm-up (~80 shapes); shared-pool memory under ~1 GB. Gates: if capture
  exceeds ~3 s/shape or the pool ~2 GB, scope the captured kv spans down to
  the workload's p99 span instead of all 16.

## 7. A/B protocol

`ab_5090_cudagraphs.toml`, two arms: the current CUDA default
(sdpa + buckets + warm-up) vs the same + graphs. Workloads short_chat, code,
multi_turn, long_context; same seeds and metrics as the shape-buckets A/B,
plus the `profile_step` recheck and the hit-rate/step-time stats. Success
gate: short_chat aggregate +20% or better with long_context non-regressing
(> −3%). Correctness gate: greedy token-for-token equivalence vs eager on
fixed seeds, plus the bit-exact replay-logits test.

## 8. Division of labor

Revised at implementation time: the author delegated this build (the viz
chapter served the learning goal), and Claude implemented all of it in
review-sized chunks — wrapper, then write-map padding, then the `BatchMeta`
API, then assembly — with the suite green after each. The 5090 runs and the
results write-up (`cuda-graphs-results.md` will be the record) remain.

## 9. Open questions (hers to call)

1. Scratch destination: the extra position column (recommended, §3.4) vs a
   reserved slot vs anything cleverer.
2. Captured kv-span set: all spans, or workload-informed subset from day one.
3. The `BatchMeta` API for the injected padded write map.
4. v2 candidates, deliberately deferred: sampling inside the graph, the
   packed one-copy prologue, and whether the lm_head mid-prefill-row skip
   (model.py TODO) lands before capture so the recordings include it.

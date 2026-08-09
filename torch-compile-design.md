# Design note: torch.compile on the batched forward (Phase 3)

**Status: proposed 2026-08-08, implemented same day, awaiting the 5090
A/B.** Landed as designed in three chunks (hoists + `torch_compile`
config/wiring; the §3.2 strategy knob with per-step dim marking in the
runtime front; CLI/bench assembly with `ab_5090_compile{,_longctx}.toml`),
suite green after each; the explain-day findings are §7. The last Phase 3
optimization before the H100 day. Background reading:
`cuda-graphs-results.md` for the target numbers (decode is now GPU-bound),
`step-profiling.md` for the kernel census, `cuda-graphs-design.md` §3.9 for
why graphs and compile are complements. The A/B protocol is §7; §6's
predictions get graded after the 5090 runs, wrong ones kept, not edited.

## 1. The target

CUDA graphs removed the dispatch toll: `fwd_call` is 0.06 ms at every
occupancy and a 16-row decode step replays at p50 5.93 ms, which is
step-profiling's GPU-busy figure. What remains is the GPU work itself, and
it is fatter than the physics requires. A decode step must at minimum read
every weight once: 0.6B in bf16 is ~1.2 GB, roughly 0.8-0.9 ms at the
5090's achievable ~1.4-1.5 TB/s. The step spends 5.93 ms at 16 rows and
3.63 ms at 1 row against that same floor.

Where the rest goes: the recording holds ~1750 kernels per step. The seven
matmuls per layer are cuBLAS and already efficient; everything between
them (two RMSNorms plus q/k-norm, the RoPE gather and rotation, two
residual adds, the SwiGLU elementwise, the mask/gather bookkeeping) runs
as dozens of little kernels per layer, each one reading its input from
device memory and writing its output back so the next one can read it
again. At 1 row those kernels are microseconds each, so their fixed
per-kernel costs and redundant round-trips dominate the step.

`torch.compile` attacks exactly this: Dynamo traces the forward into one
op graph, Inductor fuses the between-matmul tissue into a few Triton
kernels per layer that keep intermediates in registers. Fewer kernels,
fewer bytes. The SDPA attend stays an opaque cuDNN call (Inductor does not
generate attention), and the matmuls stay cuBLAS; this is a play for the
tissue, so the win is bounded and §6 sizes it soberly. Graphs then make
the fused kernels free to launch: capture records whatever the compiled
forward runs, so the two compose by construction.

## 2. The shape of the change

Compile slots inside the existing wrapper stack, below the graphs wrapper
and below the runtime front's host work:

```
scheduler.step()
  plan → shape_step → forward_fn(input_ids, meta, pool) → sample → finalize
                          │
              GraphedBatchedForward            (unchanged)
                ├─ table hit: marshal → replay
                └─ miss: inner ─→ ModelRuntime.forward_batched
                                    ├─ host: .to() moves, validation
                                    │        hoists, force kv_write_map
                                    └─ compiled(model forward)   ← new
```

The compiled region is the model's batched forward: embedding → mask →
28 layers → last-token gather → lm_head. Host-side Python (validation
over `meta.rows`, the write-map derivation) moves out of the traced
region into the runtime front, mirroring what the graphs wrapper already
did for the replay path. The scheduler, the graphs wrapper, and the
sequential engine do not change.

Warm-up order behind Ready becomes: eager/compile warm sweep (paying
Inductor compiles and cuDNN plans) → graph capture (recording the fused
kernels). Serving never compiles; a shape that would need a new artifact
is a bug the tripwire counter catches, not a stall.

## 3. Decisions

1. **Compile the model forward only, with the host work hoisted out, and
   `fullgraph=True`.** Two pieces of the current forward cannot be traced
   without poisoning the cache: the `_validate_batched` /
   `padded.forward_batched` bounds loops iterate `meta.rows` (per-step
   Python ints, so Dynamo would guard on their concrete values and
   recompile every step), and `meta.kv_write_map` is a `cached_property`
   whose derivation is a host loop ending in a `.to()`. Both hoist to the
   runtime front: validate there (host-side, as the graphs wrapper
   already does for replay), and force the map before entering the
   compiled region so tracing sees cached tensors, not construction.
   With those out, the forward is pure tensor dataflow and
   `fullgraph=True` should hold; it is also the tripwire (SDPA lesson): a
   future graph break fails loudly instead of silently fragmenting.

2. **Dynamic dimensions, with the artifact count as the experiment.** A
   decode shape is (rows, kv span): 5 × 16 = 80 on the short geometry,
   3 × 40 = 120 on longctx, and static compile per shape is minutes of
   Inductor time for one program. The A/B compares two strategies:
   (i) *fully dynamic*: `mark_dynamic` on the batch and kv dimensions,
   expecting ~2-3 artifacts to cover the whole grid (size 1 specializes);
   (ii) *per batch bucket*: kv dynamic only, one artifact per batch
   bucket (5 short / 3 longctx), keeping row-count constants baked into
   the fused kernels. (ii) is attractive because the kv span mostly feeds
   the opaque cuDNN call and the gather bound, while the row count is
   what the fused kernels iterate over. Fully static (80+) is ruled out
   by the compile bill. One wrinkle either way: the shape carriers
   `num_new_max` / `max_history_len` are Python ints on `BatchMeta`, and
   the explain day (§7) confirms they go symbolic under automatic
   dynamic rather than specializing; if not, they get derived from
   tensor shapes inside the region. *Implementation note (2026-08-08):
   the batch dims take a hard `mark_dynamic`; the width dims take the
   soft `maybe_mark_dynamic`, because `num_new_max` (the int) burns the
   width into the mask arange, specializing the dim, and torch
   hard-errors when a hard mark's promise breaks. Width goes symbolic
   when automatic dynamic promotes the int on its second value.*

3. **One compiled forward serves every shape, prefill included.** The
   width dimension is just another dynamic dim ({1} ∪ prefill menu), so
   prefill and mixed steps run the same compiled artifacts eagerly (no
   graphs, as today). Prefill is matmul-bound, so the expected win is
   small, but it is the only lever on the numbers graphs deliberately
   left alone (TTFT, the prefill-bound step p99s), and it costs nothing
   extra to include. Attribution comes from the metrics: eITL p50 reads
   the decode effect, TTFT/p99 the prefill effect.

4. **Graphs capture the compiled forward.** The compiled callable becomes
   `GraphedBatchedForward`'s inner; the warm-up sweep drives it (so
   Inductor compiles land behind Ready, per shape family rather than per
   shape), and capture then records the fused kernels. The per-shape
   side-stream warm before capture (`_WARM_ITERS`) already runs the
   inner, which now also guarantees the shape's artifact is built before
   recording starts. Everything `cuda-graphs-design.md` §4 says about
   capture safety applies to the compiled code unchanged: no mid-forward
   allocations, no H2D copies, no syncs. `40fbcf9` is the cautionary
   tale, and the bit-exact replay test is still the tripwire.

5. **Default Inductor mode; not `reduce-overhead`, not `max-autotune`
   (v1).** `reduce-overhead` would apply its own CUDA graphs per
   compiled fragment; ours are whole-forward, wrap the write-map
   contract explicitly, and already exist. `max-autotune` buys matmul
   template search at a multiple of the compile bill; it is a
   measurement for another day, noted in §9.

6. **CLI: tri-state `--torch-compile` / `--no-torch-compile`, default
   OFF until the A/B.** Config field `torch_compile` on `BatchingConfig`
   (decided 2026-08-08), requiring `warmup_shapes` the same way
   `cuda_graphs` does: the bounded vocabulary is what makes the compile
   bill payable behind Ready. Bench gains a `[server] torch_compile`
   key. This time the default waits for the numbers: the SDPA A/B
   reversed the expected outcome once, and graphs'
   default-on-before-validation was flagged as the arguable call.

7. **The compile bill is measured cold and warm.** Inductor caches
   compiled artifacts on disk; the A/B records Ready time on a cold
   cache and again warm, because the steady-state answer (does the box
   pay this every boot?) decides whether the bill matters at all.

## 4. Hazards the implementation will meet

Named now so they are recognized as expected, not debugged as surprises:

- **Guard explosion on `meta.rows`.** Any traced read of the rows list
  burns guards on that step's concrete (slot, start, num_new) values:
  the artifact "works" and then recompiles on every step, a silent
  throughput disaster rather than an error. This is why decision 1 hoists
  both loops rather than letting Dynamo trace through them. The §7
  recompile counter exists to prove it stayed fixed.
- **The `sdpa_kernel` priority pin under tracing.** The pin is a context
  manager inside `_attend_batched`. If Dynamo traces it cleanly (torch
  2.10 should), fine; if it graph-breaks, the pin hoists to wrap the
  compiled call site, since backend selection reads dispatcher state at
  run time. Either way the existing kernel-actually-ran test must pass
  under compile: a fused attend that silently fell back to math would be
  the SDPA failure all over again, one level down.
- **Numerics move; the gates move with them.** Fused kernels reorder
  float math, so compiled logits differ from eager in the low bits by
  construction, and "bit-exact vs eager" is no longer a meaningful bar.
  The correctness gates become: greedy token-for-token equivalence vs
  eager (argmax shrugs off low-bit drift), a logits-tolerance check, and
  the graph-replay test tightens to bit-exact vs *compiled* eager (same
  recorded kernels, so exact equality remains the expectation there).
- **In-place pool writes under functionalization.** The KV scatter
  mutates pool views that are graph inputs. Inductor supports input
  mutation, but a regression here would show up as an extra copy of the
  pool slice per layer; the kernel-count numbers in §7 would catch it.
- **`inference_mode` stays outside.** The runtime front's decorator
  wraps the compiled call; compiling under it is supported, but the
  region itself should not toggle modes.
- **The recompile limit is a hard error under fullgraph (found
  2026-08-08).** Dynamo caps artifacts per code object at 8 by default,
  and with `fullgraph=True` hitting the cap raises instead of falling
  back to eager: a serve-time crash, since the batch-bucket strategy
  alone wants one artifact per bucket plus kv/width promotions.
  `enable_torch_compile` sizes `cache_size_limit` to 64.

## 5. Phase 4 interaction

Paged KV reshapes the gather into block tables, which invalidates the
compiled artifacts exactly as it invalidates the graph recordings: block
indices are tensor inputs like everything else, so the compile boundary,
the dynamic-dim strategy, and the warm-up wiring all survive as a
recompile, not a redesign. The one watch item is that block-table
indexing must stay traceable dataflow (no host loops over blocks), which
Phase 4 wants anyway for the same reason the write map became tensors.

## 6. Predictions on record

Graded after the 5090 runs; wrong predictions get kept, not edited.

1. Explain day: after the two hoists, the batched forward traces to one
   graph and `fullgraph=True` holds; the only plausible break is the
   `sdpa_kernel` context.
2. Artifact counts: fully-dynamic serves the whole decode grid plus
   prefill widths with ~2-4 artifacts; per-bucket with ~6-8 (5 decode +
   prefill families). Zero recompiles during A/B traffic, per the
   counter.
3. Kernel count inside a 16-row decode step: ~1750 → 400-700
   (28 × (7 matmuls + a handful of fused kernels) + head/tail).
4. Step times: 16-row decode p50 5.93 → 4.6-5.4 ms (8-25% cut); 1-row
   probe 3.63 → 2.4-3.1 ms (a larger relative cut: small-kernel
   overheads dominate there, and fusion removes kernels outright).
5. Bench, 0.6B: short_chat c=16 2466 → 2650-3050 tok/s (+8-24%);
   longctx c=1 175 → 190-215 (its decode tail mirrors the 1-2 row step);
   TTFT p50 and the prefill-bound step p99s move 0-10% (the prefill
   rider's territory).
6. The artifact experiment: per-bucket beats fully-dynamic by under 5%
   on decode step time, i.e. within or near noise; fewer artifacts wins
   the tie.
7. Bills: cold compile adds 40-150 s behind Ready for fully-dynamic,
   roughly 1.5-2x that for per-bucket; a warm disk cache brings either
   under ~20 s. No measurable steady-state memory growth.
8. Correctness: greedy token streams identical vs eager on the live
   check; replay bit-exact vs compiled-eager; the cuDNN tripwire passes
   under compile.

## 7. A/B protocol

`ab_5090_compile.toml` (16×4096 geometry: short_chat, code, multi_turn)
plus `ab_5090_compile_longctx.toml` (4×10240: long_context), three arms
each: the current CUDA default (sdpa + buckets + warm-up + graphs), the
same + compile fully-dynamic, the same + compile per-bucket. Same seeds
and metrics as the graphs A/B, plus: the `profile_step` recheck (step
times and kernel counts per arm), Ready time cold-cache and warm-cache,
and a recompile counter asserted flat across each run (Dynamo's compile
events; any recompile after warm-up fails the run).

Before any of that, the explain day: `torch._dynamo.explain()` on the
batched forward, break count and guard census recorded. Dynamo tracing is
device-independent, so this starts on the Mac; only the Inductor CUDA
artifacts and timings need the box.

**Explain-day findings (2026-08-08; torch 2.10.0, CPU, the tests' tiny
2-layer fixture).** The current forward traces to 10 graphs with 9
breaks, and the root cause is not the `sdpa_kernel` pin: on Python 3.11
`functools.cached_property.__get__` takes an `RLock` when the value is
uncached, so deriving `kv_write_map` under trace breaks the graph at the
map read in `padded.forward_batched`, once per layer, and the resulting
fragments guard on per-step row values (+2 recompiles across four
start_pos values at a fixed shape, +4 across three kv spans). With the
§3.1 hoists simulated (bounds loop out of the traced region, map
pre-forced so the property is a cache hit), `fullgraph=True` holds: one
graph, zero breaks, the `sdpa_kernel` context traces clean, and
automatic dynamic covered a 16-decode-shape + 6-prefill-shape sweep
with **5 artifacts** (the first call and the B=1 row specialize before
the batch/kv/width dims go symbolic; every later shape hits). Traced
outputs were bit-identical to eager on CPU, which validates trace
fidelity only; Inductor numerics remain a box question. Net: prediction
1's break candidate was wrong (the pin is fine, the property lock was
the tripwire), the hoists cure it, and prediction 2's artifact count
looks right.

Success gate: short_chat c=16 aggregate +8% or better, nothing regressing
past −3%, warm-cache Ready bill under +30 s. Correctness gate: the §4
reworded set (greedy equivalence vs eager, logits tolerance, replay
bit-exact vs compiled-eager, cuDNN kernel tripwire). If compile clears
the gates it joins the CUDA serve default as its fourth piece and every
pre-existing config gets pinned `compile = false`; if not, the results
doc records why and the flag stays opt-in.

## 8. Division of labor

Delegated, like graphs: Claude implements in review-sized chunks (hoists
+ compile wiring, then the dynamic-dim strategy, then warm-up/CLI/bench
assembly), suite green after each, with the explain-day scouting first.
The 5090 runs go through the agent-protocol pattern (instructions file
created for the run, deleted after, per house precedent); the results
write-up (`torch-compile-results.md` will be the record) happens back
home from the numbers.

## 9. Open questions (called 2026-08-08)

1. Config/flag naming and hoist placement: **decided**, `torch_compile`
   on `BatchingConfig`, hoisted validation in the runtime front (§2).
2. The default artifact strategy if the experiment lands within noise:
   **decided**, fully-dynamic (fewer artifacts, smaller bill).
3. The lm_head mid-prefill-row skip (model.py TODO): **decided,
   dropped**. The gather-then-project order already avoids the
   (B, S, vocab) materialization; what the skip would add is bounded by
   the projection being one ~311 MB weight read regardless of row count
   (under 1%, and only on all-mid-prefill steps), and it would introduce
   a data-dependent shape exactly where the shape vocabulary, capture,
   and compile all want uniformity. The model.py docstring records the
   decision, so the A/B baseline is stable.
4. v2 candidates: **confirmed deferred**: `max-autotune`, compiling the
   sequential engine's forward, sampling inside the compiled region, and
   cache persistence strategy across box reprovisions.

# torch.compile on the 5090: A/B results (2026-08-08/09)

The Phase-3 compile implementation (landed 2026-08-08 per
`torch-compile-design.md`) measured against the serve default (sdpa +
buckets + warm-up + graphs), with the artifact strategy as the secondary
experiment. Verdict up front: **fusion beat every performance prediction —
short_chat c=16 2461 → 3683 tok/s (+49.6%), long_context c=1 179 → 294
(+64%), 16-row decode step 5.93 → 4.0 ms, kernels/step 1859 → 218, every
cell in both configs positive — but only after the round found and fixed
four implementation bugs the CPU suite was structurally blind to, three of
them caught by the protocol's own tripwires. Compile is now the fifth
piece of the CUDA serve default (strategy dynamic), with the greedy
bf16-tie drift vs eager reviewed and accepted.** Environment: torch
2.10.0+cu128, driver 580.159.03→.173.02 (matched, post-reboot), sm_120,
0.6B bf16. Runs: `bench/history/2026-08-08T222750_9b49283_ab-5090-compile`
(+ its `agent-summary.md`, the raw record, with a 2026-08-09 addendum) and
`…T224249_9b49283_ab-5090-compile-longctx`; the blocked first attempt is
`…T204000_c4c7f1e_compile-5090-blocked/`.

## 1. The blocked round: view mutations vs functionalization

The as-designed implementation was unusable on the real backend and the
suite could not know: the compile tests run Dynamo with `backend="eager"`,
and AOTAutograd functionalization + Inductor lowering only run on the
real one. First contact: the §2 step profile OOM'd at the 48-slot
geometry on a 10.5 GiB Inductor-allocated buffer — a full copy of the K
pool — and at serve geometry a compiled decode step ran **213 ms vs 9 ms
eager** at 24.5 GiB peak.

Root cause, isolated to a 6-line repro (blocked-round dir,
`scripts/repro_view_mutation.py`): the pool was one stacked
`(L, B, S+1, G, D)` tensor per K/V, and each layer's scatter wrote through
the `pool.k[i]` select view. AOTAutograd keeps mutations of graph
*inputs* in place, but functionalizes mutations through *views* of inputs
into full-base rebuilds that Inductor's re-inplacing does not recover —
chained across 28 layers. The design note's §4 hazard entry predicted "an
extra copy of the pool slice per layer" caught by kernel counts; reality
was pool-*scale* copies caught by a 23x step time and an OOM. Identical
scatter code against per-layer tensors compiles to an in-place scatter
(repro: 104.7 → 0.13 ms). Fix (`ab4f438`): `PaddedKVPool` holds per-layer
tensors — the shape every production static-KV cache uses (vLLM,
gpt-fast, HF `StaticCache`), for exactly this reason. `pool.layer(i)`
survived unchanged; a real-Inductor CUDA test (`TestInductorCUDA`:
numerics, writes land, no pool-scale allocations) now tripwires the
class.

## 2. Two guard leaks, caught by the recompile counter

The §3 tripwire (zero recompiles after Ready, `TORCH_LOGS=recompiles`)
failed twice before it passed, each failure a lesson in what "warm-up
covers the guard set" actually requires:

- **Dispatch keys (`1049827`).** Every artifact the sweep built was
  rejected by the first live request: traffic metas become *inference
  tensors* inside the runtime front's `@inference_mode` device move,
  while warm-up metas built on-device skipped the move and kept
  `ADInplaceOrView` — and Dynamo guards on the dispatch key set. Warm-up
  now builds CPU tensors exactly like the scheduler and takes the same
  move (the front carries the seeded map through its `replace`); capture
  buffers allocate under `inference_mode` to match.
- **Write-map length coverage (`9b49283`).** One recompile survived: a
  solo prefill chunk (B=1, width 128, 38 real tokens) rejected the
  batch-1 family's artifact, which had specialized at map length 1
  (torch's 0/1 rule). Every (batch, width) family needs both the
  length-1-specialized and the length≥2-symbolic artifact, so the sweep
  alternates seeded lengths 1 / max(2, batch) along each family's kv
  sweep — zero extra forwards. (This is also why warm-up seeds
  scratch-parked write maps at all: the fillers' natural length-0 maps
  cover nothing.)

On the final sha: zero post-capture recompiles in every compiled arm of
both configs, and decode-graph replay at **100.0%** of pure-decode steps
in all 24 cells — capture composes with compile as designed, and is
*faster* under it (80 shapes: 3.4 → 1.7 s; fewer kernels to record).

## 3. The A/B

Medians of 3 repeats, zero validity warnings. Baseline arms are the
cross-product's deliberate repeat-noise pair and reproduce the graphs
round (2461/2469 vs 2466 recorded).

| cell | baseline | +compile dynamic | +compile batch-bucket |
|---|---:|---:|---:|
| short_chat c=4 tok/s | 810.7 | 1307.6 (+61%) | 1370.3 (+69%) |
| short_chat c=16 tok/s | 2461.2 | **3682.6 (+49.6%)** | 3800.1 (+54%) |
| code c=8 tok/s | 1365.6 | 2097.1 (+54%) | 2167.3 (+59%) |
| multi_turn c=8 tok/s | 1149.1 | 1698.3 (+48%) | 1756.7 (+53%) |
| long_context c=1 | 179.4 | **294.2 (+64%)** | — |
| long_context c=2 / c=4 | 207.6 / 207.7 | 285.1 / 254.7 (+37/+23%) | — |
| 16-row decode step p50 | 5.9 ms | 4.0 | 3.8 |
| TTFT p50 (short c=16) | 66.2 ms | 40.2 | 51.1 |

The step-profile recheck: kernels/step at 16 rows 1859 → **218** (under
the predicted 400-700), the census clean — cuBLAS matmuls + softmax +
microsecond fused Triton kernels. The compiled `fwd_call` is ~3.6 ms of
Dynamo guard evaluation + launches, which graphs then erase; eITL p50
lands at 2.3-4.0 ms depending on occupancy. TTFT improves 10-39% — the
prefill rider (predicted 0-10%) was undersold, since prefill and mixed
steps run the same fused artifacts eagerly. Strategy experiment:
batch-bucket is consistently ~3% ahead at c=16 (baked row constants),
within the predicted 5% band; dynamic wins on half the artifacts (9 vs
21) and −140 s of cold Ready, and is the default.

## 4. The bills, and the fourth bug

Cold (fresh Inductor cache): baseline Ready 78.1 s → compile-dynamic
225.6 s (+147.5, the top edge of the predicted 40-150) → batch-bucket
365.3 s (1.95x, as predicted). Warm was the failed gate — **+65.6 s
against the < +30 s bar** — until investigating "how much would cache
persistence buy" found it was self-inflicted: the sdpa cuDNN priority
pin, traced into the graph, plants `_backend_from_string`, and that one
unserializable call made AOTAutograd **bypass** its cache entirely
(`autograd_cache_bypass: 9`, `fxgraph_cache_miss: 9` on a warm disk
cache) — every boot re-ran full Inductor codegen; only the Triton binary
cache ever hit.

Fix (`a30f2ee`): the pin hoisted to `AttentionMethod.execution_context()`,
entered by the forward's entry points and never traced — design §4's own
contingency, executed for cacheability rather than graph breaks.
Artifacts still trace inside the context, so cuDNN stays baked at trace
time; the kernel-ran tripwire and `TestExecutionContext` (entry-point
wiring + no dispatcher machinery in traced graphs) pin it. After:
`autograd_cache_hit: 9 / fxgraph_cache_hit: 9`, warm respawn **98.9 s →
+21.0 s, gate passes**. The residual ~21 s is Dynamo re-tracing + guard
building + cache loads — not disk-cacheable on this torch. First boot
after a code change still pays the cold bill (new keys, once per
change). And since systemd empties /tmp at boot, serve now parks the
cache in `~/.cache/cantollm/inductor` (`8fc8cd9`) so reboots keep the
warm bill; the §9 mega-cache item stays deferred, relevant only to
reprovisioned boxes.

## 5. Correctness: drift, quantified and accepted

Replay is bit-exact by the suite and 100% engaged; compiled-vs-eager
logits are well inside tolerance (max |Δ| 0.50 / mean 0.037 on a
max-|logit| scale of 15.2, realistic 16-row decode state). But **greedy
streams are not token-identical vs eager: 5 of 7 live-check prompts
diverge** (both compiled arms; they agree with each other on 5/7). Every
flipped row sits at a top-2 margin of exactly 0.0625 — one bf16 ulp —
and the divergent continuations are coherent alternates. Fused kernels
reorder float math; at a one-ulp tie the argmax flips and greedy decoding
amplifies one token into a different, equally-sensible continuation. §4
predicted the low-bit drift but expected argmax to shrug it off; at bf16
ties it does not. Neither path is ground truth at that precision — both
bf16 orderings sit further from the fp32 answer than from each other —
and the author accepted the drift for the serve default (2026-08-09);
`--no-torch-compile` remains the opt-out for bit-stable-vs-eager output.

## 6. §6 predictions, graded (kept, not edited)

1. Fullgraph holds, sdpa pin traces clean: **confirmed** — and tracing
   clean turned out to be the trap; the pin belonged outside anyway (§4).
2. ~2-4 artifacts dynamic / ~6-8 bucket, zero traffic recompiles:
   **half wrong, instructively** — 9 / 21 (the length-pair seeding the
   zero-recompile goal itself demanded roughly doubles lineages), and
   zero-after-Ready held only on the third sha.
3. Kernels/step ~1750 → 400-700: **beaten** (218).
4. 16-row decode 5.93 → 4.6-5.4, 1-row 3.63 → 2.4-3.1: **beaten**
   (4.0 engine-side; longctx c=1 decode eITL p50 2.3 ms).
5. short_chat c=16 +8-24%, longctx c=1 190-215, TTFT 0-10%: **exceeded
   on all three** (+49.6%; 294; TTFT −10-39%).
6. Strategies within 5%, fewer artifacts wins the tie: **confirmed**,
   though batch-bucket is consistently (not coin-flip) the faster one.
7. Cold 40-150 s / bucket 1.5-2x / warm under ~20 s: **confirmed /
   confirmed / wrong-then-fixed** — warm was +65.6 s from the cache
   bypass; +21.0 s after the hoist, close to the prediction it was
   supposed to validate.
8. Greedy identical, replay bit-exact, cuDNN tripwire: **missed / 
   confirmed / confirmed** — the greedy miss quantified as one-ulp tie
   drift, accepted (§5).

## 7. Decisions and the residue

- **`torch_compile` is default-on for CUDA** (`8b05110`), the fifth
  piece of the serve default; strategy `dynamic`; pre-compile-round A/B
  configs pinned `torch_compile = false`. Mac/CPU stays eager.
- The compile boundary survives Phase 4 as designed: paged block tables
  are tensor inputs like everything else, and the per-layer pool is
  already the shape a paged pool takes.
- Deferred, unchanged from §9: `max-autotune`, compiling the sequential
  engine, sampling inside the region, mega-cache for reprovisioned
  boxes. New residue for some future round: the ~21 s warm Dynamo
  re-trace (experimental precompile/guard-serialization could kill it;
  not worth the churn today), and a quality-eval pass over the drift if
  it ever needs more than the one-ulp analysis.
- The meta-lesson of the round, for the walk-through: **every one of the
  four bugs was invisible to correctness tests and visible to a
  tripwire** — the step profile (pool copies), the recompile counter
  (two guard leaks), and the cache-hit counters (the traced pin). The
  SDPA round's "a fast path that can fall back silently needs a counter
  that proves it ran" generalizes: a cache that can bypass silently
  needs one too.

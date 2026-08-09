# 5090 validation summary: torch.compile (Phase 3) — 2026-08-08

Agent run log for the CUDA-TEST-AGENT-INST.md protocol, graded against
torch-compile-design.md §6/§7. Covers this run dir and its longctx sibling
`2026-08-08T224249_9b49283_ab-5090-compile-longctx`. The
`torch-compile-results.md` write-up happens back home from these numbers.
Environment: torch 2.10.0+cu128, driver 580.173.02 (matched stack after
the post-graphs-round reboot), sm_120.

**Headline: fusion delivers far past every performance prediction —
short_chat c=16 +49.6%, long_context c=1 +64%, 16-row decode step
5.93 → 4.0 ms, kernels/step 1859 → 218 — but only after TWO
implementation bugs that the protocol's own tripwires caught (both fixed
and committed this round), and the warm-cache Ready bill gate FAILS:
+65.6 s against the < +30 s gate. Perf gates pass, bill gate does not;
default-flip is the author's call.**

## The round's story: blocked, fixed, fixed again, then clean

This is the third A/B attempt of the day, on the third sha. The record
of the first, blocked attempt is
`bench/history/2026-08-08T204000_c4c7f1e_compile-5090-blocked/`.

1. **`ab4f438` — per-layer KV pool.** The as-designed stacked pool made
   every layer's KV scatter a mutation through a `pool.k[i]` select view
   of a graph input; AOTAutograd functionalizes view-of-input mutations
   into pool-scale rebuild chains (design §4's hazard, in its worst
   form): 213 ms compiled decode steps at serve geometry, OOM at the
   48-slot probe geometry. Per-layer tensors make the scatter a direct
   input mutation, kept in place. A standalone 6-line repro isolates it
   (blocked-round dir, `scripts/repro_view_mutation.py`), and a new
   CUDA-only real-Inductor test (`TestInductorCUDA`) now tripwires it —
   the CPU compile tests run `backend="eager"` and are structurally
   blind to functionalization.
2. **`1049827` — dispatch-key guard leak.** The first rerun tripped the
   §3 recompile counter: every artifact the warm-up sweep built was
   rejected by the first live request with `dispatch key set mismatch
   (ADInplaceOrView)`. Traffic metas become *inference tensors* inside
   the runtime front's `@inference_mode` device move; warm-up metas
   built on-device skipped that move and kept normal keys. Fix: warm-up
   builds CPU tensors exactly like the scheduler and takes the same
   move (the front now carries the seeded map through its `replace`),
   and capture's static buffers are allocated under `inference_mode`.
3. **`9b49283` — artifact coverage for length-1 vs length-N maps.** One
   recompile still survived: a solo prefill chunk (B=1, width 128, 38
   real tokens) rejected the batch-1 family's artifact, which had
   specialized at map length 1 (torch's 0/1 rule). Every (batch, width)
   family needs both the length-1-specialized and the length≥2-symbolic
   artifact; the sweep now alternates seeded lengths 1 / max(2, batch)
   along each family's kv sweep — zero extra forwards.

Both A/B configs then ran clean end to end on `9b49283` with a cold
Inductor cache (`rm -rf /tmp/torchinductor_$USER` first) and
`TORCH_LOGS=recompiles` asserted flat after capture.

## Step 1 — suite

457 passed, 3 skipped (MPS-only), including `TestCaptureReplayCUDA`
(replay bit-exact vs eager, `torch.equal`) and the new
`TestInductorCUDA` real-backend tripwire (compiled ≈ eager numerics,
pool writes land, no pool-scale allocations). `fullgraph=True` raised
nowhere on this box.

## Step 2 — profile_step recheck (probe, 0.6B, attention "padded")

Eager arm (unmodified script) reproduces step-profiling.md: 1-row
9.44 ms/step (fwd_call 9.05), 16-row 13.50 ms, 1859 cudaLaunchKernel
calls/step. Compile arm (scratch copy per protocol: 48-slot
`default_shape_buckets`, `warmup_shapes=True, torch_compile=True,
cuda_graphs=False`, dynamic):

| rows | total ms (eager → compiled) | fwd_call ms | fwd_sync ms |
|-----:|----------------------------:|------------:|------------:|
| 1    | 8.93 → 3.76                 | 8.61 → 3.43 | 0.19 → 0.19 |
| 8    | 9.64 → 4.56                 | 8.88 → 3.63 | 0.25 → 0.39 |
| 16   | 13.33 → 6.33                | 8.90 → 3.58 | 3.52 → 1.83 |
| 32   | 23.78 → 10.18               | 11.53 → 3.69 | 10.40 → 4.59 |
| 48   | 38.29 → 17.35               | 18.98 → 3.70 | 16.60 → 10.91 |

cudaLaunchKernel calls/step at 16 rows: **1859 → 218** (prediction 3
said 400-700 — beaten). The compiled `fwd_call` is ~3.6 ms flat: Dynamo
guard evaluation plus ~600 launches — the number the protocol asked
for, and the residual CPU cost graphs then remove. Census is clean:
cuBLAS matmuls + softmax + microsecond-scale fused Triton kernels, no
pool-scale copies (the first attempt's census — ~13 Triton kernels at
14.5-17 ms each rebuilding the pool — is in the blocked-round dir).
The first attempt's 48-slot OOM did not recur (peak fits comfortably).

## Step 3 — the A/B (this run dir + longctx sibling)

All arms sdpa + buckets + warm-up + graphs; compile and its strategy
flip. Medians of 3 repeats; zero validity warnings in any cell.

Short config (16×4096; baseline arms are the repeat-noise pair and
reproduce the graphs round: 2461/2469 vs 2466 recorded):

| cell | baseline | +compile dynamic | +compile batch-bucket |
|---|---:|---:|---:|
| short_chat c=4 tok/s | 810.7 | 1307.6 (+61%) | 1370.3 (+69%) |
| short_chat c=16 tok/s | 2461.2 | **3682.6 (+49.6%)** | 3800.1 (+54%) |
| code c=8 tok/s | 1365.6 | 2097.1 (+54%) | 2167.3 (+59%) |
| multi_turn c=8 tok/s | 1149.1 | 1698.3 (+48%) | 1756.7 (+53%) |
| 16-row decode step p50/p99 | 5.9/25.3 ms | 4.0/14.8 | 3.8/14.0 |
| TTFT p50 (c=16) | 66.2 ms | 40.2 | 51.1 |

Long-context config (4×10240, dynamic only):

| cell | baseline | +compile | Δ |
|---|---:|---:|---:|
| c=1 | 179.4 | **294.2** | +64% |
| c=2 | 207.6 | 285.1 | +37% |
| c=4 n=16 | 207.7 | 254.7 | +23% |
| c=4 n=32 | 202.9 | 242.8 | +20% |

eITL p50 at longctx c=1: 4.3 → 2.3 ms. TTFT p50 improves 10-28% across
longctx cells and 24-39% at short_chat — the prefill rider (prediction
5 said 0-10%) is much bigger than predicted, because prefill and mixed
steps run the same fused artifacts eagerly.

**Tripwires, all clean on this sha:**

- Recompiles after capture: **0 in every compiled arm's log** (dynamic:
  8 recompiles total, all during warm-up; batch-bucket: 20; longctx
  dynamic: 8).
- Artifact counts: dynamic 9 (short) / 9 (longctx); batch-bucket 21.
  Prediction 2 said ~2-4 / ~6-8: over, because the deliberate
  length-1/length-N seeding pairs roughly double the lineages (see the
  round story), and batch-bucket adds one pair per bucket.
- Graph capture under compile: present in every compiled arm — and
  *faster* (80 shapes: 3.4 s eager arms → 1.7 s compiled; 120 shapes:
  4.9 → 2.4 s): fewer kernels to record.
- Decode-step replay rate: **100.0% of pure-decode steps in all 24
  cells across both configs and all arms** (from engine_steps).
- nvidia-smi peak through the whole A/B: 13.1 GiB (graphs round:
  ~12.4 GiB); flat across warm-up→capture windows, no steady-state
  growth across repeats.

**Ready bills** (spawn→/health from the executor, cold Inductor cache):

| arm | short Ready | longctx Ready |
|---|---:|---:|
| baseline | 78.1 s | 113.3 s |
| +compile dynamic | 225.6 s (**+147.5**) | 257.0 s (+143.7) |
| +compile batch-bucket | 365.3 s (+287.2) | — |

Warm cache (manual respawn of compile-dynamic, short geometry, after
the runs): **143.7 s → warm bill +65.6 s.** The warm disk cache saves
Inductor codegen (~82 s of the cold warm-up) but Dynamo tracing +
AOTAutograd + guard install (~7 s × 9 artifacts) is not disk-cacheable
on torch 2.10. **Gate: warm bill under +30 s — FAILED.** The §9
deferred item (cache persistence / mega-cache) is the known remedy
avenue; halving the artifact count by dropping the length-pair seeding
would roughly halve the bill but reintroduces first-request compile
stalls.

## Step 4 — greedy equivalence across arms

Three manually spawned servers on the A/B's short geometry (default /
+compile dynamic / +compile batch-bucket), seven greedy requests with
prompt lengths spanning ~7 to ~1000 tokens, temperature 0, ignore_eos,
64 max_tokens (`extras/greedy_outputs.json`, per-arm server logs in
`extras/`).

**Streams are NOT identical vs eager: 5 of 7 prompts diverge from the
default arm** (both compiled arms; first divergence at char 79 / 225 /
246 / 16 / 125 of the respective outputs). The two compiled arms agree
with *each other* on 5 of 7 prompts (their own two disagreements are the
same mechanism between two different fused-kernel sets). Recorded
verbatim, prompt 0, divergence at char 79:

> eager  : `…Let me think. They might be greeting me, so a simple "Hello!" would work…`
> compiled: `…Let me think. A simple greeting is good. Maybe start with "Hello!"…`

Characterization (this is drift, not corruption): every divergent
continuation is coherent alternative text, and a direct logits
measurement on the real model (16-row decode over real prefilled
histories, same inputs both paths; `extras/logits_delta.log`) gives
max|Δlogit| 0.50 / mean 0.037 on a max-|logit| scale of 15.2, with
argmax matching 14/16 rows in one run and 15/16 in a repeat — **every
flipped row sits at a top-2 margin of exactly 0.0625, one bf16 ulp**
(which ties flip varies run to run). Fused kernels reorder float math; at
a near-tie the argmax flips and greedy decoding amplifies one flipped
token into a different (equally sensible) continuation. Design §4
predicted the low-bit drift but expected argmax to shrug it off; at bf16
near-ties it does not.

## §6 predictions, graded (kept, not edited)

1. Hoisted forward traces to one graph, fullgraph holds, the sdpa pin
   traces clean: **confirmed on CUDA** (suite, probes, and both A/B runs;
   no break anywhere).
2. Artifacts ~2-4 dynamic / ~6-8 batch-bucket, zero traffic recompiles:
   **half wrong, and wrong in an instructive way.** Final counts 9 / 21 —
   the extra lineages are the deliberate length-1/length-N seeding pairs
   the zero-recompile goal itself demanded. Zero-after-capture held only
   on the third sha: the as-landed implementation failed this prediction
   twice (dispatch-key leak, then the length-1 specialization), both
   caught by the §3 counter, both fixed this round.
3. Kernels/step ~1750 → 400-700: **beaten** (1859 → 218).
4. 16-row decode 5.93 → 4.6-5.4 ms: **beaten** (4.0 ms engine-side);
   1-row probe 3.63 → 2.4-3.1: **beaten via the A/B** (longctx c=1
   decode eITL p50 2.3 ms; the no-graphs probe's 1-row step went
   8.93 → 3.76).
5. short_chat c=16 → 2650-3050 (+8-24%): **exceeded** (3682.6, +49.6%).
   longctx c=1 → 190-215: **exceeded** (294.2, +64%). TTFT/prefill-bound
   numbers 0-10%: **exceeded** (10-39% improvements).
6. Strategies within 5%: **confirmed** (0.1-4.6% per cell — though
   batch-bucket is consistently the faster one, not a coin flip; its
   baked row constants are worth ~3% at c=16). The tie-break (fewer
   artifacts, smaller bill) still favors dynamic: half the artifacts,
   -140 s of cold Ready.
7. Cold bill 40-150 s dynamic, 1.5-2x for per-bucket: **confirmed at the
   top edge** (+147.5 s; bucket 1.95x). Warm under ~20 s: **wrong** —
   +65.6 s; Dynamo/AOT re-tracing dominates and the FX cache cannot
   help it. No measurable steady-state memory growth: **confirmed**.
8. Greedy token streams identical vs eager: **missed** — 5/7 prompts
   diverge, via bf16 argmax near-ties (margins ≤ 0.0625) under fused-
   kernel reordering; logits tolerance itself is comfortably met
   (max|Δ| 0.5 on |logit| ≤ 15.2) and the divergent continuations are
   coherent. Replay: **confirmed** at 100% of pure-decode steps in
   every cell; suite replay tests bit-exact. The cuDNN tripwire tests
   pass and the sdpa pin traces clean under compile.

## Gates (§7)

- short_chat c=16 aggregate ≥ +8%: **PASS**, +49.6%.
- Nothing regressing past −3% (longctx included): **PASS** — every cell
  in both configs improves (worst: longctx c=4 n=32, +19.7%).
- Warm-cache Ready bill < +30 s: **FAIL**, +65.6 s (independently
  reproduced at +64.6 s during the Step-4 spawns).
- Correctness set: logits tolerance PASS, replay PASS, cuDNN tripwire
  PASS, greedy token-identity vs eager **FAIL as worded** (bf16
  near-tie flips; see Step 4 — quantified as drift, not corruption).

Per the design note, clearing the gates was the condition for compile
joining the CUDA serve default. The performance gates clear decisively;
the bill gate and the strict greedy-identity gate do not.
Recommendation recorded for the author: keep `torch_compile` opt-in for
v1 (the note's stated fallback) — the +48-64% is real and reproducible,
but a default flip should wait on (a) the §9 cache-persistence item to
cut the +66 s warm boot bill, and (b) an explicit decision that
greedy-vs-eager token drift at bf16 ties is acceptable for the serve
default, since temperature-0 outputs change vs the eager engine.
Strategy default stays `dynamic` (half the artifacts and bill;
batch-bucket's ~3% edge doesn't cover +140 s of Ready).

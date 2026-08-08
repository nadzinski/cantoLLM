# CUDA graphs on the 5090: A/B results (2026-08-07)

The Phase-3 graphs implementation (landed 2026-08-05, `44bcebd`) measured
against the serve default (sdpa + buckets + warm-up), only `cuda_graphs`
flipping. Verdict up front: **the decode dispatch floor is gone and every
gate cleared with room to spare. 1859 → 50 cudaLaunchKernel calls/step at
16 rows, 16-row decode p50 10.43 → 5.93 ms — step-profiling's ~5.9 ms
GPU-busy figure, i.e. decode is now GPU-bound — short_chat c=16 jumps
1457 → 2466 tok/s (+69%, past the predicted +30–55%), and long_context,
predicted "within noise," instead gains +15–88% because its decode tails
were dispatch-bound too. Replay rate is 100% of pure decode steps in every
cell; greedy tokens are identical across arms. The bill: 3.4–5.0 s of
capture on top of warm-up, no measurable memory.** Environment: torch
2.10.0+cu128, driver 580.159.03, sm_120. Runs:
`bench/history/2026-08-07T180038_40fbcf9_ab-5090-cudagraphs` and
`…T180710_40fbcf9_ab-5090-cudagraphs-longctx` (each dir's
`agent-summary.md` is the raw run log; §6/§7 references are
`cuda-graphs-design.md`).

One bug first: the CUDA-marked tests, on first contact with real hardware,
failed exactly the way design-note §4 warned. `ModelRuntime.forward_batched`
compared `meta.positions.device` (`cuda:0`) against `self.device` (bare
`cuda`), concluded a move was needed, and `dataclasses.replace`d the capture
meta — dropping the seeded `kv_write_map`, so the `cached_property` rebuilt
the map *inside* the capture region and its pageable H2D `torch.tensor()`
invalidated every capture. Fixed in `40fbcf9` (gate the move on `.to()`
identity, not device equality); both runs sit on that sha. The tripwire
lesson keeps paying: without the bit-exact replay test the failure mode
would have been "graphs on, every shape silently eager."

## 1. Capture bill, observed at serve

Capture rides the warm-up sweep behind Ready, decode shapes only:

| config | decode shapes | capture wall | spawn→Ready |
|---|---:|---:|---:|
| 16×4096, 512/step | 80 | **3.4 s** | 74 → 78 s |
| 4×10240, 512/step (longctx) | 120 | **5.0 s** | 107 → 111 s |

~42 ms/shape — the §6 prediction (40–90 s) was off by an order of
magnitude in the good direction: it extrapolated the eager warm-up's
per-shape cost, but capture re-records already-warm kernels; the plan
compiles were already paid for by the sweep. Memory: nvidia-smi is flat
through the capture window (12,443 → 12,447 MiB; both arms peak
~12.4 GiB during warm-up), well under the §6 "<1 GB" gate. The only
observed pool number at all comes from a probe OOM at the *larger*
48-slot geometry (112 shapes): 1.03 GiB of graph private pools.

## 2. The A/B

Medians across 3 repeats from run.json; eITL/step in ms. Same seeds,
greedy, both arms per cell.

| cell | arm | tok/s | TTFT p50 | eITL p50/p99 | step p50/p99 |
|---|---|---:|---:|---:|---:|
| short_chat c=4 | default | 400.0 | 0.032 | 9.9 / 10 | 9.9 / 10 |
| | **+graphs** | **812.9** | 0.027 | 4.8 / 5 | 4.8 / 10 |
| short_chat c=16 | default | 1457.4 | 0.072 | 10.5 / 11 | 10.4 / 26 |
| | **+graphs** | **2466.4** | 0.066 | 6.0 / 6 | 5.9 / 25 |
| code c=8 | default | 754.4 | 0.072 | 10.1 / 17 | 10.0 / 17 |
| | **+graphs** | **1344.6** | 0.065 | 5.2 / 26 | 5.2 / 23 |
| multi_turn c=8 | default | 692.6 | 0.130 | 10.2 / 28 | 10.2 / 38 |
| | **+graphs** | **1151.5** | 0.124 | 5.5 / 28 | 5.5 / 38 |
| longctx c=1 | default | 93.1 | 0.080 | 9.6 / 10 | 9.6 / 20 |
| | **+graphs** | **175.2** | 0.072 | 4.2 / 6 | 4.2 / 20 |
| longctx c=2 | default | 147.7 | 0.193 | 9.9 / 24 | 9.8 / 24 |
| | **+graphs** | **207.9** | 0.180 | 6.3 / 23 | 6.3 / 23 |
| longctx c=4 n=16 | default | 177.0 | 0.440 | 13.5 / 37 | 13.7 / 37 |
| | **+graphs** | **207.3** | 0.416 | 10.4 / 35 | 10.8 / 35 |
| longctx c=4 n=32 | default | 175.0 | 0.398 | 13.8 / 37 | 13.9 / 56 |
| | **+graphs** | **201.9** | 0.388 | 10.7 / 42 | 10.8 / 53 |

**(a) The dispatch toll is gone; what's left is the GPU.** The
`profile_step` recheck (probe, exact protocol from the design note):
`fwd_call` — CPU time for the forward to *return* — collapses from
9.0–19.5 ms/step to **0.06 ms flat at every occupancy** (1→48 rows), and
cudaLaunchKernel goes 1859 → **50** calls/step at 16 rows (sampling +
the ~9-copy prologue). Engine-side, the 16-row pure-decode step lands at
p50 5.93 / p99 6.03 ms vs eager's 10.43 / 10.67 — i.e. at step-profiling's
~5.9 ms GPU-busy floor. The 1-row probe step is 3.63 ms (§6 said ≤2 ms:
missed — the prediction under-counted the 1-row GPU-busy floor, not the
dispatch win; `fwd_call` is 0.06 ms there too and the rest is `fwd_sync`).

**(b) Decode-heavy cells scale with their dispatch share.** +103% at
short_chat c=4 (a half-empty batch is *pure* toll), +69% at c=16, code
+78%, multi_turn +66%. eITL p50 roughly halves everywhere (10 → 5-6 ms);
the p99s don't move because they are prefill-step-bound, which graphs
deliberately leave eager.

**(c) The long_context prediction was wrong, in the good direction.**
§6 called it "within noise" (compute-bound, SDPA's territory). Actual:
+88% at c=1, +41% at c=2, +15–17% at c=4. The miss: each request is a
~8k prefill *plus a 128-token decode tail*, and at c=1–2 those tails run
at 1–2 rows — maximum dispatch share, nothing to overlap with. Decode
eITL p50 drops 9.6 → 4.2 ms at c=1. By c=4 prefill genuinely dominates
and the gain shrinks toward the gate's noise band. Same lesson as (b)
seen from the other side: what graphs buy is proportional to time spent
in small decode steps, and even "long-context" workloads spend plenty.

**(d) The tripwire numbers.** Replay rate over *pure decode* steps
(width 1, no prefill tokens): **100.0%** in all eight cells across both
configs — zero decode-shaped steps fell through to eager. Whole-run
rates are 93.6–98.5% (short config) because prefill/mixed steps are
eager by design; multi_turn is lowest only because it is
prefill-heaviest. Shape census: this traffic touched 10 of 80 captured
shapes (short config) and 31 of 120 (longctx) — the same
which-shapes-is-workload-dependent story as the warm-up sweep's 179/477.

## 3. Correctness

Full suite on this box: 440 passed, 3 skipped, including the four
CUDA-only capture tests — replay logits **bit-exact** vs eager
(`torch.equal`, no tolerance), the filler-padded step matches eager with
the pool byte-identical, and the wiring test replays with hits > 0. On
top, a live greedy equivalence check on 0.6B: seven requests, prompt
lengths 7–1023 (spanning batch buckets and kv spans, 63 replayed / 6
eager steps), token streams **identical** graphs vs eager. Both §7
correctness gates pass.

## 4. §6 predictions, graded (kept, not edited)

1. ~1900 API calls/step → <100: **confirmed** (1859 → 50).
2. 1-row 9.3 → ≤2 ms: **missed** (3.63 ms; residual is GPU execution,
   dispatch itself is 0.06 ms).
3. 16-row p50 ~10.5 → 6-7 ms: **confirmed** (10.43 → 5.93).
4. short_chat c=16 1469 → 1900–2300: **exceeded** (2466, +69%).
5. long_context within noise: **wrong, good direction** (+15–88%).
6. Decode hit rate ≥95%: **confirmed at 100%**.
7. Capture bill 40–90 s: **wrong, good direction** (3.4–5.0 s).
8. Shared pool <1 GB: **confirmed** (unmeasurable at serve geometry).

## 5. Costs, anomalies, options (decisions are the author's)

- **The startup bill is now warm-up, not capture.** Capture adds ~4 s to
  a 74–107 s Ready; any startup work goes to the sweep itself (the
  decode-shapes-first / span-ceiling ideas from shape-buckets-results §5
  apply unchanged, and capture would ride along for free).
- **The probe leaks across schedulers under graphs.** `profile_step`'s
  Phase B OOM'd until a `gc.collect()` preceded `empty_cache()`: captured
  graphs' private pools + reference cycles keep the previous pool alive.
  Scratch-copy fix only; worth remembering if the engine ever rebuilds a
  scheduler in-process (serving never does).
- **Grade-2's residual is the next target, if wanted.** At 1 row the step
  is 3.63 ms of pure GPU with dispatch at 0.06 ms — further wins are
  kernel-count/fusion territory (`torch.compile` remains the open Phase 3
  experiment; the graphs compose with it, §3.9).
- **Environment caveat for reproduction:** mid-session Ubuntu upgraded
  the NVIDIA userspace to 580.173.02 under the loaded 580.159.03 kernel
  module (CUDA error 804 for new processes). Everything here ran with
  LD_LIBRARY_PATH pinned to the matched 580.159.03 userspace extracted
  from the apt cache; the box needs a reboot for a matched installed
  stack.

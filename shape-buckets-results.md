# Shape vocabulary on the 5090: A/B results (2026-07-19)

The `shape-vocabulary` branch measured against 2026-07-19's ab-5090-sdpa run,
same geometry (0.6B bf16, 4×10240 slots, 512/step), same points. Verdict up
front: **the bounded vocabulary fixes cuDNN-SDPA outright. The plan-churn
stall tail is gone — zero steps over 100 ms where the unbucketed run had
23.9% of them — short_chat TTFT is back to 0.03 s, and warm cuDNN now wins
long_context by 1.6–2.2× (146–179 tok/s vs padded's 76–92 at c=2/4).
Bucketing overhead on padded is noise-level. The bill: 102 s of warm-up at
server start on sdpa.** Environment: torch 2.10.0+cu128, driver 580, sm_120.
Runs: `bench/history/2026-07-19T182642_a85e589_ab-5090-shape-buckets` (new)
vs `2026-07-19T152024_ef5d280_ab-5090-sdpa` (old).

## 1. Warm-up cost, observed at serve

At this geometry the vocabulary is 477 shapes (batch {1,2,4} × width
{1,128,256,512} × 256-token KV spans up to 10240, minus the unreachable
kv < width pairs). One all-filler forward per shape behind Ready:

| attention | shapes | wall clock |
|---|---:|---:|
| sdpa | 477 | **102.0 s** (~214 ms/shape) |
| padded | 477 | **19.3 s** |

The ~83 s gap is the cuDNN plan compiles themselves — per-shape cost matches
sdpa-results.md's isolated ~200 ms/plan. Padded's 19.3 s is just the raw
forwards (some are big: 4×512@10240), a lower bound any backend pays.

## 2. The four-way

Medians across 3 repeats from run.json; step p50/p99 pooled over each cell's
engine_steps. eITL/step columns in ms.

| cell | arm | tok/s | TTFT p50 | eITL p50/p99 | step p50/p99 |
|---|---|---:|---:|---:|---:|
| longctx c=1 n=16 | padded | 86.2 | 0.091 | 9.3 / 13 | 9.3 / 41 |
| | padded+buckets | 90.3 | 0.093 | 9.3 / 10 | 9.3 / 41 |
| | sdpa | 28.9 | 0.257 | 9.7 / 208 | 9.7 / 208 |
| | **sdpa+buckets** | **92.9** | **0.080** | 9.7 / 10 | 9.6 / 20 |
| longctx c=2 n=16 | padded | 92.3 | 0.398 | 15.7 / 93 | 16.0 / 98 |
| | padded+buckets | 90.5 | 0.503 | 15.4 / 65 | 15.6 / 68 |
| | sdpa | 30.5 | 0.933 | 10.0 / 262 | 10.1 / 269 |
| | **sdpa+buckets** | **146** | **0.195** | 10.0 / 24 | 10.0 / 24 |
| longctx c=4 n=16 | padded | 80.0 | 1.00 | 26.7 / 195 | 27.1 / 195 |
| | padded+buckets | 81.5 | 1.16 | 25.4 / 115 | 25.7 / 115 |
| | sdpa | 23.1 | 2.40 | 208 / 296 | 210 / 296 |
| | **sdpa+buckets** | **179** | **0.414** | 13.6 / 37 | 13.8 / 37 |
| longctx c=4 n=32 | padded | 78.0 | 0.911 | 27.3 / 189 | 27.1 / 193 |
| | padded+buckets | 76.4 | 1.08 | 25.9 / 116 | 25.9 / 179 |
| | sdpa | 57.2 | 1.17 | 13.6 / 295 | 13.7 / 295 |
| | **sdpa+buckets** | **175** | **0.404** | 13.8 / 37 | 13.8 / 56 |
| short_chat c=4 n=32 | padded | 400 | 0.030 | 9.9 / 10.6 | 9.9 / 10.6 |
| | padded+buckets | 402 | 0.033 | 9.8 / 10.5 | 9.8 / 10.8 |
| | sdpa | 303 | 0.434 | 10.0 / 10.9 | 10.0 / 264 |
| | **sdpa+buckets** | 396 | **0.032** | 10.0 / 10.4 | 10.0 / 10.9 |

**(a) Plan churn: eliminated.** sdpa's eITL p99 drops 208/262/296/295 →
10/24/37/37 ms across the longctx cells; short_chat's step p99 264 → 10.9 ms
and its TTFT 0.434 → 0.032 s (the ~0.03 s target). The blunter census: the
unbucketed sdpa arm had 4 370 steps over 100 ms (23.9% of 18 316, max
300 ms); the bucketed arm has **zero** over 100 ms in 19 024 steps, max
61 ms. The cache-warming signature is gone too — old sdpa repeats trended
hard (longctx c=1: 10.4 → 28.9 → 90.9 tok/s as plans accumulated), new
repeats are flat (92.9 / 93.1 / 92.1).

**(b) Warm cuDNN vs padded: it wins long_context, decisively.** Against the
unbucketed padded baseline: +8% at c=1 (92.9 vs 86.2), +58% at c=2 (146 vs
92.3), +124% at c=4 (179 vs 80.0 n=16, 175 vs 78.0 n=32) — the number the
SDPA design decision was waiting for is a 2.2× at the batch sizes long
context actually runs at. Same story vs padded+buckets, so it's the kernel,
not the bucketing. TTFT drops 2–5× at c≥2 alongside. The magnitude is
consistent with the kernel-level probe (cuDNN 0.094 ms vs math-class
1.5 ms on the longctx shape): at c=4 the median step is 13.8 ms vs padded's
25.9. short_chat is the one cell sdpa doesn't win: 396 vs 400–402 tok/s,
a ~1% deficit at decode-dominated small-KV shapes.

**(c) Bucketing overhead on padded: noise.** tok/s deltas per cell: +4.8%,
−1.9%, +1.9%, −2.1%, +0.5% — all inside these cells' repeat spread (the
run's CV warnings, 5–20%, match the old run's padded variance). The step
p99 actually *improved* (93 → 65, 195 → 115 ms): chunk quantization narrows
the worst prefill chunks (a raw 511-token chunk becomes 256), trimming the
tallest steps. Cost of pad columns, over-read spans, and filler rows:
not measurable at this geometry.

## 3. The vocabulary held in production traffic

`StepStats` grew `fwd_rows` / `fwd_width` / `fwd_kv_len` this run (small
instrumentation addition on the branch): the forward's actual post-bucketing
problem shape, which the existing `rows` (real sequences) and `kv_tokens`
(position sum) fields cannot reconstruct. Census over the new run's
engine_steps: **179 distinct shapes** across ~19 000 steps per arm, the
*identical* set in both arms (deterministic scheduling, same traffic), every
one a member of the 477-shape vocabulary — batch ∈ {1,2,4}, width ∈
{1,128,256,512}, KV span a 256-multiple ≤ 8960. That matches the CPU
shape-property test, now confirmed on real traffic. For contrast, the old
unbucketed run produced ≥1 579 distinct KV spans among its single-row steps
alone — each one a fresh ~200 ms compile on sdpa.

## 4. Correctness

Full suite on this box: 415 passed (the 3 CUDA sdpa-equivalence tests ran),
3 MPS skips. On top of the CPU equivalence suite, a GPU spot-check drove
live servers on CUDA: greedy outputs (one 1.3k-token multi-chunk prefill +
three concurrent shorts, forcing filler rows, width rounding, and span
over-read) are **byte-identical** bucketed vs unbucketed, for both sdpa and
padded, with self-determinism verified on repeat runs. No equivalence
failure anywhere.

## 5. Costs and options (decisions are the author's)

- **The 102 s startup.** It buys a strictly better steady state, but it's
  real wall clock behind Ready, and it scales with the vocabulary
  (kv spans × widths × batches). If it matters: warm decode shapes
  (width 1) first and let prefill shapes trail Ready; or derive the KV-span
  ceiling from expected prompt lengths instead of slot capacity. Only 179
  of 477 shapes were touched by this workload — but which 179 is
  workload-dependent, so pruning trades a startup win for reintroducing a
  first-hit stall.
- **Step-count overhead is small.** Bucketed arms ran ~5% more steps
  (quantization defers tokens to the next chunk), visibly not costing
  throughput.
- **short_chat stays padded-fast either way** (396 vs 402); nothing here
  argues for flipping any default until the author decides — per the branch
  rules, no defaults were changed and PLAN.md is untouched. This note and
  the bench history are the input to that decision.

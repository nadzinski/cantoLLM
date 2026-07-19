# SDPA on the 5090: validation results (2026-07-19)

The Phase-3 SDPA validation day, run against the attend landed in 910b7bd.
Verdict up front: **the attend is correct, the decided routing assumption
was wrong twice over, and the fused path loses the A/B badly for a reason
none of the microbenchmarks could see — cuDNN compiles a ~200 ms execution
plan per distinct problem shape, and continuous batching churns shapes by
construction.** `--attention padded` stays the default; nothing here says
otherwise. Environment: torch 2.10.0+cu128, driver 580, sm_120, 0.6B bf16.

## 1. The dispatcher story (two silent fallbacks deep)

The design assumed the explicit per-row bool mask routes to the
memory-efficient fused backend. On this stack it does not — and the failure
mode was *silent* both times:

- **Memory-efficient rejects dense GQA** (`enable_gqa`): "both fused
  kernels require q/k/v to have the same num_heads" — so our call can't
  route there at all. Flash honors GQA but rejects the mask tensor, as the
  design documented. The one fused backend that takes mask + GQA is
  **cuDNN**.
- **The default dispatcher ranks cuDNN below math on this build**, so the
  as-landed attend ran the unfused math backend on CUDA — correct outputs,
  einsum-class performance, zero warnings.
- **`sdpa_kernel([CUDNN, MATH])` alone doesn't fix it**: the context
  restricts the backend *set* but keeps the build's priority order, and
  math still outranks cuDNN. The pin needs `set_priority=True` (now in the
  attend; math stays listed as the CPU/MPS fallback).

`bench/probe_sdpa.py` (pinned, fixed shapes, 0.6B geometry): cuDNN accepts
the call everywhere and is fast — decode 16×1@512 0.019 ms, longctx 4×1@8192
0.094 ms vs math 0.366 / 1.531 ms. Flash also rejects `is_causal` outright
for rectangular q≠kv ("use varlen"), confirming Phase 4's restructure can't
shortcut through the causal flag. cuDNN-causal vs cuDNN-explicit-mask is
0.007 vs 0.094 ms on longctx — the mask-tensor cost preview (alignment
semantics differ for non-square, so cost preview only).

Tests now pin the corrected reality (all green, 401 + 3 skipped):
raw-call tripwires at production bf16 geometry (cuDNN accepts / efficient
rejects GQA / flash rejects mask — the latter still the obsolescence
tripwire), a cuDNN-vs-math numerics check, and
`test_attend_runs_fused_on_cuda`, which profiles the real attend and
asserts a cuDNN kernel executed — the only test shape that catches a
silent priority fallback, which output-level tests never will.

## 2. The A/B (`bench/history/2026-07-19*_ab-5090-sdpa`)

Geometry mirrors the longctx baseline (4×10240 slots, 512/step). Padded arm
first — it re-validates the 2026-07-18 baseline (86/92/80 tok/s longctx
c=1/2/4 vs 89/94/85; long context didn't move with the dispatch fixes,
confirming it's compute-bound). Then the same cells on `--attention sdpa`:

| cell | padded | sdpa | sdpa eITL p99 |
|---|---:|---:|---:|
| long_context c=1 (n=16) | 86 tok/s | **29** | 208 ms |
| long_context c=2 (n=16) | 92 | **30** | 262 ms |
| long_context c=4 (n=16) | 80 | **23** (step p50 209 ms!) | 296 ms |
| long_context c=4 (n=32) | 78 | **57** | 295 ms |
| short_chat c=4 (n=32) | 400 (TTFT 0.03 s) | **303** (TTFT 0.43 s) | 11 ms |

The *shape* of the loss is the diagnosis: median steps are fine (9.7 ms vs
padded's 9.3 at c=1), the damage is a ~200–300 ms p99 stall tail;
short_chat decode cadence is untouched while its TTFT exploded 13×; and
every sdpa cell improves as more requests run (n=16 → n=32 nearly
doubled). All of that says: a per-shape one-time cost, amortizing as a
cache warms.

## 3. The mechanism, isolated

First-call vs repeat-call cost of the exact pinned call as shapes vary by
±1 (batch 4, GQA 16/8, head_dim 128):

| shape change | first call | repeat |
|---|---:|---:|
| kv_len 8192 / 8193 / 8194 / 4096 / 300 / 301 | 194–212 ms each | 0.22–0.26 ms |
| num_new 17 / 18 / 19 (kv 2048) | ~197 ms each | ~0.28 ms |

cuDNN builds and caches an execution plan keyed on the exact problem
shape; every new shape is a fresh ~200 ms compile. The engine changes
`max_history_len` every decode step and `num_new_max` per prefill chunk —
shape churn is intrinsic to continuous batching. Fixed-shape
microbenchmarks (the probe, the tripwire tests) structurally cannot see
this; only the A/B could.

## 4. Options from here (decisions are the author's)

- **Bucket the attention shapes.** Round the KV gather (`max_history_len`)
  up to, say, 256-token boundaries — the per-row causal mask already fences
  everything past the real history, including stale slot data, so
  correctness machinery for over-reading exists. That caps distinct decode
  shapes at ~40 per geometry (~8 s one-time compile, amortized or
  pre-warmed at startup) at the cost of some wasted attention compute.
  Prefill's `num_new_max` would want the same treatment (more invasive:
  input width, RoPE positions).
- **Pre-compile at server start** — a warmup sweep over the bucketed shape
  vocabulary behind `/ready` (pairs with Phase 3.5's warm-up story).
- **Accept and defer to Phase 4**: the varlen/FlexAttention restructure
  (lengths metadata, no mask tensor, flash-family kernels without
  per-shape plan compilation) dissolves the problem class — this A/B is
  arguably the measured argument for it.
- Not worth it on the numbers: manually expanding KV heads to route to the
  efficient backend (2× KV read bandwidth to dodge a compile cache).

## 5. Status

`--attention sdpa` works, is correct, runs fused — and loses end-to-end
under shape churn, so **padded remains the CUDA default**. The equivalence
suite, the corrected pin, and the tripwires are in; the A/B and this note
are the record. Next Phase-3 moves (compile, CUDA graphs) should keep this
in view: CUDA graphs also want a bounded shape vocabulary, so shape
bucketing may end up load-bearing for both.

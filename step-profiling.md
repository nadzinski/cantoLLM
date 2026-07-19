# Step-time profiling notes (2026-07-18, 5090)

Where a decode step's wall clock actually goes on the batched engine —
measured right after the Phase-2 baseline, as the entry brief for Phase 3.
Regenerate with `bench/profile_step.py` (drives the real scheduler
in-process on 0.6B; component timers + torch.profiler). Context: the
baseline showed step time ~11.4 ms nearly independent of row count up to
16 rows, and the wide-slots run fit step time ≈ 9.2 ms fixed + 0.29 ms/row.
These notes attribute both constants.

## Headline: the CPU is the bottleneck, not the GPU

At the baseline server shape (16 rows, 0.6B, decode), a ~16 ms step keeps
the GPU busy ~5.9 ms — the GPU idles ~60% of every step waiting for Python
to feed it. At 1 row: 9.3 ms step, of which 8.97 ms is `forward_batched`
*dispatch* time (the call returning, before any GPU wait). The ~9 ms floor
is CPU.

Decomposition (ms/step, 200 decode steps; `fwd_call` = CPU dispatch,
`fwd_sync` = GPU catch-up after; "finalize" = the post-forward loop in
`step()`):

| rows | total | fwd_call | fwd_sync | finalize | plan+ids+meta+sample |
|-----:|------:|---------:|---------:|---------:|---------------------:|
| 1    | 9.3   | 8.97     | 0.20     | 0.10     | 0.06 |
| 16   | 15.8  | 12.6     | 1.25     | 1.68     | 0.22 |
| 48   | 48.1  | 33.4     | 9.3      | 4.7      | 0.60 |

(48-row caveat: the probe's sequences had accumulated ~1k tokens of KV
history by that sweep — longer than the bench cells — inflating
`fwd_sync` there. Structure of the conclusion unaffected.)

The step issues **~1750 kernel launches + ~960 async memcpys** (~2700 CUDA
API calls at ~2 µs each, plus the aten/Python dispatch that produces
them). Scheduler bookkeeping (`_plan_step`, `_build_input_ids`,
`build_batch_meta`) is noise — tens of microseconds.

## The three concrete sinks, ranked (16 rows)

1. **Ragged KV writes: ~3.9 ms/step (~25%).** *(fixed 2026-07-19,
   `9d0782e` — see addendum)*
   `PaddedAttentionMethod.forward_batched` writes each row's K/V with a
   Python slice-assign per row per layer: 16 × 28 × (K+V) = 896 tiny
   copies/step. Top CPU consumer in the profile (`aten::copy_`, ~4.8
   ms/step with children) *and* 28% of all GPU time (924 sub-µs DtoD
   memcpys). Micro-benchmark: the loop pattern costs 3.85 ms/step; one
   vectorized indexed write per layer (`layer_k[slots_tensor, pos] = ...`
   for the uniform-width decode case) costs 0.81 ms — **~3 ms/step
   available in one loop**.
2. **Per-row sampling syncs: ~1.2 ms/step, linear in rows.** *(fixed
   2026-07-19, `02e30e0` — see addendum)* The finalize
   loop calls `.item()` on the sampled token and `.log().item()` on its
   logprob per row → 32 `cudaStreamSynchronize`/step (~31 µs each;
   profiler count matches). A batched argmax + one host transfer for the
   whole batch removes all but one.
3. **The launch flood itself.** The real math is small: 197 GEMMs/step
   (~10 µs each, 7 linears × 28 layers ≈ 2 ms GPU total) plus hundreds of
   elementwise/norm/index kernels — einsum attention unfused, each RMSNorm
   a mul/mean/rsqrt chain. This is the generic launch-bound overhead that
   only fusion or graphs remove.

## Mapping to Phase 3 (magnitudes now attached)

- **CUDA graphs** attack the ~2700-API-call flood — the ~9 ms floor.
  Biggest single lever on small models, as the baseline data suggested.
- **Vectorizing the KV write** is ~3 ms/step on its own, independent of
  graphs (and shrinks what a graph has to capture). Touches the
  hand-written `forward_batched`.
- **Batching the sample/finalize path** removes the 32 syncs/step.
  Touches the hand-written scheduler finalize loop.
- **SDPA + torch.compile** fuse the softmax/elementwise chains — that's
  the 0.29 ms/row slope, and (per the baseline) the long-context story,
  which is compute- not launch-bound.

Numbers: 0.6B bf16, padded einsum path, torch 2.10.0+cu128, driver 580,
git e958a4b.

## Addendum (2026-07-19): sinks 1 and 2 fixed — measured

Two fixes landed the next morning: `9d0782e` (the ragged KV write as a
per-step `KVWriteMap` — one gather + scatter per tensor, index tensors
built once per step on the pool device, shared by all layers) and
`02e30e0` (finalize collects sampled tokens/logprobs as 0-dim GPU tensors
and transfers once per step — 32 pipeline drains down to 2).

Probe re-runs (same script, total ms/step):

| rows | pre-fixes | +KV vectorize | +batched finalize |
|-----:|----------:|--------------:|------------------:|
| 1    | 9.33      | 9.11          | 9.13 |
| 8    | 12.17     | 9.90          | 9.92 |
| 16   | 15.77     | 13.53         | 13.38 |
| 48   | 48.09     | 38.62         | 38.34 |

CPU dispatch (`fwd_call`) went from climbing 9.0→12.6 ms across 1→16 rows
to flat ~9.1 ms; `aten::copy_` calls dropped by exactly the 896/step loop
and the ~960/step `cudaMemcpyAsync` vanished (CUDA API calls ~2700 →
~1900/step). Fix 2 looks small *in the probe* by construction: the
probe's post-forward synchronize drains the pipeline before finalize
runs, hiding the serialized per-row round-trips it removed — the probe
understates it; trust the engine numbers below.

Real-engine check (`bench/history/*recheck-step-opts`, same cells as the
baseline's 16-slot variant, zero validity warnings):

| cell (16 slots, short_chat) | baseline | `02e30e0` | Δ |
|---|---:|---:|---:|
| c=8 aggregate tok/s   | 692  | 776  | +12% |
| c=16 aggregate tok/s  | 1145 | **1469** | **+28%** |
| c=8 step p50          | 11.4 ms | 10.13 ms | −11% |
| c=16 step p50         | 13.6 ms | 10.48 ms | −23% |
| c=16 engine ITL p50   | 13.6 ms | 10.53 ms | −23% |

Step time is now nearly flat 8→16 rows (10.1 → 10.5 ms): the per-row
dispatch cost in that range is essentially gone, and the real-engine win
exceeds the probe's because the un-instrumented loop lets dispatch and
GPU execution overlap once the mid-loop syncs are gone. Remaining from
the ranked list: the ~10 ms per-layer launch flood (CUDA graphs) and the
compute slope / long-context story (SDPA, compile). The Phase 3
before/after protocol is unchanged — these fixes simply move the "before"
that SDPA and graphs will be measured against.

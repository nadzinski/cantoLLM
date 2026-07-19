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

1. **Ragged KV writes: ~3.9 ms/step (~25%).**
   `PaddedAttentionMethod.forward_batched` writes each row's K/V with a
   Python slice-assign per row per layer: 16 × 28 × (K+V) = 896 tiny
   copies/step. Top CPU consumer in the profile (`aten::copy_`, ~4.8
   ms/step with children) *and* 28% of all GPU time (924 sub-µs DtoD
   memcpys). Micro-benchmark: the loop pattern costs 3.85 ms/step; one
   vectorized indexed write per layer (`layer_k[slots_tensor, pos] = ...`
   for the uniform-width decode case) costs 0.81 ms — **~3 ms/step
   available in one loop**.
2. **Per-row sampling syncs: ~1.2 ms/step, linear in rows.** The finalize
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

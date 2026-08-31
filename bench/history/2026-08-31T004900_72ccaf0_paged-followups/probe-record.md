# P4 chunk 7 round-1 follow-up probes (72ccaf0, 5090)

Companion to the round-1 dirs (`...T15*_2ac397f_*`, `...T17*_cb1a04a_*`).
Run 2026-08-31 on the 5090 box; no code edits, production scheduler path.

## Probe A — trimmed sweep bills (round-1 serve flags, TORCH_LOGS=recompiles)

| boot                    | Ready (wall) | sweep      | note |
|-------------------------|--------------|------------|------|
| cold (inductor stashed) | 368 s        | 347.4 s    | log: "20 paged (batch, width) families, one map length each" |
| warm restart            | 79 s         | 65.0 s     | |

vs cb1a04a untrimmed 602 s / 144 s, and round-1 (pre-static-fix) 299 s / 72 s.
Warm bill fully recovered; cold sits 69 s over round 1 (the ~5 per-family
static decode artifacts are real cold work the symbolic artifact never paid).

Reachability on hardware: 6-way concurrent greedy burst, then one lone
request decoding alone (the (1,1) family). Recompile lines after Ready: 0.

## Probe B — residual decode-gap split (3000-token prompts, steady decode,
mean of 10 steps; wall from clean timed loop, GPU-busy from profiler)

| arm  | B | wall     | GPU-busy | host gap | attention-kernel share |
|------|---|----------|----------|----------|------------------------|
| flex | 1 | 5.52 ms  | 3.91 ms  | 1.61 ms  | flex template 2.116 ms/step (28 calls) |
| flex | 4 | 9.78 ms  | 6.63 ms  | 3.15 ms  | flex template 4.111 ms/step |
| sdpa | 1 | 4.05 ms  | 2.22 ms  | 1.84 ms  | cuDNN 0.186 + mask-prep triton 0.280 ms/step |
| sdpa | 4 | 5.14 ms  | 3.98 ms  | 1.15 ms  | cuDNN 0.428 + mask-prep triton 1.097 ms/step |

Deltas (flex minus sdpa):
- B=1: wall +1.47 ms = GPU +1.69, gap -0.23 → entirely kernel time.
- B=4: wall +4.64 ms = GPU +2.65 (57%), gap +2.00 (43%).

Reading: the flex attention template runs ~4x the cuDNN sdpa attention time
at B=1 (2.12 vs ~0.47 ms/step incl. mask prep) and ~2.7x at B=4 (4.11 vs
~1.53). Chunk 8 graphs can absorb the host-gap share (up to ~2 ms/step at
B=4, where flex's gap also exceeds sdpa's) but not the kernel share; closing
the rest is template/split tuning at these decode geometries.

Top kernels, flex arm (self ms/step): B=1: flex template 2.116 (28), gemvx
1.075 (141), gemvx 0.170 (56). B=4: flex template 4.111 (28), cutlass bf16
gemm 1.746 (197), softmax 0.117 (4).

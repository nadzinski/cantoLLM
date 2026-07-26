# CUDA-graph capture record: 2026-07-26

Raw artifacts behind the viz "CUDA graphs" tab (the tab embeds a curated
inline copy; this directory is the record it was transcribed from).
Produced by `viz/capture_cudagraphs.py` on the `infra/` GPU node:
**NVIDIA L4 (g6.xlarge, sm_89), driver 595.71.05, torch 2.10.0+cu128,
CUDA 12.8**. Same mechanics as the 5090, smaller absolute numbers.

Files: one Graphviz DOT per captured topology (`cudaGraphDebugDotPrint`
output: kernel nodes with mangled names + launch configs, dependency
edges), `timings.json` (eager vs replay, GPU wall + CPU submit, 200 iters
after warm-up), `failures.json` (recorded rule-breakers, real error text),
`env.json`.

## Headline timings (µs per iter)

| example            | eager | replay | eager submit | replay submit |
|--------------------|------:|-------:|-------------:|--------------:|
| chain, 4k elems    |  31.3 |    5.9 |         31.4 |           3.3 |
| chain, 1M elems    |  31.5 |   27.2 |         31.6 |           3.4 |
| chain, 16M elems   |  3183 |   3187 |         33.5 |           3.4 |
| forkjoin, 1 stream |  44.9 |   11.7 |         45.0 |           3.2 |
| forkjoin, 2 stream |  90.5 |    9.7 |         90.6 |           4.5 |
| tiny-model decode  |  2066 |    134 |         2066 |           3.5 |

The constant across every row: **CPU submit collapses to ~3.5 µs**, one
`cudaGraphLaunch`, regardless of how many kernels the graph holds.

## Findings

1. **The size sweep shows the win is launch overhead, nothing else.**
   4k elements: 5.3x. 1M: 1.16x. 16M: parity (within noise, replay even
   read 0.1% slower once). Graphs buy back the space between kernels;
   they do not make kernels faster.
2. **The order trap.** `forkjoin_1stream.dot` was *written* as a fork-join
   but captured on one stream: the DOT is a straight chain of 5, because
   edges come from stream order. The two-stream variant (fork/join via
   events) records the two gemms as genuinely parallel branches
   (`node_0 -> node_2 <- node_1`), and its *replay* beats the serial
   graph (9.7 vs 11.7 µs) while its *eager* form is twice as expensive
   (90.5 vs 44.9 µs: stream/event choreography is CPU work). The graph
   keeps the parallelism and discards the choreography cost.
3. **The tiny-model decode step is the engine's floor in miniature.**
   110 kernel nodes, 109 edges, max fan-in 1 (single stream, pure chain)
   for the 2-layer 64-dim fixture: 15 gemv + 4 gemm (single-token decode
   turns matmuls into matvecs), 9 pow/mean/rsqrt trios (exactly the 9
   RMSNorms), 6 cats (the KV-cache appends), 2 softmaxes, and a tril+fill
   pair (the causal mask built inside the forward). Eager 2066 µs/iter is
   ~100% CPU submit; replay is 134 µs (15.4x). Scale 110 nodes at 2
   layers to 28 layers and you get the ~1750-launch flood of
   step-profiling.md.
4. **`debug_dump` quirk.** On this torch build the `enable_debug_mode()`
   path returns without writing anything; the working route is
   `CUDAGraph(keep_graph=True)` + `debug_dump()` + explicit
   `instantiate()` before replay. The script does that.
5. **`cache_no_advance`** (failures.json): capture bakes one step's
   buffers, so the concat-grown sequential `KVCache` sits at position 17
   forever across replays, logits bit-identical. A graphable decode needs
   the preallocated pool with device-tensor indices, which the batched
   engine already has.

## Regenerating

Needs CUDA hardware; the Mac cannot do this. From the repo root:

```
cd infra && ./up.sh && ./sync.sh
./ssh.sh 'cd cantoLLM && ~/.local/bin/uv sync && .venv/bin/python viz/capture_cudagraphs.py'
scp -r ubuntu@<ip>:cantoLLM/viz/captures/cudagraphs-<date> viz/captures/
./down.sh
```

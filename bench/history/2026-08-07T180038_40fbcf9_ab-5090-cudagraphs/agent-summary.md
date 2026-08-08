# 5090 validation summary: CUDA graphs (Phase 3) — 2026-08-07

Agent run log for the CUDA-TEST-AGENT-INST.md protocol, graded against
cuda-graphs-design.md §6/§7. Covers this run dir and its longctx sibling
`2026-08-07T180710_40fbcf9_ab-5090-cudagraphs-longctx`. The
`cuda-graphs-results.md` write-up happens back home from these numbers.

## Step 1 — suite

440 passed, 3 skipped — after one real fix. First contact of
`TestCaptureReplayCUDA` with real CUDA hardware caught a production bug:
`ModelRuntime.forward_batched` compared `meta.positions.device` (`cuda:0`)
against `self.device` (bare `cuda`), concluded a move was needed, and
`dataclasses.replace`d the capture meta — dropping the seeded
`kv_write_map`, so the `cached_property` rebuilt the map *inside* the
capture region and its pageable H2D `torch.tensor()` invalidated every
capture (`cudaErrorStreamCaptureInvalidated`) — exactly the §4 hazard the
design note named. Fixed in `40fbcf9` (gate the move on `.to()` identity).
After the fix: capture/replay bit-exact vs eager (`torch.equal`),
filler-padded step matches eager with the pool untouched, wiring test
replays with hits > 0.

## Step 2 — profile_step recheck (probe, 0.6B)

Eager arm (unmodified script) reproduces step-profiling.md: 1-row
9.40 ms/step (fwd_call 9.00), 16-row 13.36 ms; 1859 cudaLaunchKernel
calls/step at 16 rows.

Graphs arm (scratch copy: `**default_shape_buckets(...)`,
`warmup_shapes=True, cuda_graphs=True`, attention left "padded"):

| rows | total ms (eager → graphs) | fwd_call ms | fwd_sync ms  |
|-----:|--------------------------:|------------:|-------------:|
| 1    | 9.40 → 3.63               | 9.00 → 0.06 | 0.20 → 3.43  |
| 8    | 10.19 → 6.30              | 9.29 → 0.06 | 0.23 → 5.74  |
| 16   | 13.36 → 11.09             | 9.10 → 0.06 | 3.01 → 10.12 |
| 32   | 24.76 → 22.70             | 11.7 → 0.06 | 10.6 → 20.8  |
| 48   | 40.35 → 38.96             | 19.5 → 0.07 | 17.3 → 36.3  |

cudaLaunchKernel calls/step at 16 rows: **1859 → 50**.

Probe caveats applied per the protocol: high-row fwd_sync carries the
probe's accumulated ~1k-token KV histories plus kv-bucket rounding
(bucketed spans read past exact geometry), so engine-side step times
below are the trustworthy ones. One probe anomaly for the record: Phase B
of the graphs arm initially OOM'd — the captured graphs' private pools +
reference cycles keep Phase A's 48-slot pool alive past `empty_cache()`;
a `gc.collect()` between phases fixed the probe (scratch copy only).

## Step 3 — A/B, short-context (`ab_5090_cudagraphs.toml`, this run dir)

Both arms sdpa + buckets + warm-up; only `cuda_graphs` flips.

| cell            | eager tok/s | graphs tok/s | delta |
|-----------------|------------:|-------------:|------:|
| short_chat c=4  | 400.0       | 812.9        | +103% |
| short_chat c=16 | 1457.4      | 2466.4       | +69%  |
| code c=8        | 754.4       | 1344.6       | +78%  |
| multi_turn c=8  | 692.6       | 1151.5       | +66%  |

The eager arm reproduces the recorded baseline (1457 vs ~1469 at c=16).
16-row pure-decode step p50 from engine steps: 10.43 → **5.93 ms** (p99
6.03) — at step-profiling's ~5.9 ms GPU-busy figure; decode is now
GPU-bound. Replay rate over *pure decode* steps: **100.0%** in all four
cells (every "miss" in the whole-run rates of 93.6–98.5% is a
prefill/mixed step, eager by design; multi_turn is lowest only because it
is prefill-heaviest). Capture bill: **80 decode shapes in 3.4 s**
(server-2.log); nvidia-smi flat through the capture window
(12,443 → 12,447 MiB; both arms peak ~12.4 GiB during warm-up). Cell
warnings: TTFT p50 repeat-CV flags in both arms — noise, not correctness.

## Step 3 — A/B, long-context (`ab_5090_cudagraphs_longctx.toml`)

| cell              | eager tok/s | graphs tok/s | delta  |
|-------------------|------------:|-------------:|-------:|
| long_context c=1  | 93.1        | 175.2        | +88.2% |
| long_context c=2  | 147.7       | 207.9        | +40.7% |
| long_context c=4  | 177.0       | 207.3        | +17.1% |
| long_context c=4 (p1) | 175.0   | 201.9        | +15.3% |

Capture at this geometry: 120 decode shapes in 5.0 s. Decode-step replay
rate 100% in all cells. Aggregate-CV warnings on the c=4 cells (5.5–10.9%)
in both arms; the deltas dwarf them.

## Correctness (§7 gates)

- Bit-exact replay-logits test: passed (`torch.equal`, no tolerance).
- Greedy token-for-token equivalence: direct check on 0.6B, seven
  requests with prompt lengths 7–1023 (spanning kv spans and batch
  buckets), graphs arm 63 replayed / 6 eager steps: **all token streams
  identical** between arms.

## §6 predictions, graded

1. ~1900 API calls/step → <100: **confirmed** (1859 → 50).
2. 1-row step 9.3 → ≤2 ms: **missed** — 3.63 ms. Dispatch did collapse
   (9.00 → 0.06 ms); the residual is GPU execution now visible in
   fwd_sync, i.e. the prediction underestimated the 1-row GPU-busy
   floor, not the dispatch win.
3. 16-row p50 ~10.5 → 6-7 ms: **confirmed** (10.43 → 5.93, engine-side).
4. short_chat c=16 1469 → 1900-2300 (+30-55%): **exceeded** — 2466 (+69%).
5. long_context within noise: **wrong in the good direction** — +15% to
   +88%. The workload's 128-token decode tails were dispatch-bound too,
   especially at c=1-2 where prefill can't fill the gaps.
6. Hit rate ≥95% of decode steps: **confirmed at 100%** (decode-only,
   all cells, both configs).
7. Capture bill 40-90 s on the warm-up: **wrong in the good direction** —
   3.4 s / 80 shapes and 5.0 s / 120 shapes (~42 ms/shape). The
   prediction extrapolated eager warm-up cost; capture re-records
   already-warm kernels.
8. Shared-pool memory <1 GB: **confirmed** — no measurable nvidia-smi
   growth through capture at serve geometry; the larger 48-slot probe
   geometry (112 shapes) showed 1.03 GiB of graph private pools in a
   torch OOM report, consistent scale.

## Gates (§7): all passed

short_chat ≥ +20% → +69%; longctx > −3% → +15% worst cell; correctness
bit-exact + token-identical.

## Environment note

Mid-session, Ubuntu's updater upgraded the NVIDIA userspace driver to
580.173.02 under the still-loaded 580.159.03 kernel module, killing CUDA
init (error 804) for new processes. Worked around by pinning
LD_LIBRARY_PATH to the 580.159.03 libcuda/NVML extracted from the apt
cache, so the whole session ran a matched driver pair; results are
unaffected. The box needs a reboot to return to a matched installed stack.

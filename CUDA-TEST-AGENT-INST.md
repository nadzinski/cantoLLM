# 5090 validation: CUDA graphs (Phase 3)

Disposable instructions for the agent on the 5090 box (same pattern as the
shape-buckets round; delete this file when everything below is done and
recorded). Context: `cuda-graphs-design.md` (§6 predictions, §7 protocol).
The implementation is on main; nothing here changes code unless a step
fails.

## 1. Suite, including the CUDA-marked tests

```
uv sync && source .venv/bin/activate
python -m pytest tests/ -v
```

Everything must pass. The tests that only run here are in
`tests/test_graphed_forward.py` (`TestCaptureReplayCUDA`): capture/replay
vs eager is asserted **bit-exact** (`torch.equal`), the filler-padded step
must match eager and leave the pool identical, and the wiring test drives a
real request through `scheduler_from_runtime` and requires `hits > 0`.
If exactness fails but values are close, do not loosen the assert — record
the max diff and stop; that means the replay ran different kernels than
eager (likely a cuDNN plan mismatch: check the warm-up ran before capture).

## 2. Step profile recheck

`bench/profile_step.py` has no flags; it builds its own exact-geometry
config (no buckets, no graphs) — run it unmodified for the eager arm,
directly comparable to step-profiling.md. For the graphs arm, copy it to a
scratch file and change only the `BatchingConfig` construction: add
`**default_shape_buckets(...)` for its max_batch/budget, plus
`warmup_shapes=True, cuda_graphs=True` (leave `attention="padded"` so
graphs are the only axis that moves). Record against the §6 predictions:
`cudaLaunchKernel` calls/step ~1900 → <100 on decode steps; 1-row step
9.3 ms → ≤2 ms; 16-row p50 ~10.5 → ~6-7 ms. Two caveats: the probe's
post-forward sync hides submit-side wins (known from the finalize round;
trust engine-side numbers), and its `fwd_call` wrap times the graphed
wrapper on replayed steps — that is the number we want.

## 3. The A/B pair

```
canto bench run bench/configs/ab_5090_cudagraphs.toml
canto bench run bench/configs/ab_5090_cudagraphs_longctx.toml
```

Both matrices flip only `cuda_graphs`. Watch on the way through:
- the serve log line `CUDA graphs: captured N decode shapes in T s` —
  record N and T (prediction: T adds ~40-90 s to the warm-up bill), and
  nvidia-smi memory before/after capture (prediction: shared pool <1 GB).
- hit rate from StepStats `graph_replayed` over decode-heavy cells
  (prediction: ≥95%). It is in the recorded engine steps.
- gates: short_chat aggregate ≥ +20% (baseline arm ≈1469 tok/s at c=16),
  longctx > −3%.

## 4. Record and return

Commit the bench/history run directories as usual. Leave a summary (numbers
vs each §6 prediction, plus any anomaly) in the run log or a scratch file;
the results write-up (`cuda-graphs-results.md`) happens back home from your
numbers. If any capture fails at startup, the engine logs it and that shape
serves eager — note which shapes, don't work around it.

# 5090 validation: torch.compile (Phase 3)

Disposable instructions for the agent on the 5090 box (same pattern as the
graphs round; delete this file when everything below is done and recorded).
Context: `torch-compile-design.md` (§6 predictions, §7 protocol and gates,
§7's explain-day findings for what tracing already proved on CPU). Baseline
numbers to beat are the graphs A/B (`cuda-graphs-results.md`): short_chat
c=16 at 2466 tok/s, 16-row decode step p50 5.93 ms. The implementation is
on main; nothing here changes code unless a step fails.

## 0. Box sanity

The box was rebooted after the driver userspace/kernel mismatch of the
graphs round (2026-08-07). Before anything: `nvidia-smi` must run clean
and `python -c "import torch; print(torch.cuda.is_available())"` must say
True with no CUDA error 804. If the stack is mismatched again, stop and
report; do not pin LD_LIBRARY_PATH this time, a reboot is the fix.

## 1. Suite

```
uv sync && source .venv/bin/activate
python -m pytest tests/ -v
```

Everything must pass, including the CUDA-marked graph tests
(`TestCaptureReplayCUDA`: still bit-exact vs eager, graphs without
compile). The compile tests in `tests/test_torch_compile.py` run the
Dynamo layer only (backend="eager"); Inductor numerics are what the rest
of this protocol measures. If `fullgraph=True` raises anywhere on this
box (a CUDA-only graph break the CPU trace could not see), record the
full break message and stop; do not flip fullgraph off.

## 2. Step profile recheck

`bench/profile_step.py` unmodified is the eager arm (no buckets, no
graphs, directly comparable to step-profiling.md: ~1859 cudaLaunchKernel
calls/step at 16 rows). For the compile arm, copy it to a scratch file
and change only the `BatchingConfig` construction: add
`**default_shape_buckets(...)` for its max_batch/budget plus
`warmup_shapes=True, torch_compile=True`, and leave `cuda_graphs=False`
so fusion is the only axis that moves (graphs would collapse the launch
count and hide the kernel census). Attention stays as the script has it.
Record against §6:

- kernels/step at 16 rows: prediction 3 says ~1750 → 400-700.
- 1-row and 16-row step times and the fwd_call/fwd_sync split (the
  compiled fwd_call now includes Dynamo guard evaluation; record it).

Probe caveats from the last two rounds still apply: accumulated KV
histories inflate high-row fwd_sync, engine-side numbers are the
trustworthy ones, and if Phase B OOMs, a `gc.collect()` between phases
fixed it last time (scratch copy only).

## 3. The A/B pair

Cold-cache first: `rm -rf /tmp/torchinductor_$USER` (Inductor's on-disk
cache) so the first spawn of each compiled arm pays the true cold bill.
Then, with recompile logging on for the tripwire:

```
export TORCH_LOGS=recompiles
canto bench run bench/configs/ab_5090_compile.toml
canto bench run bench/configs/ab_5090_compile_longctx.toml
```

The short config is 4 arms (the cross-product duplicate baseline is
deliberate: it is the repeat-noise reference); longctx is 2 arms,
strategy dynamic. Watch on the way through:

- **Ready bill, cold**: per compiled arm, spawn→Ready from the server
  log, minus the baseline arm's. Prediction 7: cold compile adds
  40-150 s (dynamic), ~1.5-2x that for batch-bucket.
- **Ready bill, warm**: after both runs, manually spawn the
  compile-dynamic server once more with the same flags, record
  spawn→Ready, kill it. Prediction 7: under ~20 s of compile on a warm
  cache. Gate: warm bill under +30 s.
- **Recompiles**: after Ready, the compiled arms' server logs must show
  zero recompile lines from TORCH_LOGS (prediction 2). Any recompile
  during traffic is a guard leak; record the reason verbatim.
- **Artifact counts** at warm-up (the compile log lines): prediction 2
  says ~2-4 (dynamic), ~6-8 (batch-bucket).
- **Capture still works under compile**: the `CUDA graphs: captured N
  decode shapes in T s` line must still appear in compiled arms, and
  decode-step replay (StepStats `graph_replayed` in engine_steps) must
  stay 100% on pure-decode steps. If capture is invalidated under the
  compiled forward (cudaErrorStreamCaptureInvalidated: §4's
  capture-safety hazard), record which shape, then re-run the compiled
  arms with `cuda_graphs = false` as a clearly-labeled fallback so the
  fusion measurement is not lost.
- **Gates** (§7): short_chat c=16 aggregate ≥ +8% over the baseline arm,
  nothing regressing past -3% (longctx included), warm Ready bill under
  +30 s. Strategy question: prediction 6 says the two compiled arms land
  within 5% of each other; fewer artifacts wins a tie.
- nvidia-smi before/after warm-up per arm, for the record.

## 4. Greedy equivalence across arms

The CPU suite proved trace fidelity, not Inductor numerics, so the
correctness gate runs live (same protocol as the graphs round): three
manually spawned servers with the A/B's geometry, one per arm (default /
+compile dynamic / +compile batch-bucket), and the same seven greedy
requests with prompt lengths spanning the buckets (roughly 7 to 1023
tokens, temperature 0, ignore_eos, ~64 max_tokens). Token streams must
be **identical across all three arms** (prediction 8). If a stream
diverges, record the first divergent token index and both continuations;
that is a result, not a tweak-and-retry.

## 5. Record and return

Commit the bench/history run directories as usual, plus an
`agent-summary.md` in the first run dir: numbers against each §6
prediction (graded, wrong ones stated plainly), the gate outcomes, and
any anomaly. Do not edit the design note's predictions, do not touch
`viz/`, and the results write-up (`torch-compile-results.md`) happens
back home from your numbers. Delete this file in the final commit.

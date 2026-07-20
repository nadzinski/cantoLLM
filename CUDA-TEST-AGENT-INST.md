# 5090 test instructions: shape-vocabulary branch

**Branch: `shape-vocabulary` — stay on it.** Do not merge to main, do not
rebase, do not flip any defaults. This is an experiment Nadia wants measured
before committing to the design. Push your results to this branch.

## Context

sdpa-results.md (main, 2026-07-19) found cuDNN-SDPA correct but losing the
A/B end-to-end: cuDNN compiles a ~200 ms execution plan per distinct problem
shape, and the scheduler produced a new shape almost every step (the
`max_history_len` span grows token by token). This branch bounds the step
shapes to a fixed vocabulary and pre-warms it:

- **Prefill chunk widths** quantize to a menu inside the water-fill
  (mid-prompt chunks are menu-sized real tokens; the step width rounds up
  into {1} ∪ menu). `ContinuousBatchingScheduler._quantize_chunk`; the
  geometry padding lives at one boundary, `shaping.py::shape_step`.
- **KV spans** (`max_history_len`) round up to 256-token buckets, capped at
  the slot capacity — the causal mask fences the over-read.
- **Batch sizes** pad to power-of-two buckets with filler rows
  (`num_new == 0`: no KV write, reads slot 0 under the mask, output
  discarded) — a request joining or leaving lands on a known shape.
- **Warm-up** (`engine/batching/warmup.py`) runs one all-filler forward per
  vocabulary shape at engine build, behind the process split's Ready.

CLI: `canto serve --engine batched --shape-buckets --warmup-shapes`
(knobs derived by `default_shape_buckets`; see `BatchingConfig`).
Bench `[server]` accepts `shape_buckets = true` / `warmup_shapes = true`.

Everything is tested on CPU (tests/test_shape_vocabulary.py: output
equivalence vs the unbucketed scheduler, shape-property, filler safety,
warm-up coverage; 412 green on the Mac). What CPU cannot show: whether the
bounded vocabulary actually deletes the cuDNN plan-churn tail, what warm-up
costs in wall clock, and what bucketing overhead does to padded.

## Rules

- Attention math and scheduler policy are author-owned. Diagnose and
  report; don't redesign. Small test/instrumentation fixes are fine.
- If output equivalence fails anywhere (tokens differing between bucketed
  and unbucketed), STOP and report — that's a correctness bug in the
  branch, and benching a wrong engine is worthless.

## Do, in order

1. `git fetch && git checkout shape-vocabulary`. `uv sync` if needed.

2. Full suite: `.venv/bin/python -m pytest tests/ -q`. Expect ~415+ passed
   (the 3 CUDA tests in test_sdpa_equivalence.py now run here), 1 MPS skip.

3. **Warm-up timing, observed directly** (this is a headline number):

       .venv/bin/canto serve --engine batched --attention sdpa \
         --max-batch 4 --batch-max-seq-len 10240 --max-tokens-per-step 512 \
         --shape-buckets --warmup-shapes --device cuda

   The warmup module logs "warming N shapes" and "shape warm-up done: N
   shapes in X s". Record N (~477 expected at this geometry) and X. Sanity:
   with `--attention padded` warm-up should be dramatically faster (no
   plans to compile) — record that too. Kill the servers after.

4. The A/B: `.venv/bin/canto bench run bench/configs/ab_5090_shape_buckets.toml`.
   Two arms (padded+buckets, sdpa+buckets), same geometry and points as
   2026-07-19's ab-5090-sdpa run.

5. Four-way comparison against `bench/history/2026-07-19*_ab-5090-sdpa`
   (padded, sdpa — both unbucketed). Per cell: tok/s, TTFT p50, engine ITL
   p50/p99, step-duration p50/p99 from engine_steps. The three questions,
   in order of importance:
   a. Did sdpa+buckets lose the 200–300 ms p99 stall tail, and did
      short_chat TTFT return to ~0.03 s? (Plan churn eliminated?)
   b. sdpa+buckets vs padded (unbucketed): does warm cuDNN now win
      long_context? By how much? This is the number the whole SDPA design
      decision has been waiting for.
   c. padded+buckets vs padded: pure bucketing overhead (pad columns,
      over-read spans, filler rows). Predicted small — quantify it.

6. Also worth capturing while there: distinct-shape count per arm from
   engine_steps (rows + step widths + kv spans if derivable) — evidence the
   vocabulary held in production traffic, matching the CPU property test.

7. Write up in `shape-buckets-results.md` (same style as sdpa-results.md:
   verdict up front, tables, mechanism, options). Commit + push to the
   `shape-vocabulary` branch. Do not update PLAN.md or viz — the design
   isn't accepted yet; the write-up is the input to that decision, which is
   Nadia's.

# H100 day session notes (2026-08-14 PT / 2026-08-15 JST, Tokyo p5.4xlarge)

Instance: i-099b61f29aa0d7e4d, ap-northeast-1c, up 00:40 UTC. $8.60/hr.
Tree on node: rsync of pushed main 7848c6d + docs-only local edits
(h100-plan.md, infra/README.md Tokyo pivot). No .git on node, so run dirs
are stamped "nogit" — provenance = 7848c6d, engine code identical.

## Bring-up (prediction 1: CONFIRMED)
- DLAMI 20260814, driver 595.91.07, CUDA 13.2, torch 2.10.0+cu128 sees H100.
- uv sync clean. Suite: 448 passed, 15 skipped (3 MPS-only, hf-parity trio,
  incremental-decoder set — same skips as any Linux box), 107.8s, FIRST TRY.
- CUDA tripwires (graphed forward, torch_compile, sdpa kernel-ran) all ran+passed.
- 32B download: 62 GB / 17 shards in ~3 min (~350-500 MB/s). Beats prediction
  7's 4-10 min.

## Tier 1: ab_h100_compile (0.6B, 16x4096) — run dir 2026-08-15T004922
Baseline pair (repeat-noise ref): agg within 0.2% of each other. Clean.

| cell | baseline | +compile dynamic | 5090 baseline | 5090 dynamic |
|---|---:|---:|---:|---:|
| short_chat c=4  | 616.9 / 614.4 | 1178.7 (+91%) | 810.7 | 1307.6 |
| short_chat c=16 | 1855.0 / 1841.0 | 3124.8 (+69%) | 2461.2 | 3682.6 |
| code c=8       | 988.3 / 975.6 | 1790.5 (+82%) | 1365.6 | 2097.1 |
| multi_turn c=8 | 850.7 / 850.3 | 1545.4 (+82%) | 1149.1 | 1698.3 |
| TTFT p50 short c=16 | 100.5 ms | 59.2 ms | 66.2 ms | 40.2 ms |

- Prediction 2 (H100 does not clearly beat 5090 at 0.6B decode; dynamic c=16
  in 2900-3700): CONFIRMED — 3124.8, the 5090 holds by ~15%.
- Prediction 3 (TTFT p50 short c=16 in 28-42ms): MISS — 59.2ms, worse than
  5090's 40.2. H100 TTFT worse across the board at 0.6B. (long_context tok/s
  part of prediction 3 pending bench2.)
- Prediction 5 (Ready bills same-order, slightly worse; cold dynamic
  180-300s): MISS on magnitude — baseline Ready 203.7s (5090: 78.1; sweep
  ~2.6x slower), compile-dynamic cold Ready 646.7s (5090: 225.6; Inductor
  adds ~443s vs ~147s = ~3x). "Same-order" barely, "slightly" no.
- Relative compile win is BIGGER on H100: +69% c=16 vs 5090's +49.6%
  (both against own baseline). PLAN.md's "compile has more headroom on
  Hopper" reads TRUE in relative terms even though absolute loses.
- FAILURE: compile=true/batch-bucket arm never became healthy in 900s
  (health_timeout_s). Cause: ~2x artifacts x ~3x-slower EPYC Inductor
  ≈ 204 + 2x443 ≈ 1090s > 900. All 4 points lost. DEFERRED possible
  redo w/ raised timeout to the webchat-wait window; dynamic (the serve
  default) carries the cross-hw anchor.
- Validity warnings: 5 cells flag TTFT p50 CV 7.9-14.3% > 5% across
  repeats (both arms incl. baseline). 5090 round had zero. Aggregate
  tok/s CVs not flagged anywhere. Note for results: TTFT repeat noise is
  elevated on this host.
- Deviation log: ab_h100_compile_longctx.toml health_timeout_s raised
  900 -> 1800 on the node (before its compiled arm; lesson from the
  batch-bucket timeout). bench2 restarted clean 01:31 UTC.
- HF_HOME hub warning cosmetic (unauthenticated requests).

## Tier 1: ab_h100_compile_longctx (0.6B, 4x10240) — run dir 2026-08-15T0131xx
All cells recorded (timeout raise vindicated: compiled arm Ready 725.2s,
would have been a second squeaker vs 900). Baseline Ready 302.5s.

| cell | baseline | +compile dynamic | 5090 base | 5090 dyn |
|---|---:|---:|---:|---:|
| long_context c=1 | 110.0 | 221.5 (+101%) | 179.4 | 294.2 |
| long_context c=2 | 147.0 | 277.2 (+89%) | 207.6 | 285.1 |
| long_context c=4 | 180.7/182.3 | 321.1/318.7 (+77%) | 207.7 | 254.7 |

- Prediction 3 longctx part: MISS at c=1 (221.5 vs predicted 310-420; the
  5090's 294 wins) but H100 compiled WINS at c=4 (321 vs 255, +26%) —
  Hopper width needs occupancy to show. TTFT longctx c=1: 96ms compiled.
- 4 more CV warnings (TTFT 5.1-6.9%, agg 6.9-7.5% at c=4 pairs).

## profile_step passes (0.6B)
Eager (flagless): 16-row step 37.2ms total, fwd_call 35.4ms — pure CPU
dispatch, flat 1→48 rows, fwd_sync ~0.1ms (GPU starved). 1915
cudaLaunchKernel/step (5090: 1859 — model-shaped, prediction 4 holds),
~5.9µs/launch on this EPYC. The un-optimized engine would be WORSE on
H100 than on 5090; the whole Phase-3 stack is what makes this host viable.
Stack scratch (buckets+warmup+compile, graphs off): census 246
kernels/step (5090: 218; prediction 4 ~close, Hopper kernel names in
~/prof_stack.log tables). 16-row step 17.3ms, fwd_call 15.4ms (Dynamo
guards + launches; 5090 compiled fwd_call ~3.6ms → host dispatch ~4.3x
slower). In production graphs erase this; here it explains the bench gap.

## 32B smoke (default stack, 8x4096) — runbook step 6
- Cold Ready **1476s (24.6 min)**: load ~4 min + warm-up sweep 252 shapes
  w/ Inductor **1244.1s** + capture 64 decode shapes **9.5s**. Prediction 7
  (6-12 min) MISS on the same EPYC-Inductor axis as everything today.
- Peak 73.4 GB / 80 GB — prediction 6 (fits at 8 slots, 72-78 GB peak)
  CONFIRMED, OOM ladder untouched. expandable_segments set from first boot.
- Proof-of-life completion: coherent Qwen3-32B thinking tokens via OpenAI
  dialect, temp 0. First 32B ever served by this engine.
- Engine banner: max_batch=8, slot=4096, budget=512/step, sdpa, buckets,
  warmup, graphs, compile-dynamic. Capture-under-compile only 9.5s at 32B
  (the record-warm-kernels effect again).

## 32B A/B, short pair (8x4096) — runs 024635 (default) / 030326 (eager)
| cell | eager | default | win | 5090 0.6B equiv |
|---|---:|---:|---:|---|
| short_chat c=4 | 50.1 | 146.0 | +191% | (~+150% post-sink-fix) |
| short_chat c=8 | 97.0 | 280.2 | +189% | |
| code c=8 | 94.4 | 244.5 | +159% | |
- Prediction 9 (win shrinks to +30-70% at 32B): MISS, best direction —
  the win holds at ~2.6-2.9x. Premise failed: eager 32B is STILL
  CPU-taxed on this host (implied eager step ~82ms at c=8 vs prediction
  8's eager 26-45ms band — launch flood x ~6µs EPYC launches + python
  padded path). Stack matters MORE on server hosts, not less.
- Prediction 9 absolute band (default short_chat c=8 260-380): CONFIRMED
  280.2. TTFT p50 default c=8 174ms.
- Default-arm warm Ready 479s (cache: 1476 cold -> 479 warm). Recompile
  tripwire: ZERO recompile lines (TORCH_LOGS=recompiles) — gate passes.
- Eager Ready = load only. code cells: 1 CV warning each arm (host TTFT
  noise, same as all day).

## 32B A/B, longctx pair (2x10240) — runs 031537 (default) / 034419 (eager)
| cell | eager | default | win |
|---|---:|---:|---:|
| long_context c=1 | 11.4 | 30.9 | +171% |
| long_context c=2 | 15.3 | 44.4 | +190% |
- Prediction 9 longctx absolute (default c=1 35-55): NEAR MISS low (30.9).
- Default TTFT p50 c=1 330ms / c=2 790ms; eager 620ms / 2830ms.
- 1 CV warning per arm at c=2 (host TTFT noise pattern).

## 32B profile_step + floor (prediction 8 grading)
Probe (graphs off both arms):
- eager: 8-row step 78.3ms, fwd_call 76.7ms — 98% CPU launch dispatch;
  4267 launches/step (prediction ~4200 CONFIRMED exactly). GPU work
  hides fully under the launch stream (fwd_sync 0.6ms).
- compile/graphs-off: 35.6ms, fwd_call 33.9ms — STILL CPU-bound; 538
  launches/step (~2.3x the 0.6B 246, layer-scaled).
Production (from ab-32b-default run.json, graphs+compile, occ 0.98):
- step_dur_p50 27.5ms vs 19.6ms floor = +40% (prediction 20-24ms MISS);
  achieved BW 65.5GB/27.5ms = 2.38 TB/s = 71% of 3.35 (predicted 2.7-3.3
  MISS). p99 41ms (prefill-mixed), engine ITL p50 27.6ms.
- Graded: launch flood exact; eager band (26-45) badly missed (78 — the
  EPYC dispatch premise); floor distance 40% not <25%.
- THE MECHANISM HEADLINE: graphs are not an optimization on this host at
  32B — they are what switches the measurement from CPU to GPU. Kernel
  census tables in p32e.log/p32s.log (synced to scratchpad node-logs).

## Descope (recorded)
- 0.6B compile batch-bucket arm H100 redo: DROPPED (would collide with
  webchat window + budget; 5090 already answered the strategy question,
  within ~3%). The lost datum: Hopper strategy cross-check only.

## Timeline
- 00:40 instance up / 00:49 bench1 start / ~01:30 bench1 done
- 01:31 bench2 (longctx) start w/ timeout 1800 / ~02:05 done
- ~02:06 prof eager / ~02:10 prof stack / 02:20 32B smoke spawn
- 02:45 32B Ready+curl / 02:46 32B default A/B start (TORCH_LOGS tripwire on)
- 03:03 32B eager arm / 03:15 longctx pair chained start
- Ops gotcha x2: pkill -f patterns matching the ssh command's own bash -c
  line kill the session mid-script (exit 255) — use [b]racket patterns.

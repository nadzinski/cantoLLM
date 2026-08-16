# 5090 validation summary: production hygiene (Phase 3.5) — 2026-08-16

Agent run log for the Phase 3.5 validation round (runbook driven by the
Mac planning session; graded there against production-hygiene-plan.md §5).
Covers this run dir and its longctx sibling
`2026-08-16T142332_3d9acef_ab-5090-compile-longctx`. Both configs are
reruns of the Phase 3 A/Bs on sha `3d9acef` — metrics are always on now,
so these runs ARE the metrics-overhead measurement. Reference records:
`2026-08-08T222750_9b49283_ab-5090-compile` and its longctx sibling.
Environment: torch 2.10.0+cu128 (cuDNN 91002), driver 580.173.02
(CUDA 13.0 userspace), RTX 5090, same box (behemoth) as the references.

**Headline: the observability-overhead gate PASSES on every arm that
matters. Baselines and compile-dynamic (the serve default) are within
repeat noise of the Phase 3 records in both configs. One real deviation
surfaced: the compile-batch-bucket arm has silently lost its ~3%
step-time edge and now measures identical to dynamic on every metric —
a strategy-specific regression vs the 08-08 record, not metrics
overhead (details below). Suite 521/0 red, chaos 5/5, and the full
serve lifecycle (bind→warm→drain→kill-9 recovery) behaved exactly as
specified.**

## Suite + chaos

- `pytest tests/ -q`: **521 passed, 3 skipped, 5 deselected**, 54 s.
  All CUDA-only tests (sdpa tripwires, graphed forward, Inductor
  tripwire) ran and passed. The faulthandler_timeout=60 tripwire never
  fired — no stall around test_bench_executor's crash test this round,
  so no stacks to capture.
- `pytest -m chaos tests/chaos/ -v`: **5 passed**, 18.4 s.

## Serve lifecycle (0.6B, full CUDA default, --max-batch 16)

- Bind: `/health` 200 within 2 s of launch while `/ready` held 503 with
  progress JSON advancing load → compile → sweep 0-240 → capture 0-80.
- Time-to-ready: **83.4 s first boot, 81.7 s second boot** — the
  predicted cold-Inductor first boot never happened; both boots were at
  the warm bill. (Explained after the fact: Phase 3.5 never touched the
  traced forward region, so the cache keys survived the pull.)
- `/version` (sha 3d9acef, clean), `/metrics` (26 cantollm_* families,
  NVML gauges live), `/debug/engine-stats` (schema_version 1, contract
  unchanged): all good. Streaming chat end to end: good.
- **Drain: PASS.** SIGTERM landed mid-stream at 422/1022 SSE deltas of a
  512-token request; the stream ran to completion (message_stop, no
  error event); exit code 143.
- **Recovery: PASS.** kill -9 of the engine child (spawn_main) mid-stream
  at ~416/1022 deltas: stream closed cleanly on the open 200 connection
  with `error: engine failed: engine process died (exit code -9)`;
  `/ready` dropped to 503 restarting (retry_in 1.0 s) → full re-warm
  with progress → 200 with `generation: 2` in **80.6 s** (≈ one warm
  boot, as predicted); fresh request then succeeded.

## The A/B vs the Phase 3 records (medians of 3 repeats, aggregate tok/s)

Short config:

| cell | arm | 08-08 | 08-16 | Δ |
|---|---|---:|---:|---:|
| short_chat c=4 | baseline | 810.7 | 813.1 | +0.3% |
| short_chat c=16 | baseline | 2461.2 | 2470.6 | +0.4% |
| code c=8 | baseline | 1365.6 | 1348.8 | −1.2% |
| multi_turn c=8 | baseline | 1149.1 | 1148.7 | −0.0% |
| short_chat c=4 | compile-dynamic | 1307.6 | 1332.3 | +1.9% |
| short_chat c=16 | compile-dynamic | 3682.6 | 3595.9 | −2.4% |
| code c=8 | compile-dynamic | 2097.1 | 2083.2 | −0.7% |
| multi_turn c=8 | compile-dynamic | 1698.3 | 1695.4 | −0.2% |
| short_chat c=4 | compile-batch-bucket | 1370.3 | 1329.4 | −3.0% |
| short_chat c=16 | compile-batch-bucket | 3800.1 | 3594.8 | −5.4% |
| code c=8 | compile-batch-bucket | 2167.3 | 2077.7 | −4.1% |
| multi_turn c=8 | compile-batch-bucket | 1756.7 | 1697.1 | −3.4% |

Longctx config (all compile arms dynamic):

| cell | arm | 08-08 | 08-16 | Δ |
|---|---|---:|---:|---:|
| c=1 n=16 | baseline | 179.4 | 174.7 | −2.6% |
| c=1 n=16 | compile | 294.2 | 306.7 | +4.2% |
| c=2 n=16 | baseline | 207.6 | 206.8 | −0.4% |
| c=2 n=16 | compile | 285.1 | 289.6 | +1.6% |
| c=4 n=16 | baseline | 207.7 | 207.1 | −0.2% |
| c=4 n=16 | compile | 254.7 | 248.4 | −2.5% |
| c=4 n=32 | baseline | 202.9 | 201.2 | −0.8% |
| c=4 n=32 | compile | 242.8 | 244.1 | +0.6% |

**Gate (within ~2-3% repeat noise): PASS for baselines and
compile-dynamic in both configs** — longctx deltas scatter both
directions (−2.6% to +4.2%), short-config baselines sit at ±1.2%, and
dynamic's worst cell is −2.4% at short_chat c=16 (the highest-rate cell,
so if always-on metrics cost anything measurable, it is ≤ ~2% there and
inside noise elsewhere). Decode replay rate: **100.0% of pure-decode
steps in all cells, all arms** (from engine_steps), zero recompiles in
any server log.

## Deviation: compile-batch-bucket ≡ compile-dynamic now

The bucket arm fails the −3% line in all four cells (−3.0/−3.4/−4.1/
−5.4%) — but the signature is not overhead, it is convergence: bucket
now measures identical to dynamic everywhere (c=16: 3594.8 vs 3595.9
tok/s; 16-row replayed decode step p50 3.994 vs 3.984 ms; TTFT p50 49.5
vs 49.2 ms). In the 08-08 record bucket's baked row constants were worth
~3% (step 3.754 vs 4.009 ms). The arm is not misconfigured: its engine
log says strategy=batch-bucket, its warm-up paid the specialized-
artifact bill (sweep 328.8 s vs dynamic's 88.8 s), replay is 100%, and
`_mark_compile_dims` still pins the batch dim static. The captured
graphs are simply no longer faster than dynamic's. Timeline note: the
08-08 bucket numbers predate the sdpa execution-context hoist
(`a30f2ee`), which changed traced-graph content and was only ever
re-validated on the dynamic strategy — so this regression may have
arrived with the hoist rather than with Phase 3.5. Not chased further
this round (measure-and-report protocol). Practical impact: none for
serving (the default is dynamic), but batch-bucket currently buys
nothing for 2-4x the warm-up bill.

## Other observations

- short_chat c=16 TTFT p50 on the dynamic arm: 40.2 → 49.2 ms (+22%)
  with aggregate tok/s unchanged; both compile arms now sit at ~49.3 ms
  and the baseline pair at 69.2/66.4 (was 66.2/66.4). Plausibly the new
  admission/metrics work in the request path; noted, not chased.
- The traced arm (OTEL_EXPORTER_OTLP_ENDPOINT set, 100% sampling) and
  the observability-stack verification were pending docker access when
  this summary was first written; the traced run dir and its comparison
  land separately once the stack is up.

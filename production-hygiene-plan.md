# Phase 3.5: production hygiene (design + plan)

The merged design note and execution plan for Phase 3.5, decided 2026-08-15 in a
planning session (all decisions the author's; implementation delegated to Claude in
review-sized chunks). PLAN.md's Phase 3.5 section carries the forward-looking scope;
this doc records the decisions, the alternatives considered, the architecture, and the
execution order.

## 1. Goal

Pick up the production basics deferred since Phase 1b, now that the engine is a
separate process and its metrics mean something. The concrete pain this phase removes:

- During warm-up (78 s warm on the 5090, up to 24 min cold on the H100) the server
  socket is dead; the bench blind-polls `/health` with an 1800 s timeout, and one H100
  arm lost all its points to exactly that.
- Nothing bounds the scheduler's waiting queue; 10,000 requests can pile up.
- An engine crash leaves a zombie API process serving errors until a human restarts it.
- Shutdown truncates in-flight streams.
- No metrics surface exists beyond the bench-oriented `/debug/engine-stats`.

## 2. Decisions and alternatives

**Metrics: Prometheus + Grafana, not OTel metrics.** The inference-engine ecosystem
(vLLM, SGLang, TGI) exposes Prometheus `/metrics` as the primary surface; the existing
`EngineStatsAccumulator` plumbing is already pull-shaped. OTel metrics would add SDK
ceremony and a collector container between us and the same Grafana dashboard.
Considered and rejected: full OTel pipeline (collector teaches deployment plumbing,
not engine content); dcgm-exporter + node_exporter (profiling-class DCGM metrics are
historically unavailable on GeForce; in-process NVML + psutil covers what the
dashboard needs with zero extra containers).

**Tracing: OTel request-phase spans, direct OTLP to Tempo.** One span tree per
request: HTTP root; children tokenize, queue-wait, per-chunk prefill, one decode span
with token/step attributes and a first-token event. Context propagates over the IPC
boundary on `InferenceRequest`. 100% sampling (it is a lab; keep every trace). No
collector (Tempo ingests OTLP directly); Tempo over Jaeger so traces live in the same
Grafana as the metrics. Rejected: per-step engine spans (steps are many-to-many with
requests and run at 100+/s; step timing already lives in the metrics histograms), and
span links machinery for the batch/request relation.

**Admission: per-model semaphore, capacity-derived default.** The bound lives where
capacity lives (the registry entry), default 4 x max_batch, queue-with-timeout then
429 + Retry-After estimated from live stats. Rejected: a global fixed cap with
immediate 429 (bursts that would clear in 200 ms eat rejections) and token-aware
admission (belongs to Phase 4/5 when KV headroom is a first-class number).

**Readiness: early accept + progress.** uvicorn binds immediately; engine
construction moves behind a background supervisor task; `/ready` 503s with warm-up
progress (stage + done/total) until the engine's Ready, then 200; inference endpoints
503 with the dialect-correct envelope while not ready. `/health` stays dumb API
liveness. Rejected: boolean `/ready` (long cold starts stay opaque) and keeping the
blocking startup.

**Shutdown: drain on SIGTERM and first Ctrl-C, second signal forces.** Stop
admitting, in-flight (including engine-queued, since the client already holds a
stream) run to a configurable 30 s deadline, then abort through the normal event path
so clients see a terminal abort. Matching uvicorn's own first-graceful-then-force
convention means the drain path gets exercised on every dev stop.

**Reload: drain-then-load only.** `POST /admin/reload` drains, frees VRAM, rebuilds
and re-warms the same model + config, with `/ready` reporting progress. Blue-green
within one GPU was rejected: physically impossible at the sizes that matter (32B was
73 GB of the H100's 80). Switching models means a server restart via the config file;
hot-swapping registrations would drag in registry mutation and `/v1/models` churn for
no current need.

**Recovery: supervisor with capped backoff.** Engine death -> in-flight fail cleanly
(existing behavior) -> auto-rebuild through the reload path, backoff 1 s doubling to a
30 s cap, give up after 5 consecutive failures into a `crashed` state (503 with the
error; `POST /admin/restart` recovers). The cap matters because a deterministic
crasher with a multi-minute warm bill would otherwise loop at full GPU burn.

**Watchdog: no-step-progress detection, subprocess engine only.** A hung-but-alive
child (process up, zero steps, work pending) is invisible today. The watchdog arms
only while the engine is ready (structurally cannot fire during warm-up), and fires
when the engine has in-flight work but no step progress for the timeout (default
60 s; steps are sub-second after warm-up, so the margin is enormous). It kills the
child (SIGKILL; a hung child may ignore SIGTERM) and lets the supervisor rebuild. A
thread cannot be killed, so the in-process engine gets no watchdog.

**Extras folded in:** OpenAI mid-stream error parity (Phase 1b leftover),
`X-Request-ID` middleware + structured JSON logs in both processes (the other 1b
leftover), a TOML serve config file with CLI overrides (the flag set has outgrown the
command line), and a `/version` endpoint. Considered and dropped: a systemd unit,
per-request deadlines (Phase 4, with priorities and the goodput metric), the
raw-tokens NDJSON endpoint, auth/rate limiting (standing non-goal: sidecar territory).

**Validation: a pytest chaos suite, not a runbook.** Registered `chaos` marker,
excluded from default runs; the bench loadgen drives load as a library; scenarios
cover the load-test exit criterion plus every failure mode this phase builds
machinery for. A hand-run procedure would rot the day Phase 4 reshapes the scheduler.

## 3. Architecture: the lifecycle core

The one new architectural piece. Everything else in the phase hangs off it.

**`EngineHandle`** (new `src/cantollm/lifecycle.py`): per-model state machine with
states starting / warming / ready / draining / restarting / crashed / stopped. The
handle is the stable identity across engine generations; the engine object is
disposable (the event multiplexer's failure latch is one-way by design, so every
restart builds a fresh engine and swaps it in). The handle owns: the factory closure
that builds an engine (run via `asyncio.to_thread`), the current engine + runtime +
generation counter, the warm-up progress snapshot, failure accounting, the
`EngineStatsAccumulator` (moved here so bench scrapes survive restarts; a seq rebase
keeps the `since` cursor monotonic across generations), an API-side in-flight
counter, and the supervisor + watchdog tasks. All mutations happen on the event loop,
so routers read state lock-free and capture the engine object once per request.

**Supervisor loop** (one asyncio task per handle): build -> warm (progress flowing) ->
ready -> await a wake event (died | reload | restart | stop). First startup is simply
the loop's first iteration, which is why a bad model no longer kills the process: it
lands in `crashed` with the error visible in `/ready`, recoverable via
`/admin/restart`. On reload/death the old engine is fully shut down (child joined,
queues closed) before the rebuild, so there is never a moment with two engines and
double VRAM.

**Progress protocol:** a new pre-Ready `Progress(stage, done, total, detail,
elapsed_s)` IPC message; stages load / compile / sweep / capture. A leaf sink module
with a ContextVar callback instruments the existing loops (weight-load steps, the
warm-up sweep's materialized vocabulary, the capture list), throttled to stage
changes + 0.5 s. The in-process path inherits the sink through the thread's context
copy, so both engine modes report the same way.

**Signals:** a `uvicorn.Server` subclass overriding `handle_exit` (the single funnel
uvicorn routes both SIGINT and SIGTERM through; verified against uvicorn 0.44).
First signal starts the drain choreography; second forces. A
`timeout_graceful_shutdown` backstop replaces today's behavior where one open SSE
stream holds shutdown open forever.

## 4. Execution order

Chunks are review-sized and land green individually.

1. Foundations: deps (prometheus-client, opentelemetry-sdk + OTLP/HTTP exporter,
   nvidia-ml-py, psutil), `obs/` package, JSON logging + request_id contextvar,
   `X-Request-ID` middleware, `/version`.
2. Lifecycle core, in five sub-chunks: (a) handle + background start + `/ready` +
   503 gating; (b) Progress protocol; (c) drain + signals; (d) supervisor backoff +
   admin endpoints; (e) watchdog + stats continuity.
3. Admission control (semaphore, 429 + Retry-After).
4. `/metrics` (engine, request, HTTP, process, GPU, event-loop lag).
5. OTel tracing (API spans + scheduler lifecycle observer + IPC context propagation).
6. OpenAI stream-error parity.
7. Serve config file (TOML, CLI overrides, bench conventions).
8. Observability stack (docker-compose: Prometheus, Grafana, Tempo; provisioned
   dashboard, light theme).
9. Bench switches from `/health` polling to `/ready` (and learns that a `crashed`
   ready-state means spawn failure).
10. Chaos suite (50-client clean run, overload 429s, drain under load, kill -9
    recovery, watchdog hang).
11. Docs, PLAN.md status, viz Roadmap close-out, phase-end 5090 bench.

## 5. Predictions (to grade at phase end)

1. Observability on (metrics always; tracing exporting at 100%) costs within repeat
   noise on the standard bench configs: short_chat c=16 within 2% of the Phase 3
   record (3683 tok/s), long_context c=1 within 2% of 294.
2. The drain path passes chaos scenario 3 with zero truncated streams at 50 clients
   on the first 5090 run (Mac shakedown bugs allowed).
3. Supervisor recovery from kill -9, including re-warm, completes in under 120 s on
   the 5090 with the full CUDA default stack (warm Inductor cache).
4. The watchdog produces zero false positives across the whole phase's bench + chaos
   runs at the 60 s default.
5. `/ready` progress makes the bench's `spawn_to_ready_s` honest and the 1800 s H100
   health timeouts removable (not removed this phase; noted for the next H100 day).
6. Total new runtime dependencies stay at 5 (the four obs libraries + psutil), and
   Mac/CPU dev needs none of the compose stack running for any test to pass.

## 6. Results

### Implementation round (2026-08-16, Mac/CPU): every chunk landed

Chunk log (one commit each, suite green at every step; 513 tests + 5 chaos at
the end, from 452 at phase start):

1. Foundations `ea9ccc7`: deps, JSON logging + request_id contextvar in both
   processes, X-Request-ID middleware end to end, /version.
2. Lifecycle core, five commits: handle + background start + /ready + 503
   gating `9832c29` (the bench /ready switch, planned as chunk 9, folded in
   here because the bench tests break the moment startup goes non-blocking);
   Progress protocol `4d9f8da`; drain + signals `08586d0`; supervisor
   backoff + admin endpoints `c9440d3`; watchdog + stats continuity
   `4b961a9`.
3. Admission control `5e42b2e`. 4. /metrics `0614f7f`. 5. OTel tracing
   `4e752ae`. 6. OpenAI stream-error parity `d8be454`. 7. Serve config file
   `7cf2f3d`. 8. Observability stack `17ce18c`. 10. Chaos suite `b197c78`
   (all five scenarios green on CPU, including a real SIGTERM drain, a real
   kill -9 of the engine child with supervisor recovery, and a watchdog
   catch of a wedged child).

Deviations from the written design, all deliberate:

- The API root span is created in the router and ended by the stream
  wrapper, not by a middleware: a middleware-owned span would end when the
  handler returns, which for streaming responses is before the first token
  exists. The span now covers the request's true lifetime.
- Engine-side spans ride the drive loop (a TraceStepObserver beside the
  existing StepStatsCollector) rather than hooks inside the hand-written
  scheduler: the loop's before/after snapshots already see promotions,
  position diffs, and terminal events, so the scheduler needed zero changes
  and span resolution is per step, which is the honest granularity anyway.
- No fabricated finish_reason on the OpenAI error path: the dialect defines
  no error finish_reason, and the existing contract test explicitly pins
  the error envelope as the terminal event. Parity gained: the usage chunk
  ships on the error path when requested, and the error type is
  consistently server_error.
- Two env-gated fault hooks landed in the drive loop for the chaos suite
  (wedge-after-N-steps, per-step delay): the tiny model generates faster
  than a client can observe, so outside-timed fault injection needed a
  pacing lever. Test-only, off unless the variables are set.
- prometheus_client instruments live in a per-app CollectorRegistry, not
  the global default: the suite builds dozens of apps per process.

Known issue, diagnostics armed: an intermittent stall (minutes, then
self-recovery; test passes, suite green) around the bench executor's
server-crash test, seen a handful of times across full-suite runs and never
at the pre-phase commit in 12 tries. `faulthandler_timeout = 60` is now in
the pytest config, so the next occurrence dumps every thread's stack into
the run output and identifies itself.

### 5090 round (2026-08-16): complete, gates pass

Run by the box session over Remote Control, driven from the Mac session.
Environment: RTX 5090, driver 580.173.02 (CUDA 13.0 userspace), torch
2.10.0+cu128, cuDNN 91002, Python 3.11.13. Records:
`bench/history/2026-08-16T*_f8a826a_ab-5090-compile*` (untraced pair, with
the round's agent-summary.md) and the traced sibling under `547705b`.

- Suite on the box: 521 passed, 3 skipped (the CUDA-only tests all ran).
  The intermittent executor stall did not reproduce; no stacks captured.
- Chaos: 5/5, first run.
- Lifecycle on the real stack: `/health` answered within 2 s of launch
  while `/ready` 503'd with progress advancing load -> compile -> sweep
  0-240 -> capture 0-80; time-to-ready 83.4 s / 81.7 s (both warm: the
  pull left the Inductor cache valid because Phase 3.5 never touched the
  traced forward region). Drain: SIGTERM at 422/1022 deltas of a
  512-token stream, ran to message_stop with no error event, exit 143.
  Recovery: kill -9 of the engine child mid-stream produced a clean
  "engine failed" error event on the open stream, then a full re-warm to
  generation 2 in 80.6 s.
- Observability: docker.io + docker-compose-v2 installed (approved);
  Prometheus serving cantollm samples, Tempo returning traces with the
  full span set (root + tokenize on cantollm-api; queue / prefill chunk /
  decode on cantollm-engine), Grafana healthy with the provisioned
  dashboard. Two follow-ups landed on the box at Nadia's request: a
  VRAM-in-GiB panel (540d885) and Tempo's metrics-generator enabled so
  the Traces Drilldown app works (e72f885). Doc quirk: Tempo's
  /api/search returns empty without explicit start/end params.
- Bench gate (untraced, medians of 3 vs the 2026-08-08 records): PASS
  for baselines and compile-dynamic, the serve default. Short config:
  baselines within +-1.2%, dynamic -2.4% at the hottest cell (short_chat
  c=16, 3595.9 vs 3682.6) and inside +-2% elsewhere; longctx scattered
  -2.6% to +4.2% both directions (c=1 dynamic 306.7 vs 294). Always-on
  metrics cost <= ~2% at the hottest cell, noise elsewhere. 100% decode
  replay everywhere, zero recompiles.
- Traced arm (OTLP into live Tempo, 100% sampling): within noise of the
  untraced run (c=16 dynamic -0.0%; the traced run's own baseline pair
  disagreed with itself by 4-5%, bounding the scatter as environmental).

Two observations recorded, not chased (candidates for a later look):

1. The compile-batch-bucket arm converged to dynamic exactly (c=16 3594.8
   vs 3595.9 tok/s; replayed 16-row step 3.994 vs 3.984 ms) and so fails
   the noise line vs its own 08-08 record (-3.0 to -5.4%) while remaining
   identical to the default. Its 08-08 record predates the a30f2ee
   sdpa-pin hoist and bucket was never re-benched after it; the ~3% edge
   may have been lost there, not in 3.5. No default is affected.
2. Dynamic-arm TTFT p50 at short_chat c=16 rose 40.2 -> 49.2 ms (+22%,
   +9 ms) with throughput unchanged; both compile arms now sit at
   ~49.3 ms. Plausibly the new request path (admission acquire, metrics
   middleware, JSON logging). Aggregate gates all passed; worth a
   profile if TTFT matters in a later phase.

### §5 predictions, graded

1. Observability within noise, short c=16 within 2% of 3683 and longctx
   c=1 within 2% of 294: **mixed**. Longctx beat its anchor (+4.2%);
   short c=16 landed at -2.4%, just past the 2% clause, though inside the
   round's own observed repeat noise (the traced run's identical baseline
   pair differed by 4-5%). The spirit (no measurable observability tax)
   holds; the letter missed by 0.4 points at one cell.
2. Drain chaos passes first 5090 run with zero truncation: **confirmed**
   (all five scenarios, first run).
3. Kill-9 recovery incl. re-warm under 120 s: **confirmed** (80.6 s).
4. Zero watchdog false positives across the round: **confirmed** (never
   fired outside the chaos scenario that provokes it).
5. `/ready` makes spawn_to_ready_s honest; 1800 s blind timeouts become
   removable: **confirmed** (progress observed end to end through real
   warm-ups; removal stays deferred to the next H100 day as written).
6. New runtime deps stay at 5 and Mac dev needs no compose stack:
   **confirmed**.

# Bench harness spec

**Status: landed 2026-07-12.** This is the Phase-0 "benchmark harness spec" deliverable
(open since 2026-04-19), written alongside the full implementation rather than ahead of
it. It pins the definitions the harness implements: workloads, load models, metrics,
run protocol, instrumentation contract, results schema, and reporting format. When a
number appears in `bench/history/`, this document is what it means.

Purpose (scoped during planning): the project's **phase-gate measurement instrument** —
characterization curves, reproducible baselines, an append-only in-repo history, and
run-validity checks. Explicitly *not* production observability (Phase 3.5 owns
`/metrics`), *not* CI perf-gating, *not* model-quality evals (Phase 9).

---

## 1. Workloads

Four prompt sets, committed under `bench/workloads/` as JSONL. The committed files are
the pinned truth; the authoring process (LLM-written at build time) is provenance, not
something a run re-executes.

| set | shape | targets |
| --- | --- | --- |
| `short_chat` | single-turn chat questions | ~30–120 input tokens |
| `long_context` | document + question | ~2k and ~8k input tokens (two bands) |
| `code` | code-writing / code-reading tasks | ~200–800 input tokens |
| `multi_turn` | one request with embedded multi-turn history | 4–8 turns, ~500–1500 input tokens |

~50 prompts per set (`long_context` may carry fewer). `multi_turn` is a *single*
request carrying prior turns in `messages` — interactive replay is deferred until
prefix caching (Phase 5) makes it measurable as something other than repeated prefill.

**File schema** (`schema_version` 1): line 1 is a meta record —
`{"schema_version": 1, "set": ..., "tokenizer": "Qwen/Qwen3-0.6B", "generated": "YYYY-MM-DD", "shared_prefixes": {name: text, ...}}` —
followed by one record per prompt:
`{"id": "short_chat-007", "messages": [...], "system": null, "prefix": null, "input_tokens": 43, "tags": [...]}`.
`messages` uses the OpenAI shape (the default dialect). `input_tokens` is stamped by
`canto bench verify-workloads` using the real Qwen3 chat-template tokenization — if the
template or tokenizer changes, re-verify and the diff shows in git.

**Shared-prefix knob** (Phase-5 hook, dormant in v1): a prompt may reference a named
entry in the meta record's `shared_prefixes`; the loader prepends it at request-build
time. v1 sets ship with unique prefixes (no accidental cache-friendliness); a Phase-5
variant can flip prompts onto common prefixes without new files.

Workload *variants* are config flags, not separate files: natural-stop
(`ignore_eos = false`) and sampled (`temperature/top_p`) runs reuse the same sets.

## 2. Load models

- **Closed-loop concurrency sweep** (primary). For each level `c` in the configured
  list: `c` workers share a seeded, shuffled prompt iterator; each worker sends its
  next request as soon as the previous completes, until the level's
  `requests_per_level` count is exhausted. Recommendation: `requests_per_level ≥ 8×c`
  so edge (fill/drain) effects are diluted — the aggregate window is not trimmed (§3).
- **Open-loop arrival schedule** (knee-finding). One scheduler task fires requests at
  seeded arrival times — exponential gaps (`poisson`) or constant (`fixed`) at each
  configured `rate_rps` — regardless of completions, up to `total_requests`, capped by
  `max_inflight` (hitting the cap is a validity warning: the load generator, not the
  server, became the bottleneck). Per-request **dispatch lag** (actual send − scheduled
  arrival) is recorded; lag p99 > 100 ms flags the generator as overloaded.

Closed-loop answers "what does the system do at occupancy N"; open-loop answers "where
does it fall over as offered load rises." Closed-loop under-reports tail latency near
saturation (coordinated omission) — that is why it is not the only mode.

## 3. Metric definitions (client side)

Timestamps per request, on the client's `time.perf_counter()`:
`t_send` (request written) → `t_headers` (response headers received) →
`t_first_token` (first SSE delta carrying `content` **or** `reasoning_content` —
thinking tokens count; Qwen3 leads with them) → per-chunk arrival times → `t_done`
(stream closed).

- **TTFT** = `t_first_token − t_send`. Includes tokenization, admission, scheduler
  queue, and (chunked) prefill. Percentiles: p50/p90/p99.
- **Accept latency** = `t_headers − t_send` (tokenize + admission + submit; the
  scheduler queue is *not* included — headers flush when the stream starts). Sanity
  metric only.
- **Completion latency** = `t_done − t_send`. p50/p90/p99.
- **Per-request decode rate** = `output_tokens / (t_done − t_first_token)`, defined for
  `output_tokens ≥ 2`.
- **Aggregate throughput** = `Σ output_tokens / (max t_done − min t_send)` over one
  repeat of one cell (untrimmed window; see §2's request-count rule).
- **Client ITL (mean, sanity only)** = `(t_done − t_first_token)/(output_tokens − 1)`.
  Per-chunk gap *distributions* are deliberately not reported for the batched engine:
  events cross IPC per engine step and fan out as SSE bursts, so client gaps measure
  delivery batching, not decode cadence. (Sequential engine streams token-by-token, so
  there the client-side gaps are honest — still reported only as the mean.)
- **Token counts** from the server's `usage` object (`stream_options.include_usage` on
  the OpenAI dialect), not client-side counting.
- **Finish-reason distribution** and **error counts** per cell.

Percentile method: linear interpolation on sorted values (as `bench.py` did).

## 4. Metric definitions (engine side, batched engines only)

Collected per scheduler step at the engine-shell layer (`drive_scheduler`), riding the
existing one-pickle-per-step event batch as `StepUpdate(events, stats)`. The
hand-written scheduler is not modified; everything derives from its public state
(`queued`, `active`, `allocator`, `config`).

`StepStats` fields: `seq` (engine-lifetime step counter), `t_wall` (`time.time()`,
coarse alignment only), `t_perf` (child `perf_counter` at step end), `dur_s`
(perf-clock time inside `scheduler.step()`), `rows` (sequences in this step's forward),
`occupied_slots`, `queue_depth` (post-command-drain, pre-step), `kv_tokens`
(Σ per-row positions), `prefill_tokens` / `decode_tokens` (consumed this step).

- **Step time**: distribution of `dur_s`; report p50/p99 and the prefill-heavy vs
  pure-decode split (a step is *pure decode* iff `prefill_tokens == 0`).
- **Engine ITL (primary ITL metric)**: per request, the gaps in `t_perf` between
  consecutive steps that emitted a token event for that request. Includes stalls from
  other rows' chunked prefill — deliberately: that is the latency a request actually
  experiences. Child perf-clock deltas only; engine and client clocks are never mixed
  (only `t_wall` gives coarse cross-process alignment).
- **KV utilization**, both definitions reported: **slot occupancy** =
  `occupied_slots / max_batch`; **token fill** = `kv_tokens / (max_batch × max_seq_len)`.
- **Occupancy / queue-depth time series** are step-indexed; idle periods emit no steps,
  so any time-weighted average weights by `t_wall` deltas (measured windows are
  saturated, making this a footnote — but it is the rule).

Sequential engine: no step loop, no engine stats (`/debug/engine-stats` reports
`available: false`); it is characterized by client metrics only.

**Instrumentation contract**: the API process accumulates `StepUpdate`s in a ring
buffer (4096 steps; engine-ITL samples 65536) on the engine's `EventMultiplexer` and
serves `GET /debug/engine-stats?model=<name>&since=<seq>` — steps with `seq > since`,
ITL samples, capacity, totals, `load_seconds` (from `Ready`), `next_since`. The harness
scrapes at ~1 Hz during measurement; a gap in `seq` (ring overflow) is a validity
warning. The endpoint is always on (localhost tool; Phase 3.5 owns the real
observability posture). Wire-protocol amendment recorded in `process-split-design.md`.

## 5. Run protocol

A **run** executes a TOML config (`bench/configs/*.toml`): a server-config matrix ×
`[[points]]` (workload, mode, levels, overrides) expanded into an ordered list of
**cells** (one workload × one level × one server config).

- **Server lifecycle**: spawn `canto serve` per *server config* (not per cell); wait on
  `/health` (`health_timeout_s`, generous — first run may download weights); record
  spawn→ready wall time and the engine's `load_seconds`. Between cells on the same
  server config, a **drain barrier**: poll engine stats until `queue_depth == 0` and
  `occupied_slots == 0` (sequential: short idle wait). `respawn = "per-cell"` opts into
  full isolation at model-load cost. `--attach --url` skips lifecycle entirely (also
  the vLLM-comparison path; engine stats then simply unavailable).
- **Sampling defaults**: greedy (`temperature = 0.0`), `ignore_eos = true`,
  fixed `max_tokens` (baseline tables use 128 unless the config says otherwise) — so
  token counts are identical across engines, configs, and phases, and numerics-level
  changes (SDPA, compile) cannot smuggle length changes into throughput numbers.
  `ignore_eos` is a request-level field on both dialects (vLLM-precedent name);
  combining it with stop sequences is a 400. Two standing variants (per workload
  config, same prompt files): natural-stop (finish-reason realism) and sampled
  (sampler-path cost).
- **Warmup**: `warmup_requests` (default 8) per server spawn, recorded with
  `excluded: true`, never aggregated.
- **Repeats**: each cell runs **3 measured repeats**; report per-repeat summaries and
  the **median across repeats** for headline tables. Coefficient of variation across
  repeats > 5% on aggregate tok/s or TTFT p50 ⇒ validity warning on the cell.
- **Dialect**: OpenAI (`/v1/chat/completions`) default; per-point `anthropic` override;
  every baseline config includes one small cross-dialect parity cell.
- **Validity rules** (attached to cells in `run.json`): request error rate > 1% warn,
  > 50% cell fails; stats `seq` gap warn; open-loop dispatch-lag p99 > 100 ms warn;
  `max_inflight` saturation warn; CV rule above; unexpected finish reasons under
  `ignore_eos` (anything but length/max_tokens) warn.
- **Failure handling**: server crash fails the cell (reason + log tail), teardown,
  continue (`stop_on_cell_failure = false` default). Abort (Ctrl-C or control panel)
  cancels in-flight streams (server frees slots via its disconnect→abort path), tears
  down, persists partials with `status: "aborted"`.

## 6. Results schema & storage

`bench/history/<run_id>/`, `run_id = <YYYY-MM-DDTHHMM>_<gitsha7>_<config-name>`:

- `run.json` — `schema_version`, `run_id`, `status`, `env` (git SHA + dirty flag,
  platform, Python/torch versions, device string + name, CPU count), fully-resolved
  config, per-cell blocks (server config, workload + its file hash, level, per-repeat
  summaries, median summary, validity warnings, spawn/ready/load timings), wall-clock
  bounds. Rewritten atomically after every repeat — it doubles as the live/partial
  state for the UI and for crash recovery.
- `requests.jsonl.gz` — one record per request (repeat-tagged): timestamps of §3,
  token counts, finish reason, error, dispatch lag (open-loop), `excluded` flag.
  **No output text** (a `--capture-text` debug flag writes a separate, gitignored
  file).
- `engine_steps.jsonl.gz` — scraped `StepStats` + derived engine-ITL samples.
- `logs/` — server stdout/stderr per spawn.

Committed to git per the PLAN.md dashboard-history commitment (text-only, a few hundred
KB per run). `schema_version` bumps on breaking shape changes; comparisons refuse
mismatched versions.

## 7. Reporting format

**Baseline table** (per model × engine): rows = workload × level; columns = TTFT
p50/p90/p99, completion p50/p99, aggregate tok/s, engine ITL p50/p99, mean occupancy,
mean KV token-fill, validity badges. **Comparison** (two runs): same table joined on
(workload file hash, mode, level, server-config hash) with absolute + % deltas;
refuses to join across differing workload hashes or schema versions. Rendered by the
control panel (`canto bench ui`, port 8002 — launch, live view, history, compare;
light mode) and, for the headline tables, printable from the CLI.

## 8. The Phase-2 close-out baseline (5090)

`bench/configs/baseline_5090.toml`, roughly (tune after the Mac smoke run):

- Models: **0.6B** (continuity with the MPS rough numbers) and **4B** (realistic
  compute). Engines: sequential and batched (process split, the production shape).
  Batched server matrix on 0.6B: `max_batch ∈ {8, 16}`, `max_tokens_per_step = 256`;
  4B stays at `max_batch = 8`.
- Closed-loop: `short_chat`, `long_context`, `code` at `c ∈ {1, 2, 4, 8, 16}`
  (sequential capped at `c ∈ {1, 2, 4}` — it serializes anyway), 128 output tokens,
  greedy, `ignore_eos`.
- Open-loop: `short_chat` on batched 0.6B, `rate_rps` swept to find the knee.
- Variants: natural-stop `short_chat` c=8; sampled `short_chat` c=8; one
  Anthropic-dialect parity cell.

These numbers become Phase 3's "before". Phase 3 re-runs the same config with SDPA
(and later compile/CUDA-graph variants) and compares in the panel.

## 9. Non-goals & future hooks

- **Not** `/metrics`/Grafana (Phase 3.5 — it will likely reuse the step-stats
  plumbing), **not** CI gating, **not** quality evals.
- vLLM side-by-side (Phase 3+): `--attach` + the OpenAI dialect already suffice;
  client metrics only.
- Prefix caching (Phase 5): flip `shared_prefixes` on; add cache-hit metrics to
  `StepStats` then (schema_version bump).
- Paged KV (Phase 4): the prefill/decode derivation leans on two current invariants
  (natural finish ⇒ decode row; finished rows freed in-step) — revisit the collector
  when block tables land.

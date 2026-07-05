# viz/ — interactive architecture explorer

A self-contained browser visualization of how cantoLLM works, for refreshing
your memory of the architecture. Open it directly (no server needed):

```
open viz/index.html
```

Four views, linked by the nav bar (semantic zoom — the overview stays sparse,
detail lives behind the zoom targets):

- **Overview** — the request path through `src/cantollm/` (clients → FastAPI →
  registry → SequentialEngine → StandardBackend → Qwen3), plus the
  continuous-batching prototype on the side. Boxes are annotated with numbers
  from the real traces; three boxes zoom into the detail views.
- **Roadmap** — PLAN.md as a metro line: 13 stops (phases 0–10) with real
  Status lines, a "you are here" marker, hardware tags (Mac → 5090 → cloud),
  per-phase detail cards, and the cross-cutting commitments. Static content
  authored from PLAN.md — update it when phase Status lines change.
- **Plumbing** — the round trip of one real request as an interactive horizontal
  horseshoe: request path along the top (messages → ChatML → `InferenceRequest`
  → `submit()` → `Sequence`), the generate yield-loop + model black box at the
  right, and the response path along the bottom (`TokenEvent` → bounded
  `asyncio.Queue` → consumer → decoder/phases → SSE). Thread regions, the
  backpressure story, and a per-stage detail card with the real traced payloads.
- **Model forward** — one real greedy generation through Qwen3-0.6B, scrubbable
  per forward pass: input tokens, mask shape, KV growth (with memory math),
  top-5 sampled candidates, and a block-anatomy diagram whose tensor shapes were
  captured by forward hooks (prefill vs decode toggle).
- **Speculative** — one real speculative run (Qwen3-0.6B drafting for
  Qwen3-1.7B): per-iteration accept/reject rhythm, draft chips vs main's
  fix/bonus token, KV rollbacks, draft-vs-main timing, and an honest
  scoreboard against a main-only baseline (which the speculative output must
  — and does — match token-for-token).
- **Token flow** — the same request as a lifecycle: messages → ChatML → prompt
  tokens → engine threads/queue → a replayable 387-token stream with
  thinking/text phases → the decoded output the client sees.
- **Tokenizer** — a live playground: type anything and see the real
  `Qwen3Tokenizer` output (token chips with ids, byte/char stats, optional
  ChatML wrapping), plus the full added/special-token table with the roles the
  codebase assigns (eos/pad stop set, think markers). Needs the tokenizer
  server running (below).
- **Weights** — where the parameters live, read from the safetensors headers
  of every downloaded checkpoint (no tensor data): param distribution,
  `weights.py`'s HF-name → module mapping with real shapes, weight-tying
  facts per checkpoint, and the KV-cache-vs-weights memory crossover.
- **Continuous batching** — the `prototypes/continuous_batching/` scheduler
  step-debugger: Gantt timeline, water-fill plan, padded batch tensor,
  sample/emit outcomes, KV slot pool, per-request output streams.
- **CB wiring** — the integration plan from `continuous_batching.md` +
  `fable-review.md` as a steppable diagram: what stays untouched, what gets
  built (in the doc's 5-step prereq order), and which prototype piece morphs
  into which real module — with the tricky points and review findings attached
  to the step they bite. Static design content, no trace needed; update it if
  the integration plan changes.

## Regenerating the traces

Trace data lives in `viz/data/*.js` and is **gitignored** — generate it once
before first use:

```
.venv/bin/python viz/trace_cb.py           # ~2s      → data/trace_cb.js
.venv/bin/python viz/trace_forward.py      # ~40s     → data/trace_forward.js + data/trace_tokenflow.js
.venv/bin/python viz/trace_weights.py      # instant  → data/trace_weights.js (safetensors headers only)
.venv/bin/python viz/trace_speculative.py  # ~2-3min  → data/trace_spec.js (loads 0.6B + 1.7B, runs spec + baseline)
```

The Tokenizer tab is live rather than trace-based — it needs its small server
(starts instantly; loads only `tokenizer.json`, no weights):

```
.venv/bin/python viz/tokenizer_server.py   # API on http://127.0.0.1:8765
```

It serves `POST /api/tokenize` + `GET /api/meta` with CORS open so the
`file://` page can call it, and also serves `viz/` itself at
`http://127.0.0.1:8765/` if you prefer opening it that way. The tab shows
start-me instructions when the server isn't running.

`trace_forward.py` loads the local Qwen3-0.6B weights (from
`src/cantollm/models/model_data/`, offline; ~30s of that time is the load) and
runs one short greedy chat completion on MPS/CPU. Greedy sampling + fixed seeds
make the traced tokens deterministic across runs; only wall-clock timings vary.
The page renders a "how to regenerate" note on any view whose data file is
missing.

## How the harnesses work

Both harnesses import the **real** code and instrument it at runtime — zero
changes to `src/` or `prototypes/`:

- `trace_forward.py` — torch `register_forward_hook` on every Qwen3 submodule
  (shapes only), plus instance/module attribute wraps around
  `StandardBackend.forward`/`.sample`, `EinsumAttentionMethod.build_mask`,
  `apply_rotary_emb` (patched in the `model.py` namespace where it's
  import-bound), `Qwen3Tokenizer.encode`, and `IncrementalDecoder.add`. The
  request is driven through the real `SequentialEngine.submit()` and
  `phase_tagged_events()` so the lifecycle trace reflects the actual
  worker-thread / asyncio-queue path.
- `trace_cb.py` — wraps the prototype scheduler's `_plan_step` /
  `_build_input_ids` / `greedy_sample` and snapshots queue/active state around
  each real `step()` call.

Traces are emitted as `window.TRACE_* = {...}` JS files (not JSON) so
`index.html` works under `file://`, where `fetch()` of local files is blocked.

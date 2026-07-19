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
  continuous-batching engine (`--engine batched`) on the side. Boxes are
  annotated with numbers from the real traces; three boxes zoom into the
  detail views.
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
  This is the sequential engine's pipe; the batched counterpart is the next tab.
- **Process split** — the Plumbing counterpart for `--engine batched`: the same
  round trip rebuilt around the process boundary, traced from a real run of
  `EngineProcessClient` + a spawned engine process serving Qwen3-0.6B. Two
  process boxes, two `mp.Queue` crossings, the bridge thread, and per-stage
  cards with the measured numbers: spawn→Ready timing, command-drain waits,
  per-step batch pickle sizes, IPC hop latency vs step duration, shutdown
  handshake, and the failure matrix (farewells + liveness both directions).
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
  sample/emit outcomes, KV slot pool, per-request output streams. The
  design it demonstrates now serves for real via `--engine batched`.
- **CB wiring** — the integration plan from `continuous-batching-plan.md`
  (the source of truth; supersedes the `old_research_continuous_batching.md`
  design note) as a steppable diagram: what stayed untouched, what got built
  (steps 0–9, all landed 2026-07-11, with owners), and which prototype
  piece morphed into which real module — with the decisions, tricky points,
  and review findings attached to the step they bit. Static design content,
  no trace needed; update it if the integration plan changes.
- **FlashAttention** — Phase-3 design content: what the einsum path's
  materialized score tensor costs (anchored to the 5090 longctx baseline
  numbers), a steppable tile-streaming animation of the online-softmax
  algorithm, an anatomy of the fused kernel (launch geometry, SRAM residency,
  the running-max rescale, the single output write), and how
  `F.scaled_dot_product_attention`'s backend dispatch + the planned
  `SDPABackend` fit the `AttentionMethod` attachment point. Static design
  content, no trace needed; revisit when the SDPA backend lands.
- **Flash walkthrough** — FlashAttention rebuilt bottom-up (companion to the
  FlashAttention tab, written from a full step-by-step walkthrough): the cast
  of tensors one thread block owns (shapes + provenance + the three-tier
  causality economy), a steppable five-number toy of the online softmax with
  ground-truth checksums, the block's whole life as a steppable HBM/SRAM/
  registers state board with byte counters (prologue → tiles → rescale →
  boundary tile → epilogue → ledger), the parallelism org chart
  (grid/block/warp/tensor core, the deliberately-serial K scan, Flash-Decoding
  split-KV merge), and what SDPA claims in cantoLLM's bench numbers. Static
  design content, no trace needed.

## Regenerating the traces

Trace data lives in `viz/data/*.js` and is **gitignored** — generate it once
before first use:

```
.venv/bin/python viz/trace_cb.py           # ~2s      → data/trace_cb.js
.venv/bin/python viz/trace_forward.py      # ~40s     → data/trace_forward.js + data/trace_tokenflow.js
.venv/bin/python viz/trace_weights.py      # instant  → data/trace_weights.js (safetensors headers only)
.venv/bin/python viz/trace_speculative.py  # ~2-3min  → data/trace_spec.js (loads 0.6B + 1.7B, runs spec + baseline)
.venv/bin/python viz/trace_split.py        # ~40s     → data/trace_split.js (spawns a real engine process on 0.6B)
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
- `trace_split.py` — runs the real process split and instruments both sides:
  the module-level scheduler factory (pickled by reference across spawn) wraps
  the real scheduler's `add_request`/`_plan_step`/`step` inside the engine
  process and dumps its half to JSON at exit; the parent wraps
  `EngineProcessClient._dispatch` and times each stream. Both sides stamp
  `time.time()` on one host, so IPC latencies subtract directly.

Traces are emitted as `window.TRACE_* = {...}` JS files (not JSON) so
`index.html` works under `file://`, where `fetch()` of local files is blocked.

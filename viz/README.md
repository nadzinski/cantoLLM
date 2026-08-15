# viz/ — interactive architecture explorer

A self-contained browser visualization of how cantoLLM works, for refreshing
your memory of the architecture. Open it directly (no server needed):

```
open viz/index.html
```

The views, linked by the nav bar (semantic zoom: the overview stays sparse,
detail lives behind the zoom targets):

- **Overview** — the request path through `src/cantollm/` (clients → FastAPI →
  registry → SequentialEngine → StandardBackend → Qwen3), plus the
  continuous-batching engine (`--engine batched`) on the side. Boxes are
  annotated with numbers from the real traces; three boxes zoom into the
  detail views.
- **Roadmap** — PLAN.md as a metro line: 15 stops (phases 0–12) with real
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
  `SDPAAttentionMethod` fit the `AttentionMethod` attachment point. Static design
  content, no trace needed; the SDPA method landed 2026-07-19 and, with the
  bounded shape vocabulary fixing cuDNN's per-shape plan compiles, is the CUDA
  serve default (long context 1.6–2.2× over einsum).
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
- **CUDA graphs**: unlike the other tabs, this one is a textbook mini-chapter
  in the format of `~/personal-projects/gr-learning` (serif prose, definition
  and rule boxes, "predict first" prompts that gate their reveals, framed
  interactives with captions, hands-on exercises, an end-of-chapter quiz);
  all styling is scoped to the view, everything else keeps the explorer look.
  Built entirely from a real L4 capture session (`viz/capture_cudagraphs.py`;
  committed record under `viz/captures/`). Arc: a motivating puzzle (five
  1-µs kernels that cost 31 µs, on a to-scale CPU/GPU timeline), the anatomy
  of one launch, the record-once-replay-as-one idea, the lifecycle on a
  steppable state board, real clickable graph dumps (the chain, then a tale
  of two fork-joins showing where edges come from), the size-sweep crossover,
  the rules derived from "replay consults no CPU" with recorded violations,
  a 110-kernel model at 15.4x, the engine's 9 ms floor re-read, and the
  cantoLLM boundary problem (KV-scatter wrinkle, buckets as capture
  vocabulary, the bills). Predictions and quiz answers persist in
  localStorage. Sections lean on each other; read in order.
- **Quantization**: a second textbook mini-chapter (same format; both chapters
  share one CSS/JS "chapter kit" inside index.html), written as read-ahead for
  Phases 7 and 11: general-purpose coverage of number formats and quantization
  methods, with this repo's real numbers as the recurring worked example. Arc:
  the checkpoint-size puzzle (params x 2 bytes; the two stakes are fitting
  30B-A3B on a 32 GB card and decode as a bytes-over-bandwidth problem), a
  format designer that derives bf16 and e4m3 from budget sliders, the number
  line under a microscope, where the codebase already refuses bf16, GPU memory
  101 with a two-ceilings calculator and an arrow-key decode-step byte
  odometer (with INT8-weight and FP8-KV replay stages), the grid-and-scale
  machinery, granularity measured on a real captured layer (layer-0 down_proj
  and its 4.4x outlier in row 35), the four-family outlier map, weight-only vs
  W8A8 vs KV-cache quantization, microscaling with a real block hand-quantized
  to NVFP4, GPTQ and AWQ toys, the deliberately-empty Phase 11 quality
  scoreboard, and a closing reread of the engine (spec.py's one dtype knob and
  how it has to split). Hardware figures were verified against vendor
  whitepapers at authoring time; specimen data is a committed capture (below).
  Predictions and quiz answers persist in localStorage.
- **H100 day** (`#/h100`): the Phase-3 close-out session's results
  (2026-08-14/15, p5.4xlarge in Tokyo), presented in the chapter kit's look
  (serif article, def/aside boxes, framed figures) but deliberately not a
  chapter: no predict gates, no quiz, no exercises. `h100-results.md` in the
  repo root is canonical; the tab carries the 0.6B cross-hardware anchor, the
  32B stack-vs-eager A/B, the decode-step-vs-bandwidth-floor figure, the
  open-loop knee, and the graded-predictions scoreboard. Charts are static
  SVG with hover tooltips, validated chapter palette, light mode.

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

Two harnesses are different, with artifacts **committed** under
`viz/captures/` instead of gitignored; their tabs embed curated inline
copies, and the committed artifacts are the record they were transcribed
from. `viz/capture_cudagraphs.py` needs CUDA hardware (run it on the
`infra/` GPU node; see `infra/README.md`) because a Mac can never
regenerate it; `--dry-run` builds the examples CPU-side for a quick import
check. `viz/capture_quant.py` is the opposite: pure safetensors reads of
the local 0.6B checkpoint, runs on the Mac in seconds, no GPU and no model
build (it feeds the Quantization tab's specimen-layer statistics and its
NVFP4 block figure).

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

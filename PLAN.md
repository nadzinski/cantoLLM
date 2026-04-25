This is a learning project.

The "production-like" framing is deliberate: real inference engines have to care about
IPC, observability, graceful shutdown, admission control, paged memory, multi-GPU
parallelism, and more — and you only build real intuition for those problems by putting
yourself in their shoes. But the goal is to understand these systems deeply enough to have
opinions about them, not to run CantoLLM in production. Educational detours over the
optimal path when they teach us something; performance is measured, not targeted.

The end state is a project that can be walked through end-to-end — from the PyTorch
attention kernel up through the paged KV allocator, the continuous-batching scheduler, and
the multi-GPU collective path — with every layer understood and defensible. That's why
multi-GPU (Phase 8) is non-optional even though the local setup doesn't need it.

## Headline features, in build order

The big milestones, stripped of plumbing and refactors. This is what CantoLLM will
actually _do_ at each stage.

1. **Decoupled API/engine architecture with OpenAI-compatible API** (Phase 1) — two
   processes, clean IPC, Anthropic + OpenAI wire formats, proper observability, graceful
   shutdown, admission control.
2. **Continuous batching with chunked prefill** (Phase 2) — API/engine process split lands
   alongside it; many concurrent requests share each forward pass; aggregate throughput
   steps up. Also: logprobs, stop strings.
3. **CUDA optimization with SDPA, `torch.compile`, and CUDA graphs** (Phase 3) — swap
   einsum for optimized kernels on the now-mature batched engine; first real perf-tuning
   pass, against a stable target that's worth optimizing.
4. **Paged KV cache** (Phase 4) — block-indexed KV pool with preemption; unlocks real
   concurrent capacity.
5. **Prefix caching with cache-aware routing** (Phase 5) — shared prefixes reused across
   requests; sticky routing keeps sessions on warm caches.
6. **Second model family: Gemma** (Phase 6) — sliding-window attention, SentencePiece,
   tied embeddings. Proves the runtime abstraction.
7. **Mixture of Experts: Qwen3-30B-A3B** (Phase 7) — router, experts, token dispatch.
   Single-GPU MoE in INT4.
8. **Multi-GPU tensor and expert parallelism** (Phase 8) — TP for dense models, EP for
   MoE. Real collectives on real cloud hardware.
9. **Quantization: INT8 weights and FP8/INT8 KV cache** (Phase 9) — decode bandwidth win
   on weights; capacity win on KV.
10. **Bonus track** (Phase 10) — guided decoding, multi-LoRA serving, disaggregated
    prefill/decode, multimodal. Pick based on interest.

Here's the full roadmap. Each phase is structured as: goal → refactors that have to land
first → feature work → hardware → what falls out.

---

## Phase 0 — Rename the project and set up measurement

**Goal:** small housekeeping before real work; costs a day, pays dividends forever.

**Status (2026-04-19):** Rename done (`src/cantollm/` with Qwen3 code under
`models/qwen3/`). Open: benchmark harness spec document.

- Rename `src/qwen3/` → `src/cantollm/`. `qwen3` becomes a backend subfolder for
  Qwen3-specific code (model, tokenizer, weight loader). Generic code (engine, api,
  kv_cache, generator, decoder) moves up.
- Define the benchmark harness **spec** _now_ (the deliverable is the spec document, not a
  full implementation): prompt sets (short chat, long context, code, multi-turn),
  concurrency ramps, metrics (TTFT p50/p90/p99, ITL, aggregate tok/s, KV utilization,
  finish-reason distribution), reporting format. Extends existing `bench.py`. Some of
  those metrics (ITL, finish reasons, KV utilization) require the `TokenEvent` widening in
  Phase 1a and the scheduler in Phase 2 before they're actually instrumentable — the spec
  names them; the instrumentation lands across phases as the contract grows.

**Hardware:** Mac.

---

## Phase 1a — Refactor the engine seams

**Goal:** land the architectural hygiene that continuous batching will need. Stay
single-process — but shape every interface so the Phase 2 process split is a later
afternoon's work, not a rewrite.

**Status (2026-04-19):** Complete. `InferenceBackend` Protocol + `StandardBackend` /
`SpeculativeBackend` split landed; `api_types.py` now at `api/anthropic_types.py`;
`TokenEvent` widened (`finish_reason`, `error`, `request_id`) and the adapter's
count-based `stop_reason` guess is gone; HTTP/SSE contract test suite green;
`ModelSpec` / `ModelRuntime` / `EngineRegistry` landed — `SequentialEngine`
takes a runtime, cache is allocated via `runtime.new_cache()`, `create_app`
takes the registry and dispatches per-model tokenizer via `body.model`;
`Message.content` flattening removed — structured `list[ContentBlockInput]`
now passes through to the tokenizer; API-side `encode_conversation` runs on a
dedicated `ThreadPoolExecutor` with tokenizer errors mapped to HTTP 400; both
latent bugs fixed — causal mask is built per-forward in `Qwen3LLM.forward`
(idle memory drops by ~1.68 GB at `max_seq_len=40960`), and the tokenizer's
single-special-token shortcut is gated on `not chat_wrapped`.

**Refactors:**

- `ModelRuntime` abstraction owning weights, tokenizer, device, dtype, cache pool, and an
  `InferenceBackend` (the `StandardBackend` / `SpeculativeBackend` Protocol already
  landed pre-phase — see `src/cantollm/engine/backend.py`). `SequentialEngine(runtime)`
  replaces `SequentialEngine(backend, config)`; the runtime composes the backend rather
  than replacing it. Today `SequentialEngine.submit` instantiates a fresh
  `KVCache(config["num_transformers"])` per request — fine for sequential, a non-starter
  for batching; moving cache ownership into `ModelRuntime.new_cache()` is the headline
  change in this bullet.
- `ModelSpec` extracted from the inline `MODEL_CONFIGS` dict in `main.py:32`. A spec is a
  single declarative record per model (arch params, dtype, weight loader, tokenizer class,
  chat-template flavor) that the registry consumes. CLI stays, but it constructs runtimes
  from specs rather than owning the knobs.
- `EngineRegistry` so requests can pick a backend by model name. `create_app(registry)`
  replaces `create_app(engine, tokenizer, model_name)`. The registry arbitrates which
  engine owns which model's weights and cache pool — relevant once batching engines want
  exclusive ownership.
- Widen `TokenEvent` with `finish_reason`, `error`, `request_id`, and plan for optional
  `logprobs` / timing fields. Today the adapter _guesses_ `stop_reason` from
  `counts.total >= max_tokens`; the engine knows whether it actually stopped from EOS, a
  stop token, `max_tokens`, or an abort, and that information should survive the boundary.
  `request_id` is harmless today and load-bearing once a scheduler multiplexes per-request
  queues.
- Stop flattening `Message.content` in `api_types.py::Message._flatten_content` — pass
  structured `list[ContentBlockInput]` blocks through. The API will still tokenize (see
  below), but it needs the structured form to do so correctly, and the engine will need it
  eventually for multimodal.
- Rename `api_types.py` → `api/anthropic_types.py` and split adapter. It's Anthropic-only
  today (the `sse(evt)` helper included — OpenAI SSE uses `data: [DONE]` with no `event:`
  lines), and Phase 1b adds an OpenAI surface that wants its own types file.
- **Tokenization stays API-side, made per-model-aware via the registry.** The API process
  runs `encode_conversation` on a thread pool (HF tokenizers releases the GIL) and hands
  the engine an `InferenceRequest` carrying `prompt_token_ids`. Rationale:
  - **Fail fast on validation.** Malformed prompts, length overflows, bad templates → 400
    before engine IPC is engaged.
  - **Protect the scheduler.** Once Phase 2's batching loop exists, it wants to step every
    ~50ms. An 8k-prompt tokenization can take tens of ms; keeping that out of the engine
    process means it doesn't compete with the scheduler for CPU.
  - **Engine stays ignorant of HF tokenizers, ChatML, SentencePiece, special tokens.**
    Engine API is just "here are IDs, generate." Cleaner boundary.
  - **Prefix caching still works** — the engine gets token IDs regardless of who tokenized
    them.
  - **vLLM precedent.** This is what a real system does.

  What changes is the _dispatch_: today `create_app(tokenizer, ...)` hardcodes one
  tokenizer instance; post-refactor the API looks up the tokenizer via
  `registry.get(model_name)`. The API stays a thin protocol adapter — it doesn't own
  chat-template _semantics_, it just calls the per-model tokenizer with structured inputs.

**Latent bug fixes:**

- `model.py:262`'s `self.mask = torch.ones(max_seq_len, max_seq_len).tril() == 0` buffer
  is ~1.68 GB at `max_seq_len=40_960`. Build causal masks on demand in the attention layer
  (or defer to SDPA's implicit causal once Phase 3 lands). Blocks long-context testing
  today and makes continuous batching's memory pressure intolerable if left alone.
- `tokenizer.py:143` quick-path bug: `encode("<|im_start|>", chat_wrapped=True)` returns
  the raw token ID instead of wrapping, because the special-token shortcut runs before the
  `chat_wrapped` check. Gate the shortcut on `not chat_wrapped`. Minor, but prompt-prep is
  about to become a first-class subsystem and it shouldn't carry known footguns into it.

**HTTP/SSE contract tests:** Before ripping up the adapter and the engine boundary, lock
down behavior that's currently only verified by hand:

- `/v1/messages` non-streaming happy path (Pydantic validation, content-block shape, usage
  accounting).
- SSE event sequence: `message_start` → block start/delta/stop pairs → `message_delta`
  (with correct `stop_reason`) → `message_stop`.
- Thinking-block framing: `<think>` / `</think>` tokens produce the correct
  `content_block_start` / `content_block_stop` pairs with `thinking_delta`s in between.
- Mid-stream abort (client disconnect): engine's `stop_event` fires, worker thread exits,
  no "Task was destroyed" warnings.
- Error propagation: exception during generation surfaces as an SSE `error` event (once
  `TokenEvent.error` lands), not a silent truncation.
- Ping cadence during long idle gaps.

**Process-split readiness (not the split itself):** The `InferenceEngine` Protocol already
exposes `start` / `shutdown` / `submit` / `abort`. Keep it that way. Every new piece of
state the engine grows in this phase (registry, runtime, KV pool) must be cleanly ownable
by a separate process — no shared Python objects across the `InferenceEngine` boundary, no
API-side reach-through into engine internals. When the split lands in Phase 2, it should
only touch the engine side.

**Hardware:** Mac.

**Exit criteria:** `InferenceRequest` carries `prompt_token_ids` + `request_id`; API
tokenization runs on a thread pool and dispatches per-model through the registry (no
hardcoded tokenizer in `create_app`); `TokenEvent` surfaces `finish_reason` / `error`;
adapter contains zero inference of engine state; `ModelRuntime` owns the KV cache
(per-request `KVCache()` in `SequentialEngine.submit` is gone); both latent bugs fixed and
verified; HTTP/SSE contract test suite is green.

---

## Phase 1b — Second API surface + the loose ends

**Goal:** exercise the registry by adding OpenAI, and pick up the small reliability wins
that fall out of the Phase 1a refactors.

**Status (2026-04-19):** OpenAI Chat Completions surface landed
(`/v1/chat/completions`) alongside a refactor of `src/cantollm/api/` into
per-dialect routers (`anthropic_router`, `openai_router`, `common_router`
for `/health` + a union-payload `/v1/models`) plus shared helpers
(`phase.py` for the thinking/text classifier, `common.py` for request
tokenization). Thinking tokens are emitted as DeepSeek-R1-style
`reasoning_content` on assistant messages / deltas. CLI gained
`--api {anthropic,openai}` (default anthropic). Open: stream error-event
parity (OpenAI mid-stream error is minimum-viable today), `X-Request-ID`
middleware, raw-tokens NDJSON endpoint.

- **OpenAI API surface** (`/v1/chat/completions`) as a second adapter.
  `api/openai_types.py` + `api/openai_adapter.py`. Forces the registry pattern to stay
  honest and gives us a real multi-surface story before batching lands.
- **Stream error events**: Anthropic SSE `error` event (and OpenAI equivalent) on
  mid-stream exceptions — straightforward now that `TokenEvent` carries `error`. No more
  silent truncation.
- **Request ID propagation**: `X-Request-ID` middleware → `InferenceRequest.request_id` →
  structured JSON logs tagged with it.
- **Raw-tokens NDJSON endpoint** as a third adapter. A forcing function for keeping
  `TokenEvent` honest, and useful for debugging.

**Hardware:** Mac.

_(The remaining "production hygiene" bundle — /metrics, admission control, /ready,
graceful shutdown, load testing — is deferred to Phase 3.5. Most of those metrics are
uninteresting until there's a scheduler and a real engine process to measure.)_

---

## Phase 2 — Continuous batching, naive version + process split

**Goal:** the headline feature. Many concurrent requests share each forward pass. Padded
KV, no paging yet. This is where the big learning happens — and where the API/engine
process split finally makes sense, because the engine is now a steady-state loop that
wants to own its own process. Perf optimizations (SDPA, `torch.compile`, CUDA graphs) are
deliberately deferred to Phase 3: this phase is about getting the scheduler and batching
right on a correctness-first attention path, then optimizing the mature target.

**Status (2026-04-25):** All four prereq refactors landed. The
pluggable attention-compute boundary is in (named `AttentionMethod` in code,
not `AttentionBackend`): `GroupedQueryAttention` delegates score +
value-aggregate + KV update + mask construction to a method, with
`EinsumAttentionMethod` as the correctness reference and
`PaddedAttentionMethod` stubbed (NotImplementedError) marking the
continuous-batching slot. Logits-processor pipeline landed: `SamplingParams`
carries `list[LogitsProcessor]` + greedy flag (built via
`from_temperature_top_p`); both `StandardBackend` and `SpeculativeBackend`
read from it, so future sampling knobs (repetition penalty, logit bias,
guided decoding) extend the pipeline instead of patching the hot path.
Per-sequence state object landed: `Sequence` dataclass in `engine/types.py`
bundles request_id, prompt_token_ids, sampling_params, stop_token_ids,
max_tokens, cache, stop_event, and tokens_emitted; `InferenceBackend.generate`
collapsed from a 6-arg call to `generate(sequence)`; `SequentialEngine.run()`
constructs the sequence, ticks `tokens_emitted` in the consumer loop, and
reads finish reason via `seq.finish_reason_after_normal_exit()`. Tiny
test-model fixture landed in `tests/tiny_model.py`: a `tiny_qwen3_spec()`
builder (2-layer, 64-dim, vocab 2048, random-init weights, `FakeTokenizer`)
plugs straight into `build_runtime` so scheduler/batching tests can exercise
the full `SequentialEngine → StandardBackend → Qwen3 → KVCache` path in
under a second; `tests/test_tiny_model.py` smoke-tests the seam.
Deliberately deferred: explicit `position` decoupled from `cache.position`
(only needed once batching has divergent positions), per-sequence draft
cache (speculative is batch=1 today). Open: all feature work (process
split, scheduler, batched forward, padded KV, chunked prefill, logprobs,
stop strings).

**Refactors that have to land first:**

- **`AttentionBackend` as a named artifact.** `GroupedQueryAttention` loses its inline
  einsum and delegates to a backend. Prefill and decode have different compute shapes
  (matmul vs. outer-product); backends must handle them separately. Interface roughly:
  `forward_prefill`, `forward_decode`, `build_mask`, plus backend-specific KV-access
  methods. Initial backends:
  - `EinsumBackend` — the current implementation, factored out. Becomes the correctness
    reference going forward; stays live as a fallback, especially on Mac.
  - `PaddedBackend` — continuous batching, this phase. Starts as einsum extended with
    per-sequence masking; becomes the slot that Phase 3's `SDPABackend` swaps into.
  - `PagedBackend` — slots in during Phase 4 with no other changes to
    `GroupedQueryAttention`.
- **Logits processors as a pipeline** — per-sequence `list[LogitsProcessor]` replaces
  hardcoded temperature/top-p in `get_probs`. Each request gets its own.
- **Per-sequence state object** — replaces the implicit single-request state in
  `SequentialEngine`.
- **Tiny test-model fixture** (2-layer, 64-dim) so scheduler/batching tests run in
  milliseconds, no HF downloads.

**Feature work:**

- **Process split**: API process (FastAPI, async) and engine process (sync busy loop). IPC
  over ZMQ or multiprocessing queues — pick one and commit. Engine drops asyncio
  internally; a bridge thread translates at the boundary. Events batch per engine step,
  not per token (per-token IPC would eat every perf win). Tokenization stays API-side as
  established in Phase 1a — this is exactly the phase where that decision pays off,
  because the scheduler loop in the engine process now has something real to protect. Fix
  the `put_nowait` backpressure edge case at the new IPC boundary while we're rebuilding
  it. Switch uvicorn to `uvloop` + `httptools` as part of the split — the API is now
  serving many concurrent streams and wants the faster loop before this phase's
  end-of-phase benchmark locks in the baseline that Phase 3 will measure against.
- **Scheduler**: FCFS waiting queue, running set, per-step token budget. Decode requests
  prioritized; prefill requests pulled in when budget allows.
- **Batched forward pass**: `model.py`'s `batches` dim is already plumbed through —
  finally exercise it. Verify attention mask + RoPE offsets handle variable sequence
  lengths.
- **Padded KV cache** sized per active slot. Preallocate
  `max_batch_size × max_seq_len × ...` tensors.
- **Per-sequence stop logic**, per-sequence sampling, per-sequence event streams over IPC.
- **Chunked prefill** — falls out once token budget exists. Long prompts don't monopolize
  a step.
- **Logprobs in response** — trivial addition once the sampler is per-sequence.
- **Stop strings** (not just stop token IDs) — requires decoder-level backtracking. Minor
  but nice.

**Speculative decoding in batched mode:** disabled. Divergent accept counts across
sequences turn the batched forward pass into a ragged horror that isn't worth debugging
right now. Keep a separate unbatched engine mode where speculation is enabled — useful as
a single-request fast path and as a sandbox for revisiting batched speculation later as
its own research detour.

**Minimal CUDA bring-up** (not the optimization pass — that's Phase 3): enough MPS-ism
cleanup (env vars, dtype dispatching) to run the new batched engine on the 5090 for
functional validation. Einsum attention runs fine on CUDA; it's just not fast.

**Hardware:** 5090 for functional and scale validation; Mac for CPU-only scheduler
correctness tests and day-to-day development.

**Bench:** first big win on the dashboard. Aggregate tok/s under concurrency should step
up dramatically. Individual request TTFT may get slightly worse under heavy load — that's
the tradeoff, make it visible. End-of-phase numbers become the "before" baseline that
Phase 3's CUDA optimizations measure against.

**Adjacent:** one H100 day post-phase (~$30) for a proper comparison vs. vLLM on the same
hardware.

---

## Phase 3 — CUDA optimization beachhead

**Goal:** now that the batched engine is mature and stable, do the first real perf-tuning
pass. Swap einsum for SDPA, measure `torch.compile`, capture CUDA graphs where they help.
The target is worth optimizing because its shape is stable — no major refactor is about to
invalidate the work (paged KV in Phase 4 _will_ reshape the attention path, but in a
well-contained way).

**Core, load-bearing:**

- Add a new `SDPABackend` that delegates to `F.scaled_dot_product_attention` and make it
  the default on CUDA. EinsumBackend stays as correctness reference and as the Mac/CPU
  fallback. SDPA handles both the single-request and batched padded paths cleanly — it
  slots straight into the `AttentionBackend` interface landed in Phase 2, without
  disturbing the scheduler or KV layout.
- Re-run the bench harness against the batched+SDPA engine. Compare to Phase 2's
  end-of-phase numbers. This is the first "before vs. after" comparison that actually
  measures an optimization rather than a feature.

**Exploratory, expected to tune partially and revisit:**

- Enable `torch.compile` on the batched forward pass. Dynamic shapes from variable prompt
  lengths and varying batch sizes will fight you; budget for a day of
  `torch._dynamo.explain()`. Pragmatic compromise: compile per batch-size bucket, or
  accept recompilation overhead on shape changes. Paged KV (Phase 4) will invalidate some
  of this work by making KV access non-contiguous — expect to revisit. Do it here anyway;
  experiencing the tension first-hand is the point.
- CUDA graphs for steady-state decode. The honest story here is the classic
  continuous-batching tension: graphs want fixed shapes, schedulers produce dynamic ones.
  Pragmatic approach: capture graphs for a small set of batch-size buckets (e.g. 1, 4,
  16, 32) and dispatch to the matching graph at runtime, with an eager fallback. Measure
  the ITL win per bucket and the fraction of steps that hit each. This is exactly how vLLM
  does it — the implementation detail _is_ the lesson.

**Hardware:** 5090 primary.

**Adjacent:** one H100 day (cloud, ~$30) to see how the same optimizations scale on bigger
iron — especially `torch.compile`, which has more headroom on Hopper.

---

## Phase 3.5 — Production hygiene

**Goal:** now that the engine is a separate process and metrics actually show useful
things (queue depth, batch size, preemption count, KV utilization), pick up the production
basics that were deferred from Phase 1.

- **Observability**: `/metrics` Prometheus endpoint. Engine metrics (queue depth, active
  requests, tok/s, TTFT, ITL, KV utilization, preemption count, batch size). HTTP metrics
  (latency histograms, status codes). Process metrics (CPU, mem, GPU util via nvidia-smi).
  **Event loop lag** as a core API-side metric. Local Grafana via docker-compose.
- **Admission control**: bounded in-flight-requests semaphore at the API layer. Over
  limit: queue with timeout, then 429 with `Retry-After`.
- **Readiness vs liveness**: `/health` (liveness), `/ready` (readiness — model loaded,
  warm-up done, engine subprocess healthy, not shutting down). Warm-up does a dummy
  forward pass before flipping `/ready`.
- **Graceful shutdown**: SIGTERM → `/ready` flips to 503 → drain in-flight streams with
  30s deadline → abort the rest. Engine shutdown waits on the scheduler loop rather than
  being the stub it is today.
- **Model reload**: build a new engine for the target model, atomically swap it into the
  registry, drain the old one. Falls out cleanly once the registry exists; wire it to a
  signal or admin endpoint.
- **Non-goal**: auth and rate limiting stay out of process. Those belong in a sidecar
  (nginx, envoy) and building them in-process is a distraction from the
  learning-vs-production split this project is about.
- **Load test exit criterion**: 50 concurrent clients with mixed prompt sizes run to
  completion with no stream truncation. Kill -9 the engine, API serves 503; restart
  engine, API recovers.

**Hardware:** 5090 primary.

---

## Phase 4 — Paged KV cache

**Goal:** replace padded contiguous KV with a block-indexed pool + block tables. The
memory management lesson.

- KV blocks of fixed size (16 tokens is the vLLM default) in a single preallocated pool.
- Per-request block table mapping logical token positions → block IDs.
- `PagedBackend` learns to gather K/V from non-contiguous blocks (slots into the Phase 3
  `AttentionBackend` interface).
- Block allocator with free-list, refcounts, eviction policy.
- Preemption support: when out of blocks, evict a low-priority request's cache
  (recompute-on-resume first; swap-to-CPU is a fancier alternative).

**Hardware:** 5090 primary. Cloud H100 at the end for a scale check.

**Bench:** KV utilization goes up materially. Maximum concurrent requests at a given
prompt-length distribution climbs.

---

## Phase 5 — Prefix caching + cache-aware routing

**Goal:** reuse prefixes across requests. Natural stack on top of paging.

- Hash block contents (include position/prev-block-hash in the hash to avoid collisions).
- Lookup table: `block_hash → block_id`.
- On new request: walk the prompt in block-sized chunks, look up the longest cached
  prefix, skip prefill for those blocks.
- Refcount cached blocks so they're not evicted while in use.
- `/metrics`: cache hit rate, tokens saved.

**Cache-aware routing experiment:** stand up 2 API instances in front of 2 engine
processes, both cohabiting on the 5090 (consumer Ada has no MIG — each engine just takes a
PyTorch context and is sized to roughly half VRAM; imperfect, but it's the routing layer
that's the point, not the isolation). Consistent hashing on session ID at the API layer so
the same conversation lands on the same engine. Measure cache hit rate with vs. without
sticky routing. This teaches the distributed-systems half of §5.3.3.

**Hardware:** 5090 primary.

---

## Phase 6 — Second model: Gemma 4

**Goal:** prove the abstractions. Different architecture, different tokenizer, different
weight layout.

**Refactors that have to land first:**

- **Per-layer config** — layers aren't identical anymore; sliding window alternates with
  global attention.
- **Weight loader abstraction** — factor the tensor-name mapping into a per-model object.
- **Tokenizer interface** — SentencePiece, not BPE. `Qwen3Tokenizer` becomes one
  implementation.
- **Tied embeddings** support.

**Feature work:**

- Gemma 4 model class (verify the architecture specifics — sliding window pattern, RoPE
  config, tying — against the actual release).
- Sliding window attention as a new `AttentionBackend` variant.
- Second tokenizer implementation.

**Hardware:** 5090 (Gemma 4B / 12B comfortably fit).

---

## Phase 7 — MoE: Qwen3-30B-A3B

**Goal:** MoE architecture, single GPU first. Teeing up parallelism.

**Prereq — INT8 weight-only quantization:** Quant infrastructure has to land before
Qwen3-30B-A3B fits on the 5090. Build INT8 weight-only (per-channel scale factors) on the
dense Qwen3 models first — debug the quant path and benchmark it against bf16 on a model
whose quality you have intuition for. Then extend to INT4 for A3B. (Further quant work —
KV-cache quant, AWQ, FP8 — lives in Phase 9.)

**Feature work:**

- Router / top-k gating layer.
- Expert layers (many small FFs replacing the single FF block).
- Token dispatch: group tokens by assigned expert, run each expert's batch, scatter
  results back.
- Load imbalance metrics (tokens per expert per step).
- INT4 weight quantization (extension of the INT8 path) to fit on 5090 (~15GB).

**Hardware:** 5090 (INT4). Cloud H100 for bf16 runs as a sanity check.

---

## Phase 8 — Multi-GPU parallelism (cloud-primary phase)

**Goal:** the big cloud session. ~$60–100 budget, one focused weekend. Non-optional for
the end state — understanding collective comms and sharding is half of what a modern
inference engine is doing.

**Local dev (Mac / 5090):** implement TP/EP using PyTorch distributed with `gloo` backend
on CPU. Multiple processes, all-reduce / broadcast / rank routing — validates the _shape_
of the code. Zero perf signal; pure correctness. Expect real debug work in the cloud on
top of this, not a clean "just run it."

**Cloud (4-8×H100):**

- **Tensor parallelism** on Qwen3 dense models and Gemma 4. Split weights across GPUs,
  all-reduce at each layer.
- **Expert parallelism** on Qwen3-30B-A3B. Shard experts across GPUs.
- Benchmark each against single-GPU baseline.

**Hardware:** Mac/5090 for dev, cloud for runs. Probably 2–3 cloud sessions of a few hours
each.

---

## Phase 9 — Quantization, deeper

**Goal:** INT8 weight-only landed in Phase 7 as a prereq; this is where quant gets serious
— KV-cache quant, FP8, and AWQ.

- FP8/INT8 KV cache with per-block scales. Doubles effective KV capacity.
- Interaction with paged attention (scale factors live per block).
- Quality measurement (needle-in-haystack, basic reasoning evals) across the full quant
  matrix: bf16, INT8 weight-only, INT4, INT8-weight + INT8-KV, FP8-KV on Hopper.
- AWQ — activation-aware weight quantization, a conceptual step up from uniform
  per-channel scaling.

**Hardware:** 5090. FP8 requires Hopper+ for native — 5090 can do it in software but it's
a cloud H100 session for real numbers.

---

## Phase 10 — Bonus track

Pick based on interest. All are self-contained after Phase 5.

- **Guided decoding (FSM)** via xgrammar or outlines. Logit masking at each step.
  Interesting CS, practical for agentic use.
- **Multi-LoRA serving**. Swap adapter weights per request. Different parallelism flavor
  than paged KV.
- **Disaggregated P/D**. Prefill engine and decode engine as separate processes, KV
  transfer between them. Can do meaningfully even single-node.
- **Vision / multimodal** (Gemma 4 or Qwen-VL). Biggest undertaking; largely new ground.
- **Batched speculative decoding**, revisiting the Phase 2 punt. Ragged accept counts
  across sequences is a genuinely interesting research problem.

---

## Hardware cadence summary

| Phase                           | Mac        | 5090    | Cloud                 |
| ------------------------------- | ---------- | ------- | --------------------- |
| 0 — rename + bench spec         | primary    | —       | —                     |
| 1a — engine-seam refactors      | primary    | —       | —                     |
| 1b — OpenAI + loose ends        | primary    | —       | —                     |
| 2 — continuous batching + split | dev        | primary | 1 H100 day ($30)      |
| 3 — CUDA beachhead              | dev        | primary | 1 H100 day ($30)      |
| 3.5 — production hygiene        | dev        | primary | —                     |
| 4 — paged KV                    | dev        | primary | optional              |
| 5 — prefix + routing            | dev        | primary | —                     |
| 6 — Gemma 4                     | dev        | primary | —                     |
| 7 — MoE (+ INT8 prereq)         | dev        | primary | occasional            |
| 8 — multi-GPU                   | dev (gloo) | dev     | **primary** ($60–100) |
| 9 — quant deeper                | dev        | primary | 1 H100 day for FP8    |
| 10 — bonus                      | varies     | varies  | varies                |

Total cloud budget: ~$150, front-loaded on Phase 8.

---

## A few cross-cutting commitments

- **Every phase ends with a bench run on the 5090** and a commit to the dashboard history.
  No unmeasured optimization.
- **Reference-parity tests** against HF `transformers` for every new backend (padded,
  paged, quant, TP). `tests/compare_implementations.py` is the seed — extend it, don't let
  it bit-rot.
- **vLLM side-by-side** at every major milestone where it's meaningful (Phase 3 onward).
  One H100 day is enough for a useful comparison.
- **Every phase has its own short design note** (in-repo, not in conversation) — one page,
  decisions made, alternatives considered. Cheap discipline, huge future-you value. Doubly
  cheap given the walk-through-able end state is the point.
- **Nsight Systems capture at each major milestone.** 10 minutes, produces the "oh
  _that's_ where the time goes" moment.
- **Refactors are bundled with the phase that needs them**, not done upfront en masse.
  "Just-in-time refactoring" — easier to motivate, less speculative.

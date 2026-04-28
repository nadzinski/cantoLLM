# Continuous Batching — Coding Challenge

## The brief

You are given a toy single-layer transformer (`toy_model.py`) that runs one
batched forward pass per call: it takes a stack of input tokens, writes
fresh K/V into a shared cache at the slot/position you tell it, and returns
logits at the last real token of each row.

You're going to build the runtime that drives it: a **continuous-batching
scheduler** that takes in many concurrent generation requests, packs them
together into each forward pass, and streams tokens back per request.

Your job is to fill in two files:

- `padded_kv.py` — the KV cache pool the scheduler hands to the model.
- `scheduler.py` — the loop that decides what runs each step and processes
  the results.

Everything else (the model, the tokenizer-free `Request`/`Sequence` types,
a sequential reference engine that serves as the correctness oracle, a
test runner, and the test suite) is already written. The tests pin your
output token-for-token against the reference engine — if your scheduler
diverges by one token, you'll see it.

## Setup

Dependencies are already installed in the project's `.venv` (PyTorch,
pytest). From this directory:

```
cd prototypes/continuous_batching
pytest -v
```

You should see 14 failing tests, all `NotImplementedError`. Your goal is
to get all 14 green.

Suggested order:

1. Fill in `padded_kv.py`. The 5 tests in `test_padded_kv.py` go green.
2. The 3 tests in `test_reference.py` also go green (the reference engine
   uses your cache too).
3. Fill in `scheduler.py`. The 6 tests in `test_scheduler.py` go green.

## What's provided

| File | What it does |
|---|---|
| `cb_types.py` | `Request`, `Sequence`, `TokenEvent`, `FinishReason` dataclasses. |
| `toy_model.py` | `ToyModel` — single-layer attention, single head, no normalization. Real PyTorch, fixed random init. |
| `sampler.py` | `greedy_sample(logits)` — argmax. |
| `reference.py` | `SequentialReference.generate(request) -> list[int]`. The oracle — runs one request at a time. |
| `runner.py` | `run_to_completion(scheduler, requests)` and `run_with_late_arrivals(scheduler, schedule)`. |
| `tests/` | The full test suite + a `conftest.py` with the fixtures (`model`, `reference`, `make_request`, `cache_factory`). |

Read `toy_model.py` carefully — its forward signature defines the contract
your scheduler has to call. In particular, look at how `slot_metas` and
`kv_cache` are used; that tells you exactly what your scheduler needs to
compute and pass in.

## What you implement

### `PaddedKVCache` (in `padded_kv.py`)

A pool of preallocated K and V tensors plus a slot allocator.

**Constructor:** `PaddedKVCache(max_batch, max_seq_len, dim, device="cpu")`

**Required public attributes after construction:**

- `k_cache`: `torch.Tensor` of shape `(max_batch, max_seq_len, dim)`,
  zero-initialized. Public — the model writes/reads it directly via
  slicing.
- `v_cache`: same shape, zero-initialized.
- `max_batch`, `max_seq_len`, `dim`: as passed.

**Methods:**

- `allocate_slot() -> int | None` — return a free slot index in
  `[0, max_batch)`, or `None` if all slots are in use.
- `free_slot(slot_idx: int) -> None` — mark the slot reusable. You don't
  need to zero its contents — whoever next allocates the slot will start
  writing at position 0.
- `num_free() -> int`
- `num_active() -> int`

The scheduler is allowed to free and reallocate any slot any number of
times. Two simultaneous `allocate_slot()` calls must return distinct
indices.

### `ContinuousBatchingScheduler` (in `scheduler.py`)

The loop that drives the model.

**Constructor:**
`ContinuousBatchingScheduler(model: ToyModel, cache: PaddedKVCache, max_tokens_per_step: int)`

`max_tokens_per_step` is the per-step **token budget** across the whole
batch. Each token of model input — whether it's part of a prefill chunk
or a single decode step — counts against this budget. Use it.

**Methods:**

- `add_request(request: Request) -> None` — admit a request. Don't run
  the model here; that happens in `step()`.
- `step() -> list[TokenEvent]` — do one batched forward pass and return
  every event it produced (token outputs and finish events).
- `is_idle() -> bool` — `True` iff no requests are waiting and none are
  in flight.

**Per-request output contract.** A request submitted with `max_tokens=N`
should produce up to `N` `TokenEvent`s with `token_id` set, followed by
exactly one `TokenEvent` with `finish_reason` set:

- `"end_turn"` if a sampled token is in `request.stop_token_ids`. The
  stop token itself **is** emitted, then the request finishes. (Same
  shape as the reference — see `reference.py`.)
- `"max_tokens"` if `max_tokens` outputs were emitted without hitting a
  stop token.

**Concurrency requirements.** All these have to work:

- More requests than `cache.max_batch` slots — the extras wait.
- Requests with prompts longer than `max_tokens_per_step` — each prompt
  is processed across multiple steps.
- Requests added between `step()` calls — they get admitted in time for
  the next step they fit in.
- Requests with different `max_tokens` and different prompt lengths
  running concurrently — they each finish independently.

**Determinism.** With `greedy_sample` and a seeded model, the output
token list for a given request must be **identical** to what
`SequentialReference.generate(request)` produces, regardless of what
other requests are running alongside it. That's what the bulk of the
test suite checks.

## What this is testing

Each scheduler test fails on a specific class of bug:

| Test | Likely failure mode |
|---|---|
| `test_single_request_matches_reference` | prefill→decode boundary or off-by-one in position tracking |
| `test_three_concurrent_requests_match_reference` | per-row independence broken (slot indexing, mask, K/V write addresses) |
| `test_chunked_prefill_matches_reference` | token budget dropping or duplicating tokens during chunked prefill |
| `test_more_requests_than_slots` | promotion from waiting → running broken |
| `test_stop_token_emitted_at_correct_position` | stop-check vs. token-emission ordering wrong |
| `test_late_arriving_request` | mid-stream admission corrupts running sequences |

## Where to look in the parent project

This prototype mirrors the shape of cantoLLM's real engine:

- `src/cantollm/engine/types.py` — the real `Sequence` / `InferenceRequest` / `TokenEvent`.
- `src/cantollm/engine/sequential.py` — the real one-at-a-time engine.
- `PLAN.md` Phase 2 — the feature this prototype is rehearsing.

Don't read `src/cantollm/models/attention/padded.py` in the parent
project — it's a `NotImplementedError` stub that's the real version of
the work you're about to do, and the point is to figure it out yourself.

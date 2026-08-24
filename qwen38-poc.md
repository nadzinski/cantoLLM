# Qwen 3.8-27B PoC: hybrid GDN + gated attention in the existing engine

**Status: complete and parked (2026-08-23).** The PoC answered its
question: a hybrid linear-attention model serves through the unchanged
engine (registry, API routers, continuous batching, chunked prefill)
with model code fully separate from qwen3. Branch `qwen38-27b-poc`;
plan of record in the session plan file; validated on the 5090
2026-08-20. Not merged, no PLAN.md/viz changes, by decision.

## What Qwen 3.8 is (verified against HF, config.json, checkpoint index)

`model_type: qwen3_5`, the same architecture as Qwen 3.5/3.6. 64 layers
in a repeating [linear, linear, linear, full] pattern: 48 Gated
DeltaNet layers (16 QK / 48 V heads at dim 128, causal conv k=4, fp32
recurrent state) and 16 gated full-attention layers (GQA 24Q/4KV,
head_dim 256, per-head output gate: q_proj is double width, the second
half multiplies the attention output as sigmoid). Dense SwiGLU
(17408) every layer. Details that would have been silent wrong guesses:

- **Zero-centered RMSNorms.** All layer norms and the q/k norms store
  weights as offsets from 1; forward scales by `(1 + w)`. Loading them
  into a plain RMSNorm shifts every scale by 1.
- **Two different gates.** Attention output gate is sigmoid; the GDN
  output norm's gate is silu.
- **Partial RoPE, 64 of 256 dims, theta 1e7**, half-split layout, and
  the interleaved mRoPE (sections [11,11,10]) collapses exactly to 1-D
  RoPE for text-only, verified numerically against HF.
- Vocab 248,320; ChatML unchanged (im_start 248045, im_end 248046 =
  eos, think 248068/248069); untied embeddings; vision tower and MTP in
  the config but skipped (text-only PoC; the release ships no MTP
  weights).

## What was built (src/cantollm/models/qwen38/, 9 commits)

Model, GDN token-loop scan (recurrent form only, ported from HF's
fallback), partial-RoPE helpers over the shared complex-multiply core,
tokenizer on the new vocab (tool tags added to content neutering),
`HybridCache` (sequential) and `HybridStatePool` (CB): KV tensors only
on the 16 full-attention indices, per-slot fp32 GDN S + conv state on
the rest. The pool satisfies the existing KVPool protocol; slot reuse
is handled by zeroing GDN state when a row arrives at start_pos 0, and
per-slot position counters make any non-monotone chunk (replay/skip)
fail loudly. Scheduler and allocator: zero changes.

Wiring outside the package is four touches: optional ModelSpec hooks
`cache_factory`/`kv_pool_factory` with runtime delegation,
`resolve_spec` family dispatch (`--model qwen38-27B`), the process
factory, and CLI knob forcing (qwen38 is eager-only: shape buckets,
warmup, CUDA graphs, torch.compile all refused/off; `--speculative`
refused since the recurrent state cannot rewind).

FP8: the official `Qwen/Qwen3.8-27B-FP8` checkpoint (30.9GB, e4m3 with
128x128 block scales) loads via a dtype-driven swap to `FP8Linear`
(weight-only, blockwise dequant to bf16 per matmul), meta-device init
with per-tensor materialization from a lazy sharded view, so peak host
RAM tracks checkpoint size.

## Validation record

- **HF parity** (the anchor): tiny seeded Qwen3_5ForCausalLM reference,
  weights loaded through the real mapping; max |dlogprob| 4.8e-7,
  argmax exact at every position.
- **Equivalence:** CB forward vs sequential einsum oracle across
  chunked-prefill/decode/filler mixes, live-slot-0 fillers, slot
  reuse; engine-level token-for-token vs SequentialEngine including
  concurrent requests. Full suite 633 passed.
- **5090 (2026-08-20, FP8, `--engine batched --max-batch 1/2
  --batch-max-seq-len 2048 --max-tokens-per-step 128 --attention sdpa
  --in-process`):** ready 2.3 s warm; 3/3 coherent generations
  including a 512-token chunked-prefill summarization; decode ~2.1
  tok/s per request, TTFT 0.97 s; max_batch 2 fits at peak 31773/32607
  MiB (~830 MiB headroom); no OOM, no CUDA asserts. Record in commit
  3226df6.

## Gotchas found during the round

- `serve --model qwen38-27B` without `--engine batched` takes the
  sequential default, boots, and its background factory starts the
  31GB download. The eager-only guard sits in the batched branch only.
- Conv-state test assertions must be allclose, never bitwise: the conv
  window is gathered from projection outputs, and matmul reduction
  order differs per shape across BLAS backends (equal on Mac ARM,
  ~2e-7 apart on the box).
- Thinking mode defaults on; small max_tokens budgets often finish
  inside the thinking block.

## If the PoC ever graduates

Assessed but deliberately not done:

- **Real fp8 GEMMs.** Today every step re-reads and dequantizes ~24GB
  to transient bf16 (~120GB traffic/step, hence ~2 tok/s). Two routes:
  (a) requantize weights per output row at load + dynamic per-token
  activation quant + `torch._scaled_mm`: ~1-2 days, blast radius is
  FP8Linear only; verify rowwise-scale support on sm_120 first and A/B
  the logit cost of coarser scales. (b) Checkpoint-faithful 128x128
  block-scaled W8A8 (Triton kernel, vLLM-style): ~1-2 weeks of kernel
  work. Either lands near a ~15 ms/step floor, ceiling ~60-70 tok/s,
  realistically 30-50 while the loop stays eager.
- Chunked-parallel GDN prefill (the token loop is the prefill cost).
- CUDA graphs / torch.compile over the hybrid pool; embed-on-CPU knob
  (next OOM-ladder rung, unwritten).
- Folding hybrid per-layer state into Phase 6's hybrid-KV-allocator
  prereq in PLAN.md, if this family joins the roadmap properly.

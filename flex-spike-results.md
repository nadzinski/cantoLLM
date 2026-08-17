# FlexAttention viability spike: 5090 results (2026-08-16)

The Phase 4 kernel-route gate check, run before any design or code. Verdict
up front: **all four gates pass; the paged attention path lands on
FlexAttention first, and flash-attn drops to a possible later A/B arm.**
Environment: torch 2.10.0+cu128, sm_120 (device capability 12.0),
Qwen3-0.6B attention geometry (16 query heads, 8 KV heads, head_dim 128,
bf16), repo at 61ce0b4. Run on the 5090 with the GPU otherwise idle;
scratch scripts only, nothing installed into the project venv, nothing
committed from the box.

## 1. What was being decided

Phase 4's paged read path has to move off `F.scaled_dot_product_attention`,
which takes no block tables. Two candidate routes: flash-attn's
varlen/paged entry points, or FlexAttention. The tentative call, made
before the spike, was Flex first: the paged index translation is the
educational core of the phase and Flex makes you write it; its eager
fallback fits the CPU oracle-equivalence test pattern; and Phase 6's
sliding window and Phase 8's architectures are each one more `mask_mod`
under Flex. Four gates stood between that call and commitment. A bad
failure on gate 2 (decode speed) or gate 4 (CUDA-graph capture) would have
flipped the order to flash-first.

## 2. Gate 1: version and feature floor. PASS

`torch.nn.attention.flex_attention` on this build has everything needed:
`flex_attention(..., enable_gqa=True)`, `BlockMask` + `create_block_mask`,
and the dedicated short-query decode kernel
(`torch._inductor.kernel.flex.flex_decoding`).

Better than expected: the paged route does not need a translating
`mask_mod` at all. `BlockMask.from_kv_blocks(kv_num_blocks, kv_indices,
..., seq_lengths=(q_len, kv_len), mask_mod=...)` builds the mask directly
from caller-owned int32 tensors, and `kv_indices` *is* a block table
(logical position to physical block id). The engine can own those tensors
outright.

API wrinkle for the design note: at decode, `from_kv_blocks` must be given
`seq_lengths=(1, S)` explicitly, or the mask defaults to a q_len of one
128-token block and `flex_attention` raises a shape `ValueError`. (The
error suggests `_adjust()`; passing `seq_lengths` at construction is
cleaner.) Also: `create_block_mask(..., _compile=True)` is deprecated in
2.10 in favor of `torch.compile(create_block_mask)`.

## 3. Gate 2: decode q_len=1 timing. PASS

`torch.compile(flex_attention, dynamic=False)` vs
`F.scaled_dot_product_attention(enable_gqa=True)` under
`sdpa_kernel([SDPBackend.CUDNN_ATTENTION])`. B=16, q_len=1, GQA 16/8,
head_dim 128, bf16; CUDA-event timing, 200 iterations after 30 warm-up.
Per call:

| KV length | cuDNN SDPA | compiled Flex | ratio |
| --------- | ---------: | ------------: | ----: |
| 512       |    12.6 µs |       24.6 µs | 1.96x |
| 1024      |    14.4 µs |       23.9 µs | 1.66x |
| 2048      |    84.2 µs |       87.8 µs | 1.04x |

The bar was "within 2x"; the result is inside it everywhere and the ratio
*shrinks* with KV length, reaching parity at 2048: the flex-decoding
split-KV path holds up exactly where the time is. Worst case against a
4 ms production decode step: ~12 µs per layer over 28 layers is ~0.3 ms,
and only if every row sat at the short-KV end; real mixes sit lower.
Numerics: max |diff| vs cuDNN <= 0.002, bf16 noise.

## 4. Gate 3: BlockMask per-step overhead. The concern dissolves

Construction costs measured at B=16, q=1, KV=2048: `create_block_mask`
0.60 ms eager / 0.50 ms compiled; `from_kv_blocks` 0.17 ms. None of these
are per-step costs. The reuse story is the engine's existing seeded-buffer
pattern: allocate `kv_num_blocks` / `kv_indices` once, build one
`BlockMask` over them, then mutate the tensors in place each step.
Verified: after `kv_num_blocks.fill_(nblk // 2)`, the *same* BlockMask
object through the *same* compiled callable computes the truncated result,
matching a from-scratch truncated reference (max |diff| 0.001, bf16
tolerance). The per-step CPU cost is writing ints into preallocated
tensors, which is what the scheduler already does with the KV write maps.

## 5. Gate 4: CUDA-graph capture. PASS, bit-exact

A compiled `flex_attention` call whose BlockMask came from
`from_kv_blocks` over engine-owned table tensors was captured in
`torch.cuda.CUDAGraph` after three side-stream warm-ups, no exception.
Replays reflect in-place table mutations, `torch.equal`-exact against
eager references, for all three states probed: (a) full tables, (b)
`kv_num_blocks` shrunk in place (a length update), (c) `kv_indices`
permuted in place (the actual paged logical-to-physical translation).
Same seeded-buffer discipline as the existing decode graphs; it composes.

## 6. flash-attn lookup (non-gating; wheel index only, nothing installed)

- No clean official wheel for this stack. Latest stable v2.8.3.post1
  (2026-06-10) ships wheels tagged up to torch2.9 only; no torch2.10 tag.
- sm_120: `setup.py` emits sm_100/sm_120 gencode only when built with
  CUDA >= 12.8, and the explicit SM120 work (Pack-GQA, SplitKV fallback)
  lands in the FA4 beta line, not FA2 stable. So the flash arm means a
  source build with the cu128 toolchain (community reports it works on the
  5090: issues #1987, #1683), an unofficial community wheel, or waiting on
  FA4.
- Paged-path constraint, from the current `flash_attn_with_kvcache`
  docstring: KV cache shaped `(num_blocks, page_block_size, nheads_k,
  headdim)` with "page_block_size must be a multiple of 256". Documented,
  not runtime-asserted. 256-token pages would forfeit most of the
  small-block fragmentation and sharing story, which makes flash a
  genuinely deferred arm here, not a close second.

Sources: the flash-attention GitHub releases page and
`flash_attn_interface.py` on main, as of 2026-08-16.

## 7. What this feeds the design note

1. Flex-first is confirmed; padded + the full Phase 3 stack stays the
   always-runnable reference arm and the bar to beat.
2. The paged interface is `BlockMask.from_kv_blocks` over engine-owned
   int32 tensors; carry the `seq_lengths=(1, S)` decode wrinkle into the
   implementation.
3. Per-step mask work must stay at "write ints into preallocated tensors";
   `create_block_mask` never runs in the step loop.
4. Flex only performs compiled, so torch.compile stops being an optional
   fifth piece and becomes required for attention: the compile and
   attention stacks merge, and the graphs story captures through
   Inductor-generated kernels.
5. Block size stays an allocator config knob. Flex imposes no constraint
   of its own; the flash arm, if ever taken, needs 256-token pages.

Exact APIs exercised: `torch.nn.attention.flex_attention.{flex_attention,
create_block_mask, BlockMask.from_kv_blocks}`, `enable_gqa=True`,
`torch.compile(flex_attention, dynamic=False)`, `torch.cuda.CUDAGraph`
with side-stream warm-up, `sdpa_kernel([SDPBackend.CUDNN_ATTENTION])` for
the baseline. The probe scripts (`flex_decode_bench.py`,
`blockmask_overhead.py`, `flex_graph_capture.py`) were scratch files on
the box and are not part of the repo; this document is the record.

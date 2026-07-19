"""SDPA dispatcher probe: which backends accept cantoLLM's attention call?

Run on a CUDA box (no model load — pure attention shapes at 0.6B geometry):

    .venv/bin/python bench/probe_sdpa.py

Phase 3's design decision (PLAN.md) says: keeping the explicit per-row bool
mask routes `F.scaled_dot_product_attention` to the memory-efficient backend,
because flash only applies masks it can compute from index arithmetic. That
claim is torch-version- and hardware-specific — this script checks it
empirically instead of trusting it. For each bench-shaped scenario (decode,
prefill chunk, long-context decode) it pins the dispatcher to each backend
via `torch.nn.attention.sdpa_kernel` and reports accept/reject + timing:

  - "ours"   — the production call: explicit bool mask (True = attend,
               broadcast over heads), enable_gqa=True. The decided design.
  - "causal" — the same shapes with is_causal=True and NO mask: what the
               Phase 4 lengths-metadata restructure would hand the kernel.
               (Semantically wrong for our ragged batches — PyTorch aligns
               the causal diagonal top-left — so this row is a speed
               preview, not a correctness option.)
  - "bare"   — no mask at all: isolates whether a rejection is about the
               mask or about shapes/dtype.

Interpreting: EFFICIENT (and possibly CUDNN) should accept "ours"; FLASH
should reject it but accept "causal"/"bare". The FLASH-vs-EFFICIENT gap on
the "causal" row is roughly what Phase 4's restructure has to gain.
"""
import time

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

assert torch.cuda.is_available(), "probe needs CUDA (run on the 5090)"
DEVICE = torch.device("cuda")
DTYPE = torch.bfloat16

# 0.6B geometry: 16 query heads over 8 KV groups, head_dim 128.
H, G, D = 16, 8, 128

# (name, batch, new tokens per row, history length) — mirrors the bench
# baseline shapes: steady decode at 16 slots, one 256-token prefill chunk
# against 512 cached, and longctx decode.
SCENARIOS = [
    ("decode 16x1 @512", 16, 1, 512),
    ("prefill 1x256 @768", 1, 256, 768),
    ("longctx 4x1 @8192", 4, 1, 8192),
]

BACKENDS = [
    SDPBackend.FLASH_ATTENTION,
    SDPBackend.EFFICIENT_ATTENTION,
    SDPBackend.CUDNN_ATTENTION,
    SDPBackend.MATH,
]

WARMUP, ITERS = 10, 50


def build(batch: int, new: int, hist: int):
    torch.manual_seed(0)
    q = torch.randn(batch, H, new, D, device=DEVICE, dtype=DTYPE)
    k = torch.randn(batch, G, hist, D, device=DEVICE, dtype=DTYPE)
    v = torch.randn(batch, G, hist, D, device=DEVICE, dtype=DTYPE)
    # Per-row causal fence, True = attend (SDPA's convention — the inverse
    # of build_batched_mask's), rows offset like a ragged batch: row b's
    # history is hist - b tokens, exercising per-row start_pos raggedness.
    i = torch.arange(new, device=DEVICE)
    j = torch.arange(hist, device=DEVICE)
    start = (hist - new) - torch.arange(batch, device=DEVICE).clamp(max=hist - new)
    keep = j[None, None, :] <= (start[:, None] + i[None, :])[:, :, None]
    mask = keep[:, None, :, :]  # broadcast over heads
    return q, k, v, mask


def timed(fn) -> float:
    for _ in range(WARMUP):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(ITERS):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / ITERS * 1e3


def attempt(backend, fn) -> str:
    try:
        with sdpa_kernel([backend]):
            fn()  # dispatch check (raises "No available kernel" on reject)
            return f"{timed(fn):8.3f} ms"
    except RuntimeError as exc:
        return f"reject ({str(exc).splitlines()[0][:40]})"


print(f"torch {torch.__version__} on {torch.cuda.get_device_name(0)}")
print(f"q/k/v {DTYPE}, {H} query heads / {G} KV groups / head_dim {D}, enable_gqa=True\n")

for name, batch, new, hist in SCENARIOS:
    q, k, v, mask = build(batch, new, hist)
    calls = {
        "ours": lambda: F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, enable_gqa=True
        ),
        "causal": lambda: F.scaled_dot_product_attention(
            q, k, v, is_causal=True, enable_gqa=True
        ),
        "bare": lambda: F.scaled_dot_product_attention(q, k, v, enable_gqa=True),
    }
    print(f"── {name} " + "─" * (46 - len(name)))
    print(f"{'':>10} " + " ".join(f"{b.name.split('_')[0]:>14}" for b in BACKENDS))
    for variant, fn in calls.items():
        cells = [f"{attempt(b, fn):>14}" for b in BACKENDS]
        print(f"{variant:>10} " + " ".join(cells))
    print()

print(
    "The 'ours' row is the decided Phase-3 call. If EFFICIENT rejects it,\n"
    "the design needs revisiting; if FLASH accepts it, the explicit-mask\n"
    "compromise is obsolete (good news — revisit PLAN.md Phase 3/4)."
)

"""Standalone repro: index_put_ into a select view of a stacked input vs
directly into per-layer input tensors, under torch.compile/Inductor.

Mimics the KV-pool write: pool (L, B, S, G, D), per-layer scatter
layer_k[slot, pos] = new, chained across layers like the model's loop.
Measures step time and peak memory for each form at a pool sized like the
16x4096 serve geometry (28, 16, 4097, 8, 128) bf16 (~3.5 GiB).
"""
import time

import torch

L, B, S, G, D = 28, 16, 4097, 8, 128
N = 16  # write entries per layer (16-row decode)

slot = torch.arange(N, device="cuda")
pos = torch.full((N,), 100, device="cuda", dtype=torch.long)
new = torch.randn(N, G, D, device="cuda", dtype=torch.bfloat16)


def stacked_form(pool_k, slot, pos, new):
    x = new
    for i in range(L):
        layer_k = pool_k[i]
        layer_k[slot, pos] = x
        x = x + layer_k[slot, pos]  # read-after-write, like the gather
    return x


def perlayer_form(layers, slot, pos, new):
    x = new
    for layer_k in layers:
        layer_k[slot, pos] = x
        x = x + layer_k[slot, pos]
    return x


def bench(fn, *args, n=30):
    for _ in range(3):
        fn(*args)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    for _ in range(n):
        fn(*args)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n * 1000, torch.cuda.max_memory_allocated() / 2**30


with torch.inference_mode():
    pool_k = torch.zeros(L, B, S, G, D, device="cuda", dtype=torch.bfloat16)
    base = torch.cuda.memory_allocated() / 2**30
    print(f"pool: {pool_k.numel() * 2 / 2**30:.2f} GiB, allocated {base:.2f} GiB")

    ms, peak = bench(stacked_form, pool_k, slot, pos, new)
    print(f"eager  stacked:   {ms:6.2f} ms  peak {peak:.2f} GiB")

    c_stacked = torch.compile(stacked_form, fullgraph=True)
    ms, peak = bench(c_stacked, pool_k, slot, pos, new)
    print(f"compiled stacked:  {ms:6.2f} ms  peak {peak:.2f} GiB   <- view-of-input mutation")

    del pool_k
    torch.cuda.empty_cache()
    layers = [torch.zeros(B, S, G, D, device="cuda", dtype=torch.bfloat16) for _ in range(L)]

    ms, peak = bench(perlayer_form, layers, slot, pos, new)
    print(f"eager  per-layer: {ms:6.2f} ms  peak {peak:.2f} GiB")

    c_perlayer = torch.compile(perlayer_form, fullgraph=True)
    ms, peak = bench(c_perlayer, layers, slot, pos, new)
    print(f"compiled per-layer:{ms:6.2f} ms  peak {peak:.2f} GiB   <- direct input mutation")

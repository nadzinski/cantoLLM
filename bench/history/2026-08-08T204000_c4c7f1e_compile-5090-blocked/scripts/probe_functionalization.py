"""Decisive probe: does the serve-geometry compiled arm (16x4096, sdpa,
dynamic) pay the functionalized full-pool copy per layer?

Compares eager vs compiled 16-row and 1-row decode steps at the A/B serve
geometry, reports ms/step and peak CUDA memory, no scheduler involved.
"""
import time

import torch

from cantollm.engine.batching import BatchingConfig
from cantollm.engine.batching.config import default_shape_buckets
from cantollm.models.attention.protocol import BatchMeta
from cantollm.runtime import build_runtime
from cantollm.spec import qwen3_spec

DEVICE = torch.device("cuda")


def decode_meta(batch: int, start: int, kv_len: int) -> BatchMeta:
    return BatchMeta(
        rows=[(slot, start, 1) for slot in range(batch)],
        slots=torch.arange(batch),
        start_pos=torch.full((batch,), start),
        num_new=torch.ones(batch, dtype=torch.int64),
        positions=torch.full((batch, 1), start),
        num_new_max=1,
        max_history_len=kv_len,
        device=DEVICE,
    )


def time_steps(runtime, pool, metas, n=50):
    for meta in metas:  # warm / compile
        ids = torch.zeros((len(meta.rows), 1), dtype=torch.int64, device=DEVICE)
        for _ in range(3):
            runtime.forward_batched(ids, meta, pool)
    torch.cuda.synchronize()
    out = {}
    for meta in metas:
        b = len(meta.rows)
        ids = torch.zeros((b, 1), dtype=torch.int64, device=DEVICE)
        t0 = time.perf_counter()
        for _ in range(n):
            runtime.forward_batched(ids, meta, pool)
        torch.cuda.synchronize()
        out[b] = (time.perf_counter() - t0) / n * 1000
    return out


print("loading 0.6B (sdpa)...", flush=True)
runtime = build_runtime(qwen3_spec("0.6B"), DEVICE, attention="sdpa")
config = BatchingConfig(
    max_batch=16, max_seq_len=4096, max_tokens_per_step=512,
    **default_shape_buckets(16, 512),
)
pool = runtime.new_kv_pool(config)

# NOTE: metas are single-use per timing loop on purpose; kv_write_map is
# forced once by the runtime front and cached, so steps are repeatable.
metas = [decode_meta(1, 128, 256), decode_meta(16, 128, 256)]

torch.cuda.reset_peak_memory_stats()
eager = time_steps(runtime, pool, metas)
eager_peak = torch.cuda.max_memory_allocated() / 2**30
print(f"eager   ms/step: 1-row {eager[1]:.2f}  16-row {eager[16]:.2f}  "
      f"peak {eager_peak:.2f} GiB", flush=True)

runtime.enable_torch_compile("dynamic")
# fresh metas: the compiled front marks dims on the map tensors; reuse of the
# eager-run metas is fine but keep it clean
metas = [decode_meta(1, 128, 256), decode_meta(16, 128, 256)]
torch.cuda.reset_peak_memory_stats()
t0 = time.perf_counter()
compiled = time_steps(runtime, pool, metas)
compiled_peak = torch.cuda.max_memory_allocated() / 2**30
print(f"compile wall (incl. artifact builds): {time.perf_counter()-t0:.1f} s")
print(f"compiled ms/step: 1-row {compiled[1]:.2f}  16-row {compiled[16]:.2f}  "
      f"peak {compiled_peak:.2f} GiB", flush=True)

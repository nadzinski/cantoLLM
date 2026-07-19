"""Bottleneck probe: dissect ContinuousBatchingScheduler.step() wall time.

Run on a CUDA box (loads 0.6B):

    .venv/bin/python bench/profile_step.py

Findings as of 2026-07-18 are written up in step-profiling.md; this script
regenerates them. Two phases:

Phase A — component decomposition at occupancy 1/8/16/32/48 (one 48-slot
scheduler, topped up between sweeps). The forward is split into
'fwd_call' (CPU time for forward_batched to *return* — Python dispatch +
kernel launches, no GPU waiting) and 'fwd_sync' (explicit synchronize
afterward — the GPU catching up). In a launch-bound regime fwd_call
dominates; in a compute-bound one fwd_sync does. 'rest' is step() minus
all probes: the finalize loop's event/list churn plus its per-row
token .item() / logprob .log().item() syncs.

Caveat: sequences persist across the occupancy sweeps, so later rows
carry ~1k tokens of KV history — longer than the bench baseline's cells.
That inflates fwd_sync at 32/48 rows relative to bench numbers; the
low-occupancy rows and the fwd_call column are unaffected.

Phase B — torch.profiler at the baseline server shape (16 slots, 16
rows): top ops by self CUDA and self CPU time, kernel-launch counts.
"""
import time

import torch

import cantollm.engine.batching.scheduler as sched_mod
import cantollm.engine.sampler as sampler_mod
from cantollm.engine.batching import BatchingConfig
from cantollm.engine.batching.engine import scheduler_from_runtime
from cantollm.engine.types import InferenceRequest, SamplingParams
from cantollm.runtime import build_runtime
from cantollm.spec import qwen3_spec

PROMPT_LEN = 64
DEVICE = torch.device("cuda")

print("loading 0.6B...", flush=True)
runtime = build_runtime(qwen3_spec("0.6B"), DEVICE, attention="padded")

acc: dict[str, float] = {}
orig_sample = sampler_mod.sample


def wrap(obj, name, key):
    orig = getattr(obj, name)

    def timed(*a, **k):
        t0 = time.perf_counter()
        out = orig(*a, **k)
        acc[key] = acc.get(key, 0.0) + time.perf_counter() - t0
        return out

    setattr(obj, name, timed)


def instrument(sched):
    wrap(sched, "_plan_step", "plan")
    wrap(sched, "_build_input_ids", "build_ids")
    wrap(sched_mod, "build_batch_meta", "meta")
    wrap(sampler_mod, "sample", "sample")
    orig_fwd = sched.forward_fn

    def timed_fwd(*a, **k):
        t0 = time.perf_counter()
        out = orig_fwd(*a, **k)
        t1 = time.perf_counter()
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        acc["fwd_call"] = acc.get("fwd_call", 0.0) + t1 - t0
        acc["fwd_sync"] = acc.get("fwd_sync", 0.0) + t2 - t1
        return out

    sched.forward_fn = timed_fwd


def add_requests(sched, n, start):
    for i in range(n):
        sched.add_request(InferenceRequest(
            request_id=f"r{start + i}",
            prompt_token_ids=list(range(100, 100 + PROMPT_LEN)),
            sampling_params=SamplingParams.from_temperature_top_p(0.0, 1.0),
            max_tokens=3500,
            stop_token_ids=set(),
        ))


def run_steps(sched, n):
    t0 = time.perf_counter()
    for _ in range(n):
        sched.step()
    torch.cuda.synchronize()
    return time.perf_counter() - t0


# ── Phase A ──────────────────────────────────────────────────────────
config48 = BatchingConfig(max_batch=48, max_seq_len=4096, max_tokens_per_step=256)
sched = scheduler_from_runtime(runtime, config48)
instrument(sched)

N = 200
print(f"\n=== step() decomposition, ms/step over {N} decode steps ===")
cols = ["plan", "build_ids", "meta", "fwd_call", "fwd_sync", "sample", "rest"]
print(f"{'rows':>4} {'total':>7} " + " ".join(f"{c:>9}" for c in cols))
for occ in (1, 8, 16, 32, 48):
    add_requests(sched, occ - len(sched.active), start=len(sched.active))
    while sched.queued or any(s.is_prefilling() for s in sched.active):
        sched.step()
    assert len(sched.active) == occ
    run_steps(sched, 20)  # warm
    acc.clear()
    total = run_steps(sched, N)
    parts = {k: acc.get(k, 0.0) / N * 1000 for k in cols[:-1]}
    parts["rest"] = total / N * 1000 - sum(parts.values())
    print(f"{occ:>4} {total / N * 1000:>7.2f} "
          + " ".join(f"{parts[c]:>9.3f}" for c in cols))

del sched
sampler_mod.sample = orig_sample
torch.cuda.empty_cache()

# ── Phase B ──────────────────────────────────────────────────────────
from torch.profiler import ProfilerActivity, profile

config16 = BatchingConfig(max_batch=16, max_seq_len=4096, max_tokens_per_step=256)
sched = scheduler_from_runtime(runtime, config16)
add_requests(sched, 16, start=1000)
while sched.queued or any(s.is_prefilling() for s in sched.active):
    sched.step()
for _ in range(20):
    sched.step()
torch.cuda.synchronize()

STEPS = 20
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    for _ in range(STEPS):
        sched.step()
    torch.cuda.synchronize()

ka = prof.key_averages()
n_kernel_launches = sum(k.count for k in ka if k.key == "cudaLaunchKernel")
print(f"\n=== profiler, 16 rows x {STEPS} steps ===")
print(f"cudaLaunchKernel calls/step: {n_kernel_launches / STEPS:.0f}")
print("\n--- top 12 by self CUDA time ---")
print(ka.table(sort_by="self_cuda_time_total", row_limit=12))
print("\n--- top 12 by self CPU time ---")
print(ka.table(sort_by="self_cpu_time_total", row_limit=12))

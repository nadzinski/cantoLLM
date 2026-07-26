"""Capture real CUDA graphs for the viz "CUDA graphs" explainer tab.

Unlike the trace_*.py harnesses (which regenerate locally, output gitignored),
this needs CUDA hardware and its artifacts are COMMITTED under viz/captures/
as the record the tab's curated inline data is transcribed from. Run it on the
infra/ GPU node (see infra/README.md); nothing here regenerates on the Mac.

    python viz/capture_cudagraphs.py --dry-run     # Mac/CPU: build everything, no capture
    python viz/capture_cudagraphs.py               # CUDA: full capture -> viz/captures/cudagraphs-<date>/

Produces, per example: a Graphviz DOT topology dump (kernel nodes + dependency
edges, via CUDAGraph debug mode) and eager-vs-replay timings (CUDA events for
GPU wall time, perf_counter for CPU submit time). Plus a size sweep on the
elementwise chain, and recorded rule-breaking demos (real error text).

Examples:
  chain      five-op elementwise chain (mul, add, relu, mul, add), fp32
  forkjoin   two matmul branches from one input joined by add + bias + relu, bf16
  tinymodel  one decode microstep of tests/tiny_model.py's 2-layer 64-dim Qwen3
             (embed -> blocks -> RMSNorm -> lm head), fp32, fixed position

Rule-breaking demos (recorded, run last since a failed capture may sour the
CUDA context):
  stale_input     replay without copy_ into the static input buffer
  shape_mismatch  feeding a bigger tensor into the baked static buffer
  cache_no_advance  the concat-grown sequential KVCache never advances on replay
  sync_in_capture  .item() inside a capture region (hard error)
"""

import argparse
import datetime
import json
import subprocess
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

WARMUP_ITERS = 20
TIMING_ITERS = 200


# ---------------- example builders ----------------
# Each returns (static_inputs, fn) where fn reads/writes only static tensors.


def build_chain(device, n=1 << 20):
    x = torch.randn(n, device=device)

    def fn():
        a = x * 1.01
        b = a + 0.5
        c = torch.relu(b)
        d = c * 0.9
        return d + a

    return {"x": x}, fn


def build_forkjoin(device, parallel_streams=False):
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    x = torch.randn(64, 256, device=device, dtype=dtype)
    w1 = torch.randn(256, 256, device=device, dtype=dtype)
    w2 = torch.randn(256, 256, device=device, dtype=dtype)
    b = torch.randn(256, device=device, dtype=dtype)

    if not parallel_streams:
        # Written as a fork-join, but captured on ONE stream: the recorded
        # graph is a straight chain, because edges come from stream order.
        def fn():
            y1 = x @ w1
            y2 = x @ w2
            return torch.relu(y1 + y2 + b)

        return {"x": x}, fn

    # Two streams + fork/join events: capture records genuine parallel
    # branches. Streams/events must be created outside the captured region.
    side = torch.cuda.Stream()
    fork_ev = torch.cuda.Event()
    join_ev = torch.cuda.Event()

    def fn():
        cur = torch.cuda.current_stream()
        fork_ev.record(cur)
        with torch.cuda.stream(side):
            side.wait_event(fork_ev)
            y2 = x @ w2
            join_ev.record(side)
        y1 = x @ w1
        cur.wait_event(join_ev)
        return torch.relu(y1 + y2 + b)

    return {"x": x}, fn


def build_tinymodel(device, prefill_len=16):
    import cantollm.engine  # noqa: F401  (must precede runtime: engine<->runtime import cycle)
    from cantollm.kv_cache import KVCache
    from cantollm.runtime import build_runtime
    from tests.tiny_model import tiny_qwen3_spec

    runtime = build_runtime(tiny_qwen3_spec(), device)
    model = runtime.model
    model.eval()

    cache = KVCache(len(model.transformer_blocks))
    prompt = torch.randint(0, 2048, (1, prefill_len), device=device)
    with torch.no_grad():
        model(prompt, 0, kv_cache=cache)

    token = torch.randint(0, 2048, (1, 1), device=device)

    def fn():
        with torch.no_grad():
            return model(token, prefill_len, kv_cache=cache)

    def reset():
        cache.truncate(prefill_len)

    return {"token": token, "cache": cache}, fn, reset


# ---------------- capture + measurement ----------------


def warm_up(fn, reset=None):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
            if reset:
                reset()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()


def capture(fn, dot_path=None):
    # keep_graph=True retains the raw cudaGraph_t so debug_dump can print it
    # (the enable_debug_mode path writes nothing on this torch build);
    # instantiation is deferred, so trigger it explicitly before replaying.
    g = torch.cuda.CUDAGraph(keep_graph=True) if dot_path else torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = fn()
    if dot_path:
        g.debug_dump(str(dot_path))
        g.instantiate()
    return g, out


def time_loop(launch, iters=TIMING_ITERS):
    """Returns (gpu_us_per_iter, cpu_submit_us_per_iter)."""
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    t0 = time.perf_counter()
    start.record()
    for _ in range(iters):
        launch()
    end.record()
    t1 = time.perf_counter()  # after submission, before completion
    torch.cuda.synchronize()
    gpu_us = start.elapsed_time(end) * 1000 / iters
    cpu_us = (t1 - t0) * 1e6 / iters
    return round(gpu_us, 2), round(cpu_us, 2)


def measure(name, fn, g, timings, note=None):
    eager_gpu, eager_cpu = time_loop(fn)
    graph_gpu, graph_cpu = time_loop(g.replay)
    timings[name] = {
        "eager_gpu_us_per_iter": eager_gpu,
        "eager_cpu_submit_us_per_iter": eager_cpu,
        "graph_gpu_us_per_iter": graph_gpu,
        "graph_cpu_submit_us_per_iter": graph_cpu,
        "iters": TIMING_ITERS,
    }
    if note:
        timings[name]["note"] = note
    print(f"  {name}: eager {eager_gpu} us/iter (submit {eager_cpu}) "
          f"vs graph {graph_gpu} us/iter (submit {graph_cpu})")


# ---------------- rule-breaking demos ----------------


def record_failures(device, failures):
    # 1. Stale input: replay without copying into the static buffer.
    x = torch.zeros(4, device=device)
    g, y = capture(lambda: x * 2 + 1)
    x.copy_(torch.tensor([1.0, 2.0, 3.0, 4.0], device=device))
    g.replay()
    torch.cuda.synchronize()
    after_copy = y.tolist()
    fresh = torch.tensor([10.0, 20.0, 30.0, 40.0], device=device)  # noqa: F841 (the mistake: rebinding, not copy_)
    g.replay()
    torch.cuda.synchronize()
    failures["stale_input"] = {
        "after_copy_into_static": after_copy,
        "after_rebinding_python_name": y.tolist(),
        "lesson": "the graph reads the captured buffer's memory; a new tensor bound to the old name is invisible to it",
    }
    print("  stale_input: recorded")

    # 2. Shape mismatch: the static buffer's shape is baked.
    big = torch.randn(8, device=device)
    try:
        x.copy_(big)
    except RuntimeError as e:
        failures["shape_mismatch"] = {
            "error": str(e),
            "lesson": "a graph replays fixed shapes and pointers; a new shape means a new capture (why the engine's shape vocabulary exists)",
        }
        print("  shape_mismatch: recorded")

    # 3. Sync inside capture: run LAST, a failed capture can sour the context.
    x2 = torch.randn(16, device=device)
    try:
        g2 = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g2):
            v = x2.sum()
            v.item()  # D2H sync inside a capture region
    except RuntimeError as e:
        failures["sync_in_capture"] = {
            "error": str(e),
            "lesson": "capture records launches without running them, so anything that needs a result NOW (item, print, assert) cannot be answered",
        }
        print("  sync_in_capture: recorded")


def record_cache_no_advance(device, failures):
    statics, fn, reset = build_tinymodel(device)
    cache = statics["cache"]
    warm_up(fn, reset)
    pos_before = cache.position
    g, logits = capture(fn)
    pos_after_capture = cache.position
    outs = []
    for _ in range(3):
        g.replay()
        torch.cuda.synchronize()
        outs.append(logits.flatten()[:4].tolist())
    failures["cache_no_advance"] = {
        "cache_position_before_capture": pos_before,
        "cache_position_after_capture": pos_after_capture,
        "cache_position_after_3_replays": cache.position,
        "logits_identical_across_replays": outs[0] == outs[1] == outs[2],
        "lesson": "the sequential KVCache grows by reallocating (concat); capture bakes one step's buffers, so replay recomputes the same step forever. A graphable decode needs preallocated static KV (the padded pool) with device-tensor indices",
    }
    print("  cache_no_advance: recorded")


# ---------------- main ----------------


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true",
                    help="build all examples on CPU, no capture (Mac-safe check)")
    ap.add_argument("--out", type=Path, default=None,
                    help="output dir (default viz/captures/cudagraphs-<date>)")
    args = ap.parse_args()

    if args.dry_run:
        device = torch.device("cpu")
        for name, built in [("chain", build_chain(device)),
                            ("forkjoin", build_forkjoin(device)),
                            ("tinymodel", build_tinymodel(device)[:2])]:
            _, fn = built
            out = fn()
            print(f"dry-run {name}: ok, output shape {tuple(out.shape)}")
        print("dry-run complete (no CUDA, nothing captured)")
        return

    if not torch.cuda.is_available():
        sys.exit("CUDA is required (use --dry-run on the Mac); see infra/README.md")

    date = datetime.date.today().isoformat()
    out_dir = args.out or REPO_ROOT / "viz" / "captures" / f"cudagraphs-{date}"
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")

    driver = subprocess.run(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
        capture_output=True, text=True).stdout.strip()
    env = {
        "gpu": torch.cuda.get_device_name(0),
        "capability": ".".join(map(str, torch.cuda.get_device_capability(0))),
        "driver": driver,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "date": date,
    }
    print(f"capturing on {env['gpu']} (sm_{env['capability'].replace('.', '')}), "
          f"torch {env['torch']}, CUDA {env['cuda']}")

    timings, failures = {}, {}

    print("chain:")
    statics, fn = build_chain(device)
    warm_up(fn)
    g, _ = capture(fn, out_dir / "chain.dot")
    measure("chain", fn, g, timings)

    print("chain size sweep:")
    for n in (4096, 1 << 20, 1 << 24):
        statics, fn = build_chain(device, n=n)
        warm_up(fn)
        g, _ = capture(fn)
        measure(f"chain_n{n}", fn, g, timings,
                note=f"{n} elements; the graph win is launch overhead, so it fades as kernels grow")

    print("forkjoin (one stream: the order trap):")
    statics, fn = build_forkjoin(device)
    warm_up(fn)
    g, _ = capture(fn, out_dir / "forkjoin_1stream.dot")
    measure("forkjoin_1stream", fn, g, timings,
            note="written as a fork-join, captured on one stream: the DOT is a "
                 "straight chain, stream order became the edges")

    print("forkjoin (two streams: a real DAG):")
    statics, fn = build_forkjoin(device, parallel_streams=True)
    warm_up(fn)
    g, _ = capture(fn, out_dir / "forkjoin_2stream.dot")
    measure("forkjoin_2stream", fn, g, timings,
            note="fork/join via events across two streams: the gemms record "
                 "as genuinely parallel branches")

    print("tinymodel:")
    statics, fn, reset = build_tinymodel(device)
    warm_up(fn, reset)
    g, _ = capture(fn, out_dir / "tinymodel.dot")
    reset()  # capture left the concat-grown cache one position long

    def eager_once():
        fn()
        reset()  # hold the eager loop at the same fixed position replay replays

    measure("tinymodel", eager_once, g, timings,
            note="both sides repeat the position-16 decode step; eager pays a "
                 "host-side cache truncate per iter (cheap slicing), replay "
                 "stays at the baked step (see failures.cache_no_advance)")

    print("rule-breaking demos:")
    record_cache_no_advance(device, failures)
    record_failures(device, failures)

    (out_dir / "env.json").write_text(json.dumps(env, indent=2) + "\n")
    (out_dir / "timings.json").write_text(json.dumps(timings, indent=2) + "\n")
    (out_dir / "failures.json").write_text(json.dumps(failures, indent=2) + "\n")
    dots = sorted(p.name for p in out_dir.glob("*.dot"))
    print(f"\nwrote {out_dir}/: env.json, timings.json, failures.json, {', '.join(dots)}")


if __name__ == "__main__":
    main()

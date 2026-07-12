"""Trace harness for the API/engine process split.

Runs the REAL split — `EngineProcessClient` in this (parent) process, the
real Qwen3-0.6B scheduler in a spawned engine process — and records both
sides of the boundary. The child side instruments itself: the scheduler
factory below is module-level (pickled by reference across spawn), wraps
`add_request` / `_plan_step` / `step` on the real scheduler, and dumps its
half to a JSON file at process exit. The parent side wraps
`EngineProcessClient._dispatch` and times each request stream. Both sides
stamp `time.time()` on the same host, so cross-process latencies subtract
directly. Zero changes to src/.

Run from the repo root:  .venv/bin/python viz/trace_split.py
(~40s — the engine process loads the 0.6B weights)
Writes:                  viz/data/trace_split.js  (window.TRACE_SPLIT = ...)
"""

import asyncio
import json
import os
import time
from pathlib import Path

from trace_common import DATA_DIR, emit_js

CHILD_JSON = DATA_DIR / "trace_split_child.json"


# ── child side: module-level factory, pickled by reference ───────────


def traced_qwen3_scheduler(size, device, config, trace_path):
    """Build the real runtime + scheduler inside the engine process, with
    recording wrappers on the seams (same instance-attribute wrapping the
    other harnesses use — no source changes)."""
    import atexit
    import pickle

    import torch

    from cantollm.engine.batching.engine import scheduler_from_runtime
    from cantollm.runtime import build_runtime
    from cantollm.spec import qwen3_spec

    rec = {"pid": os.getpid(), "t_factory_start": time.time(), "adds": [], "steps": []}

    runtime = build_runtime(qwen3_spec(size), torch.device(device), attention="padded")
    scheduler = scheduler_from_runtime(runtime, config)
    rec["t_factory_done"] = time.time()

    orig_add = scheduler.add_request
    orig_plan = scheduler._plan_step
    orig_step = scheduler.step
    plan_box = {}

    def add_request(req):
        rec["adds"].append({
            "rid": req.request_id, "t": time.time(),
            "prompt_len": len(req.prompt_token_ids), "max_tokens": req.max_tokens,
        })
        return orig_add(req)

    def plan_step():
        rows = orig_plan()
        plan_box["rows"] = [{
            "rid": r.sequence.request_id, "slot": r.sequence.slot_idx,
            "start": r.start_pos, "n": r.num_new,
            "phase": "prefill" if r.sequence.is_prefilling() else "decode",
        } for r in rows]
        return rows

    def step():
        t0 = time.time()
        events = orig_step()
        rec["steps"].append({
            "t_emit": time.time(),
            "step_ms": round((time.time() - t0) * 1000, 1),
            "rows": plan_box.pop("rows", []),
            "events": [{"rid": e.request_id, "tok": e.token_id, "fin": e.finish_reason}
                       for e in events],
            "batch_bytes": len(pickle.dumps(events)),
        })
        return events

    scheduler.add_request = add_request
    scheduler._plan_step = plan_step
    scheduler.step = step
    atexit.register(lambda: Path(trace_path).write_text(json.dumps(rec)))
    return scheduler


# ── parent side ───────────────────────────────────────────────────────


REQUESTS = [
    # (rid, user message, max_tokens) — lengths differ so the water-fill
    # plan and prefill/decode mixing show up in the per-step rows.
    ("r-short", "Say hello. /no_think", 10),
    ("r-medium", "Name the four seasons of the year, comma-separated. /no_think", 14),
    ("r-long", "You are given the list [3, 1, 4, 1, 5, 9, 2, 6]. State its length, "
               "its largest element, and its smallest element. /no_think", 18),
]


async def main():
    from cantollm.engine.batching import BatchingConfig, EngineProcessClient
    from cantollm.engine.types import InferenceRequest, SamplingParams
    from cantollm.runtime import build_tokenizer_runtime
    from cantollm.spec import qwen3_spec

    config = BatchingConfig(max_batch=3, max_seq_len=192, max_tokens_per_step=24)
    spec = qwen3_spec("0.6B")
    tokenizer = build_tokenizer_runtime(spec).tokenizer  # no weights in this process

    CHILD_JSON.parent.mkdir(exist_ok=True)
    if CHILD_JSON.exists():
        CHILD_JSON.unlink()

    client = EngineProcessClient(traced_qwen3_scheduler, {
        "size": "0.6B", "device": "cpu", "config": config,
        "trace_path": str(CHILD_JSON),
    })
    # CPU on purpose: greedy fp32-ish determinism and no device contention
    # with whatever else the machine is doing — this trace is about the
    # boundary, not the forward pass.

    dispatches = []
    t_spawn = time.time()
    print("spawning engine process + loading 0.6B (~30s)…")
    await client.start()
    t_ready = time.time()
    orig_dispatch = client._dispatch

    def dispatch(events):
        dispatches.append({"t": time.time(), "n": len(events)})
        orig_dispatch(events)

    client._dispatch = dispatch
    print(f"ready in {t_ready - t_spawn:.1f}s (engine pid {client._proc.pid})")

    greedy = SamplingParams.from_temperature_top_p(0.0, 1.0)
    req_meta = {}

    async def run_request(rid, text, max_tokens):
        ids = tokenizer.encode_conversation([{"role": "user", "content": text}])
        req = InferenceRequest(
            request_id=rid, prompt_token_ids=ids, sampling_params=greedy,
            max_tokens=max_tokens, stop_token_ids=set(tokenizer.stop_token_ids),
        )
        meta = {"prompt": text, "prompt_len": len(ids), "max_tokens": max_tokens,
                "t_submit": time.time(), "tokens": []}
        req_meta[rid] = meta
        async for evt in client.submit(req):
            if evt.token_id is not None:
                if "t_first" not in meta:
                    meta["t_first"] = time.time()
                meta["tokens"].append(evt.token_id)
            if evt.finish_reason is not None:
                meta["finish"] = evt.finish_reason
                meta["t_done"] = time.time()
        meta["text"] = tokenizer.decode(meta["tokens"])

    await asyncio.gather(*(run_request(*r) for r in REQUESTS))
    print(f"{sum(len(m['tokens']) for m in req_meta.values())} tokens streamed "
          f"across {len(dispatches)} dispatched batches")

    t_shut = time.time()
    await client.shutdown()
    t_shut_done = time.time()
    exitcode = client._proc.exitcode
    print(f"shutdown in {(t_shut_done - t_shut) * 1000:.0f}ms, engine exitcode {exitcode}")

    child = json.loads(CHILD_JSON.read_text())
    CHILD_JSON.unlink()

    # Every step that produced events → exactly one dispatch, in order
    # (drive_scheduler only emits non-empty batches; the bridge forwards
    # them FIFO). Pair them up for the IPC hop latency.
    emitting = [s for s in child["steps"] if s["events"]]
    assert len(emitting) == len(dispatches), (len(emitting), len(dispatches))
    for s, d in zip(emitting, dispatches):
        s["hop_ms"] = round((d["t"] - s["t_emit"]) * 1000, 2)

    ms = lambda t: round((t - t_spawn) * 1000)  # noqa: E731
    trace = {
        "meta": {
            "model": spec.name, "device": "cpu",
            "parent_pid": os.getpid(), "child_pid": child["pid"],
            "config": {"max_batch": config.max_batch, "max_seq_len": config.max_seq_len,
                       "max_tokens_per_step": config.max_tokens_per_step},
        },
        "boot": {
            "spawn_to_factory_ms": ms(child["t_factory_start"]),
            "load_ms": round((child["t_factory_done"] - child["t_factory_start"]) * 1000),
            "ready_ms": ms(t_ready),
        },
        "requests": {
            rid: {
                "prompt": m["prompt"], "prompt_len": m["prompt_len"],
                "max_tokens": m["max_tokens"], "n_tokens": len(m["tokens"]),
                "finish": m["finish"], "text": m["text"],
                "t_submit": ms(m["t_submit"]), "t_first": ms(m["t_first"]),
                "t_done": ms(m["t_done"]),
                "ttft_ms": round((m["t_first"] - m["t_submit"]) * 1000),
            } for rid, m in req_meta.items()
        },
        "adds": [{"rid": a["rid"], "t": ms(a["t"]),
                  "cross_ms": round((a["t"] - req_meta[a["rid"]]["t_submit"]) * 1000, 2)}
                 for a in child["adds"]],
        "steps": [{
            "i": i, "t": ms(s["t_emit"]), "step_ms": s["step_ms"],
            "rows": s["rows"], "batch_bytes": s["batch_bytes"],
            "n_events": len(s["events"]), "hop_ms": s.get("hop_ms"),
            "events": s["events"],
        } for i, s in enumerate(child["steps"])],
        "shutdown": {"ms": round((t_shut_done - t_shut) * 1000), "exitcode": exitcode},
    }
    hops = sorted(s["hop_ms"] for s in emitting)
    trace["totals"] = {
        "steps": len(child["steps"]),
        "events": sum(s["n_events"] for s in trace["steps"]),
        "batches": len(dispatches),
        "bytes": sum(s["batch_bytes"] for s in trace["steps"]),
        "hop_ms_p50": hops[len(hops) // 2],
        "hop_ms_max": hops[-1],
    }
    out = emit_js("trace_split.js", "TRACE_SPLIT", trace)
    print(f"wrote {out} ({len(trace['steps'])} steps, "
          f"hop p50 {trace['totals']['hop_ms_p50']}ms)")


if __name__ == "__main__":
    asyncio.run(main())

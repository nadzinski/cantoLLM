"""Trace harness for the continuous-batching prototype.

Imports the REAL prototype code (scheduler, toy model, padded KV) and records
a step-by-step trace by wrapping instance methods at runtime — zero changes
to the prototype source. The trace drives the "Continuous batching" view in
viz/index.html.

Run from the repo root:  .venv/bin/python viz/trace_cb.py
Writes:                  viz/data/trace_cb.js  (window.TRACE_CB = ...)
"""

import random
import sys
from pathlib import Path

PROTO_ROOT = Path(__file__).resolve().parents[1] / "prototypes"
sys.path.insert(0, str(PROTO_ROOT))

from trace_common import emit_js  # noqa: E402

import continuous_batching.scheduler as sched_mod  # noqa: E402
from continuous_batching.cb_types import Request  # noqa: E402
from continuous_batching.padded_kv import PaddedKVCache  # noqa: E402
from continuous_batching.scheduler import ContinuousBatchingScheduler  # noqa: E402
from continuous_batching.toy_model import ToyModel  # noqa: E402

VOCAB = 32
DIM = 16
MAX_BATCH = 3
MAX_SEQ_LEN = 20
BUDGET = 6

MODEL_SEED = 23  # chosen (with prompt seed 1) for varied greedy outputs
rng = random.Random(1)


def make_prompt(n):
    return [rng.randrange(1, VOCAB) for _ in range(n)]


PROMPTS = {
    "A": make_prompt(5),
    "B": make_prompt(14),
    "C": make_prompt(3),
    "D": make_prompt(4),
}
MAX_TOKENS = {"A": 6, "B": 4, "C": 5, "D": 4}
ARRIVALS = {0: ["A", "B"], 1: ["C"], 3: ["D"]}


def run_scenario(stop_tokens_by_id):
    model = ToyModel(vocab_size=VOCAB, dim=DIM, seed=MODEL_SEED)
    cache = PaddedKVCache(max_batch=MAX_BATCH, max_seq_len=MAX_SEQ_LEN, dim=DIM)
    sched = ContinuousBatchingScheduler(model, cache, max_tokens_per_step=BUDGET)

    # --- runtime instrumentation (no source changes) ---
    capture = {}

    orig_plan = sched._plan_step

    def plan_wrapper():
        rows = orig_plan()
        capture["rows"] = [
            {
                "id": r.sequence.request_id,
                "slot": r.sequence.slot_idx,
                "start_pos": r.start_pos,
                "num_new": r.num_new,
                "requested": (
                    len(r.sequence.prompt_token_ids) - r.start_pos
                    if r.start_pos < len(r.sequence.prompt_token_ids)
                    else 1
                ),
                "phase": "prefill"
                if r.start_pos < len(r.sequence.prompt_token_ids)
                else "decode",
                "input_tokens": list(r.input_tokens),
            }
            for r in rows
        ]
        return rows

    sched._plan_step = plan_wrapper

    orig_build = sched._build_input_ids

    def build_wrapper(rows):
        m = orig_build(rows)
        capture["input_matrix"] = m.tolist()
        return m

    sched._build_input_ids = build_wrapper

    orig_sample = sched_mod.greedy_sample

    def sample_wrapper(logits):
        out = orig_sample(logits)
        capture["sampled"] = out.tolist()
        return out

    sched_mod.greedy_sample = sample_wrapper

    # --- drive the loop ---
    def queue_snapshot():
        return [
            {"id": s.request_id, "prompt_len": len(s.prompt_token_ids)}
            for s in sched.queued_sequences
        ]

    def active_snapshot():
        return [
            {
                "id": s.request_id,
                "slot": s.slot_idx,
                "position": s.position,
                "prompt_len": len(s.prompt_token_ids),
                "outputs": list(s.output_token_ids),
            }
            for s in sched.active_sequences
        ]

    steps = []
    step_num = 0
    max_arrival = max(ARRIVALS)
    while not sched.is_idle() or step_num <= max_arrival:
        arrived = []
        for rid in ARRIVALS.get(step_num, []):
            sched.add_request(
                Request(
                    request_id=rid,
                    prompt_token_ids=PROMPTS[rid],
                    max_tokens=MAX_TOKENS[rid],
                    stop_token_ids=set(stop_tokens_by_id.get(rid, [])),
                )
            )
            arrived.append(rid)

        active_before = {s.request_id for s in sched.active_sequences}
        queue_before = queue_snapshot()
        capture.clear()

        events = sched.step()

        rows = capture.get("rows", [])
        slot_by_id = {r["id"]: r["slot"] for r in rows}
        ev_by_id = {e.request_id: e for e in events}
        outcomes = []
        for r, tok in zip(rows, capture.get("sampled", [])):
            ev = ev_by_id.get(r["id"])
            outcomes.append(
                {
                    "id": r["id"],
                    "sampled": tok,
                    "emitted": ev is not None,
                    "finish_reason": ev.finish_reason if ev else None,
                }
            )

        steps.append(
            {
                "step": step_num,
                "arrived": arrived,
                "queue_before": queue_before,
                "admitted": [
                    {"id": r["id"], "slot": r["slot"]}
                    for r in rows
                    if r["id"] not in active_before
                ],
                "budget": BUDGET,
                "rows": rows,
                "input_matrix": capture.get("input_matrix", []),
                "outcomes": outcomes,
                "freed_slots": [
                    {"id": e.request_id, "slot": slot_by_id[e.request_id]}
                    for e in events
                    if e.finish_reason is not None
                ],
                "queue_after": queue_snapshot(),
                "active_after": active_snapshot(),
            }
        )
        step_num += 1
        if step_num > 60:
            raise RuntimeError("runaway scenario")

    sched_mod.greedy_sample = orig_sample
    return steps


# Pass 1: no stop tokens; find what A emits so we can pick a stop token that
# ends A early via end_turn (per-request stop set, so only A is affected).
pass1 = run_scenario({})
a_outputs = []
for st in pass1:
    for o in st["outcomes"]:
        if o["id"] == "A" and o["emitted"]:
            a_outputs.append(o["sampled"])
stop_for_a = a_outputs[3]  # A's 4th token becomes its stop token
print(f"A outputs (pass 1): {a_outputs} -> stop token {stop_for_a}", file=sys.stderr)

steps = run_scenario({"A": [stop_for_a]})

trace = {
    "generator": "viz/trace_cb.py",
    "config": {
        "max_batch": MAX_BATCH,
        "max_seq_len": MAX_SEQ_LEN,
        "budget": BUDGET,
        "vocab": VOCAB,
    },
    "requests": {
        rid: {
            "prompt": PROMPTS[rid],
            "prompt_len": len(PROMPTS[rid]),
            "max_tokens": MAX_TOKENS[rid],
            "arrival_step": next(s for s, ids in ARRIVALS.items() if rid in ids),
            "stop_tokens": [stop_for_a] if rid == "A" else [],
        }
        for rid in PROMPTS
    },
    "steps": steps,
}

out = emit_js("trace_cb.js", "TRACE_CB", trace)
print(f"wrote {out} ({len(steps)} steps)", file=sys.stderr)

# Sanity summary
for st in steps:
    parts = []
    for r, o in zip(st["rows"], st["outcomes"]):
        fin = f" FIN:{o['finish_reason']}" if o["finish_reason"] else ""
        emit = "emit" if o["emitted"] else "drop"
        parts.append(
            f"{r['id']}[slot{r['slot']} {r['phase']} @{r['start_pos']}+{r['num_new']}/{r['requested']} {emit} {o['sampled']}{fin}]"
        )
    q = ",".join(x["id"] for x in st["queue_after"])
    print(f"step {st['step']:2d}: {' '.join(parts)}  queue=[{q}]", file=sys.stderr)

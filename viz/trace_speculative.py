"""Trace harness for speculative decoding (draft 0.6B + main 1.7B).

Runs one real greedy generation through SpeculativeBackend and records every
iteration — draft proposals, main verification, accept/reject, the tail
sample, and both cache truncations — by wrapping instance/class methods at
runtime. Zero changes to the cantollm source. Also times a plain
StandardBackend run of the main model on the same prompt as a baseline, and
checks the two outputs match token-for-token (speculative decoding must
preserve the main model's greedy output).

Run from the repo root:  .venv/bin/python viz/trace_speculative.py
Writes:                  viz/data/trace_spec.js  (window.TRACE_SPEC)

Loads BOTH models (~5 GB); expect ~1 minute of loading + ~1-2 min generation.
"""

import asyncio
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")

import torch  # noqa: E402

from trace_common import emit_js  # noqa: E402

PROMPT = "What is 7 times 8? Reply with just the number."
MAX_TOKENS = 512
MAIN_SIZE = "1.7B"
DRAFT_SIZE = "0.6B"


def log(msg):
    print(msg, file=sys.stderr)


def pick_device():
    return torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")


def sync(device):
    if device.type == "mps":
        torch.mps.synchronize()


def main():
    device = pick_device()
    log(f"device: {device}")

    import cantollm.engine  # noqa: F401  (import order resolves the runtime<->engine cycle)
    from cantollm.engine.sequential import SequentialEngine
    from cantollm.engine.types import SamplingParams, Sequence
    from cantollm.kv_cache import KVCache
    from cantollm.runtime import build_runtime
    from cantollm.spec import qwen3_spec
    from cantollm.standard import StandardBackend

    t0 = time.perf_counter()
    runtime = build_runtime(qwen3_spec(MAIN_SIZE), device, speculative=qwen3_spec(DRAFT_SIZE))
    log(f"models loaded in {time.perf_counter() - t0:.1f}s")
    engine = SequentialEngine(runtime)
    backend = runtime.backend  # SpeculativeBackend
    tokenizer = runtime.tokenizer

    # ---- instrumentation (no source changes) ----
    events = []

    orig_gen_draft = backend.generate_draft_tokens

    def gen_draft_wrapper(input_tokens, num_tokens, sampling, stop_token_ids, draft_cache):
        sync(device)
        t = time.perf_counter()
        tokens, probs = orig_gen_draft(input_tokens, num_tokens, sampling, stop_token_ids, draft_cache)
        sync(device)
        events.append({
            "type": "draft",
            "input": list(input_tokens) if len(input_tokens) <= 4 else len(input_tokens),
            "tokens": list(tokens),
            "ms": round((time.perf_counter() - t) * 1000, 1),
        })
        return tokens, probs

    backend.generate_draft_tokens = gen_draft_wrapper

    orig_verify = backend._verify_draft_tokens

    def verify_wrapper(draft_tokens, draft_probs, main_probs, sampling):
        accepted = orig_verify(draft_tokens, draft_probs, main_probs, sampling)
        events.append({
            "type": "verify",
            "accepted": len(accepted),
            "main_argmax": torch.argmax(main_probs, dim=-1).tolist(),
        })
        return accepted

    backend._verify_draft_tokens = verify_wrapper

    orig_main_fwd = backend.main.forward

    def main_fwd_wrapper(token_ids, cache, start_pos):
        sync(device)
        t = time.perf_counter()
        out = orig_main_fwd(token_ids, cache, start_pos)
        sync(device)
        events.append({
            "type": "main_fwd", "n": len(token_ids), "pos": start_pos,
            "ms": round((time.perf_counter() - t) * 1000, 1),
        })
        return out

    backend.main.forward = main_fwd_wrapper

    # The draft cache is per-generate-call now; inject one through the
    # `draft_cache` parameter so the truncate wrapper can identify it.
    traced_draft_cache = KVCache(backend.draft_num_layers)
    orig_generate = backend.generate

    def generate_wrapper(sequence, draft_cache=None):
        return orig_generate(sequence, draft_cache=traced_draft_cache)

    backend.generate = generate_wrapper

    orig_truncate = KVCache.truncate

    def truncate_wrapper(self, pos):
        which = "draft" if self is traced_draft_cache else "main"
        events.append({"type": "truncate", "cache": which, "from": self.position, "to": pos})
        return orig_truncate(self, pos)

    KVCache.truncate = truncate_wrapper

    # ---- drive one request through the real engine ----
    from cantollm.api.common import tokenize_and_build_request

    async def run():
        with ThreadPoolExecutor(max_workers=1) as pool:
            req = await tokenize_and_build_request(
                messages=[{"role": "user", "content": PROMPT}],
                system=None,
                sampling_params=SamplingParams.from_temperature_top_p(0.0, 1.0),
                max_tokens=MAX_TOKENS,
                tokenizer=tokenizer,
                executor=pool,
            )
        t_start = time.perf_counter()
        emitted, finish = [], None
        async for evt in engine.submit(req):
            if evt.token_id is not None:
                emitted.append(evt.token_id)
            elif evt.finish_reason is not None:
                finish = evt.finish_reason
            if evt.error:
                raise RuntimeError(evt.error)
        return req, emitted, finish, (time.perf_counter() - t_start) * 1000

    req, emitted, finish, spec_ms = asyncio.run(run())
    stats = backend.get_stats()
    log(f"speculative: {len(emitted)} tokens in {spec_ms / 1000:.1f}s "
        f"({len(emitted) / (spec_ms / 1000):.1f} tok/s) · finish={finish}")
    log(f"stats: proposed={stats.draft_tokens_proposed} accepted={stats.draft_tokens_accepted} "
        f"({stats.acceptance_rate:.1%}) · {stats.tokens_per_iteration:.2f} tok/iter × {stats.iterations} iters")

    # ---- baseline: main model alone through StandardBackend ----
    KVCache.truncate = orig_truncate
    baseline_backend = StandardBackend(model=runtime.model, device=device)
    base_seq = Sequence(
        request_id="baseline",
        prompt_token_ids=req.prompt_token_ids,
        sampling_params=SamplingParams(greedy=True),
        stop_token_ids=req.stop_token_ids,
        max_tokens=MAX_TOKENS,
        cache=KVCache(len(runtime.new_cache())),
        stop_event=threading.Event(),
    )
    sync(device)
    t = time.perf_counter()
    baseline_tokens = list(baseline_backend.generate(base_seq))
    sync(device)
    base_ms = (time.perf_counter() - t) * 1000
    log(f"baseline (main alone): {len(baseline_tokens)} tokens in {base_ms / 1000:.1f}s "
        f"({len(baseline_tokens) / (base_ms / 1000):.1f} tok/s)")

    match = baseline_tokens == emitted
    if not match:
        div = next((i for i, (a, b) in enumerate(zip(baseline_tokens, emitted)) if a != b),
                   min(len(baseline_tokens), len(emitted)))
        log(f"WARNING: outputs diverge at token {div} (batched-vs-single forward numerics)")
    else:
        log("outputs match token-for-token: speculative output == main-alone output")

    # ---- assemble per-iteration records ----
    piece = lambda t: tokenizer.decode([t])  # noqa: E731
    iterations = []
    cur = None
    for e in events:
        if e["type"] == "draft":
            if cur:
                iterations.append(cur)
            cur = {"draft": e, "main_fwd": None, "verify": None, "truncates": []}
        elif cur is not None:
            if e["type"] == "main_fwd":
                cur["main_fwd"] = e
            elif e["type"] == "verify":
                cur["verify"] = e
            elif e["type"] == "truncate":
                cur["truncates"].append(e)
    if cur:
        iterations.append(cur)

    stop_ids = set(req.stop_token_ids)
    out_iters = []
    emitted_so_far = 0
    for it in iterations:
        drafts = it["draft"]["tokens"]
        n_acc = it["verify"]["accepted"] if it["verify"] else 0
        argmax = it["verify"]["main_argmax"] if it["verify"] else []
        tail = argmax[n_acc] if n_acc < len(argmax) else None
        all_tokens = drafts[:n_acc] + ([tail] if tail is not None else [])
        remaining = MAX_TOKENS - emitted_so_far
        n_emit = 0
        hit_stop = False
        for tok in all_tokens[:remaining]:
            if tok in stop_ids:
                hit_stop = True
                break
            n_emit += 1
        emitted_so_far += n_emit
        out_iters.append({
            "i": len(out_iters),
            "draft_input": it["draft"]["input"],
            "draft_tokens": drafts,
            "draft_pieces": [piece(t) for t in drafts],
            "draft_ms": it["draft"]["ms"],
            "main_n": it["main_fwd"]["n"] if it["main_fwd"] else None,
            "main_pos": it["main_fwd"]["pos"] if it["main_fwd"] else None,
            "main_ms": it["main_fwd"]["ms"] if it["main_fwd"] else None,
            "accepted": n_acc,
            "tail": tail,
            "tail_piece": piece(tail) if tail is not None else None,
            "tail_is_bonus": n_acc == len(drafts),
            "n_emitted": n_emit,
            "hit_stop": hit_stop,
            "truncates": it["truncates"],
        })

    trace = {
        "meta": {
            "main": f"Qwen3-{MAIN_SIZE}", "draft": f"Qwen3-{DRAFT_SIZE}",
            "device": str(device), "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "speculative_tokens": backend.speculative_tokens,
            "prompt_text": PROMPT, "max_tokens": MAX_TOKENS,
        },
        "prompt": {"ids": req.prompt_token_ids, "len": len(req.prompt_token_ids)},
        "iterations": out_iters,
        "emitted": {"ids": emitted, "pieces": [piece(t) for t in emitted], "finish": finish},
        "stats": {
            "proposed": stats.draft_tokens_proposed,
            "accepted": stats.draft_tokens_accepted,
            "acceptance_rate": round(stats.acceptance_rate, 4),
            "tokens_per_iteration": round(stats.tokens_per_iteration, 2),
            "iterations": stats.iterations,
            "spec_ms": round(spec_ms, 1),
            "spec_tok_per_s": round(len(emitted) / (spec_ms / 1000), 1),
            "baseline_ms": round(base_ms, 1),
            "baseline_tok_per_s": round(len(baseline_tokens) / (base_ms / 1000), 1),
            "baseline_tokens": len(baseline_tokens),
            "outputs_match": match,
        },
    }
    out = emit_js("trace_spec.js", "TRACE_SPEC", trace)
    log(f"wrote {out} ({len(out_iters)} iterations)")

    # sanity summary: first few iterations
    for it in out_iters[:8]:
        marks = "".join("✓" if j < it["accepted"] else "✗" for j in range(len(it["draft_tokens"])))
        kind = "bonus" if it["tail_is_bonus"] else "fix"
        log(f"iter {it['i']:3d}: {marks:10s} +{kind} {it['tail_piece']!r} → emit {it['n_emitted']}")


if __name__ == "__main__":
    main()

"""Trace harness for the real Qwen3-0.6B inference path.

Runs one short greedy chat completion through the REAL stack — Qwen3Tokenizer
→ SequentialEngine → StandardBackend → Qwen3 → EinsumAttentionMethod/KVCache
— and records tensor shapes, per-step token data, and the request lifecycle
by instrumenting at runtime: torch forward hooks plus instance/module
attribute wrapping. Zero changes to the cantollm source.

Run from the repo root:  .venv/bin/python viz/trace_forward.py
Writes:                  viz/data/trace_forward.js    (window.TRACE_FORWARD)
                         viz/data/trace_tokenflow.js  (window.TRACE_TOKENFLOW)

Model load takes ~20-40s; generation a few seconds more.
"""

import asyncio
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

# Weights are already on disk under src/cantollm/models/model_data/Qwen3-0.6B;
# stay offline so hf_hub_download resolves them without the network.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")  # mirrors main.py

import torch  # noqa: E402

from trace_common import emit_js  # noqa: E402

PROMPT = "What is 7 times 8? Reply with just the number."
MAX_TOKENS = 512  # 0.6B's greedy thinking is long-winded even for arithmetic
RETRY_MAX_TOKENS = 1024  # if thinking never ends within MAX_TOKENS
TOPK = 5


def log(msg):
    print(msg, file=sys.stderr)


def pick_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_runtime(device):
    # Import the engine package first: cantollm.runtime and cantollm.engine
    # are mutually referential, and only this order resolves cleanly.
    import cantollm.engine  # noqa: F401
    from cantollm.runtime import build_runtime
    from cantollm.spec import qwen3_spec

    try:
        return build_runtime(qwen3_spec("0.6B"), device)
    except Exception as e:
        if os.environ.get("HF_HUB_OFFLINE") != "1":
            raise
        log(f"offline load failed ({e}); retrying with network access")
        os.environ["HF_HUB_OFFLINE"] = "0"
        import huggingface_hub.constants as hf_constants

        hf_constants.HF_HUB_OFFLINE = False
        return build_runtime(qwen3_spec("0.6B"), device)


# ---------------------------------------------------------------------------
# Capture state + instrumentation (no source changes)
# ---------------------------------------------------------------------------


def new_capture():
    return {
        "forward_idx": -1,
        "capture_modules": False,
        "modules": [],  # hook records for the forward being captured
        "detailed": {},  # "prefill"/"decode" -> harvested module records
        "steps": [],  # one record per backend.forward call
        "samples": [],  # one record per backend.sample call
        "mask_by_forward": {},
        "rope_by_forward": {},
        "rope_seen": set(),
        "encodes": [],  # tokenizer.encode calls (ChatML text -> ids)
        "decoder_adds": [],  # IncrementalDecoder.add per generated text token
        "decoder_flush": [],
    }


def reset_capture(cap):
    cap.update(new_capture())


def _tensor_shapes(args):
    return [list(a.shape) for a in args if isinstance(a, torch.Tensor)]


def install_module_hooks(model, cap):
    """register_forward_hook on every named submodule; shapes only."""

    def make_hook(name):
        def hook(module, args, output):
            if not cap["capture_modules"]:
                return
            cap["modules"].append(
                {
                    "name": name,
                    "cls": type(module).__name__,
                    "in": _tensor_shapes(args),
                    "out": list(output.shape) if isinstance(output, torch.Tensor) else None,
                    "dtype": str(output.dtype).replace("torch.", "")
                    if isinstance(output, torch.Tensor)
                    else None,
                }
            )

        return hook

    handles = []
    for name, module in model.named_modules():
        if name:  # skip the root; backend.forward wrap covers it
            handles.append(module.register_forward_hook(make_hook(name)))
    return handles


def instrument(runtime, cap):
    backend = runtime.backend
    model = runtime.model
    tokenizer = runtime.tokenizer
    device = runtime.device

    install_module_hooks(model, cap)

    # Mask shapes: one shared EinsumAttentionMethod instance builds the causal
    # mask once per model forward (model.py:289).
    attention_method = model.attention_method
    orig_build_mask = attention_method.build_mask

    def build_mask_wrapper(start_pos, seq_len, mask_device):
        mask = orig_build_mask(start_pos, seq_len, mask_device)
        cap["mask_by_forward"][cap["forward_idx"]] = {
            "start_pos": start_pos,
            "seq_len": seq_len,
            "shape": list(mask.shape),
        }
        return mask

    attention_method.build_mask = build_mask_wrapper

    # RoPE offsets: apply_rotary_emb is import-bound into the model module
    # (model.py:5), so patch that namespace — patching cantollm.models.rope
    # would not affect the call sites.
    import cantollm.models.qwen3.model as qwen3_model_mod

    orig_rope = qwen3_model_mod.apply_rotary_emb

    def rope_wrapper(x, freqs_cis, offset=0):
        key = (cap["forward_idx"], tuple(x.shape), offset)
        if key not in cap["rope_seen"]:  # called 2x28 per forward, identical
            cap["rope_seen"].add(key)
            cap["rope_by_forward"].setdefault(cap["forward_idx"], []).append(
                {
                    "x_shape": list(x.shape),
                    "offset": offset,
                    "freqs_slice": [offset, offset + x.shape[1]],
                }
            )
        return orig_rope(x, freqs_cis, offset=offset)

    qwen3_model_mod.apply_rotary_emb = rope_wrapper

    # Per-step forward: token ids in, KV growth, logits shape, timing.
    orig_forward = backend.forward

    def forward_wrapper(token_ids, cache, start_pos):
        idx = cap["forward_idx"] + 1
        cap["forward_idx"] = idx
        cap["capture_modules"] = idx in (0, 1)
        kv_before = cache.position
        if device.type == "mps":
            torch.mps.synchronize()
        t0 = time.perf_counter()
        logits = orig_forward(token_ids, cache, start_pos)
        if device.type == "mps":
            torch.mps.synchronize()
        ms = (time.perf_counter() - t0) * 1000.0
        if cap["capture_modules"]:
            cap["detailed"]["prefill" if idx == 0 else "decode"] = cap["modules"]
            cap["modules"] = []
            cap["capture_modules"] = False
        cap["steps"].append(
            {
                "i": idx,
                "phase": "prefill" if idx == 0 else "decode",
                "input_ids": list(token_ids),
                "start_pos": start_pos,
                "kv_before": kv_before,
                "kv_after": cache.position,
                "kv_layer_shape": list(cache[0]["keys"].shape),
                "logits_shape": list(logits.shape),
                "ms": round(ms, 2),
                "worker_thread": threading.get_ident(),
            }
        )
        return logits

    backend.forward = forward_wrapper

    # Sampling: top-k candidates + the greedy pick.
    orig_sample = backend.sample

    def sample_wrapper(logits, sampling):
        token, probs = orig_sample(logits, sampling)
        p = probs.float()
        top = torch.topk(p, min(TOPK, p.shape[-1]), dim=-1)
        picked = int(token.reshape(-1)[0].item())
        cap["samples"].append(
            {
                "topk_ids": top.indices[0].tolist(),
                "topk_probs": [round(v, 6) for v in top.values[0].tolist()],
                "picked": picked,
                "picked_prob": round(float(p[0, picked].item()), 6),
            }
        )
        return token, probs

    backend.sample = sample_wrapper

    # Tokenizer: capture the ChatML frame encode_conversation builds. It
    # assembles token ids directly (content is plain-BPE encoded so request
    # bodies can't forge control tokens), so wrap the method itself and
    # reconstruct the template text by decoding — decode() renders special
    # tokens, so the ChatML string round-trips faithfully.
    orig_encode_conversation = tokenizer.encode_conversation

    def encode_conversation_wrapper(messages, system=None):
        ids = orig_encode_conversation(messages, system=system)
        cap["encodes"].append({"text": tokenizer.decode(ids), "ids": list(ids)})
        return ids

    tokenizer.encode_conversation = encode_conversation_wrapper

    # Incremental decoder: which tokens produced stable text vs were buffered
    # (multi-byte sequences). Class-level patch; new decoders are created per
    # request inside phase_tagged_events.
    from cantollm.models.qwen3.tokenizer import IncrementalDecoder

    orig_add = IncrementalDecoder.add

    def add_wrapper(self, token_id):
        text = orig_add(self, token_id)
        cap["decoder_adds"].append({"token_id": token_id, "text": text})
        return text

    IncrementalDecoder.add = add_wrapper

    orig_decoder_flush = IncrementalDecoder.flush

    def decoder_flush_wrapper(self):
        text = orig_decoder_flush(self)
        if text:
            cap["decoder_flush"].append(text)
        return text

    IncrementalDecoder.flush = decoder_flush_wrapper


# ---------------------------------------------------------------------------
# Drive one request through the real engine
# ---------------------------------------------------------------------------


async def run_once(engine, tokenizer, max_tokens):
    from cantollm.api.common import tokenize_and_build_request
    from cantollm.api.phase import DecodeState, phase_tagged_events
    from cantollm.engine.types import SamplingParams
    from cantollm.stream_events import TextChunk, ThinkingEndEvent, ThinkingStartEvent

    messages = [{"role": "user", "content": PROMPT}]
    with ThreadPoolExecutor(max_workers=1) as pool:
        req = await tokenize_and_build_request(
            messages=messages,
            system=None,
            sampling_params=SamplingParams.from_temperature_top_p(0.0, 1.0),
            max_tokens=max_tokens,
            tokenizer=tokenizer,
            executor=pool,
        )

    t0 = time.perf_counter()
    token_events = []

    async def tee(aiter):
        async for evt in aiter:
            token_events.append(
                {
                    "token_id": evt.token_id,
                    "finish_reason": evt.finish_reason,
                    "error": evt.error,
                    "t_ms": round((time.perf_counter() - t0) * 1000.0, 1),
                    "consumer_thread": threading.get_ident(),
                }
            )
            yield evt

    state = DecodeState()
    stream_events = []
    src = tee(engine.submit(req))
    try:
        async for phase, sev in phase_tagged_events(src, tokenizer, state):
            match sev:
                case ThinkingStartEvent():
                    stream_events.append({"phase": phase, "kind": "thinking_start"})
                case ThinkingEndEvent():
                    stream_events.append({"phase": phase, "kind": "thinking_end"})
                case TextChunk(text=t):
                    stream_events.append({"phase": phase, "kind": "text_chunk", "text": t})
    finally:
        await src.aclose()

    total_ms = (time.perf_counter() - t0) * 1000.0
    return {
        "messages": messages,
        "req": req,
        "token_events": token_events,
        "decode_state": state,
        "stream_events": stream_events,
        "total_ms": total_ms,
    }


# ---------------------------------------------------------------------------
# Post-processing
# ---------------------------------------------------------------------------


def summarize_modules(records, num_blocks):
    """Collapse per-block hook records: keep block 0, assert 1..N-1 identical."""
    top_level = []
    blocks = {}
    for r in records:
        name = r["name"]
        if name.startswith("transformer_blocks."):
            _, block_idx, *rest = name.split(".", 2)
            entry = dict(r, name=rest[0] if rest else "(block)")
            blocks.setdefault(int(block_idx), []).append(entry)
        else:
            top_level.append(r)

    def signature(rs):
        return json.dumps([[r["name"], r["cls"], r["in"], r["out"]] for r in rs])

    base = blocks.get(0, [])
    identical = all(
        signature(blocks.get(i, [])) == signature(base) for i in range(1, num_blocks)
    )
    return {
        "top_level": top_level,
        "block": base,
        "repeated_layers": {"count": num_blocks, "identical": identical},
    }


def build_traces(runtime, cap, run, device):
    tokenizer = runtime.tokenizer
    arch = runtime.spec.arch
    req = run["req"]
    piece = lambda tid: tokenizer.decode([tid])  # noqa: E731

    num_blocks = arch["num_transformers"]
    heads_per_group = arch["num_heads"] // arch["num_groups"]
    meta = {
        "model": f"Qwen3-{runtime.spec.size}",
        "device": str(device),
        "dtype": str(arch["dtype"]).replace("torch.", ""),
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "prompt_text": PROMPT,
        "arch": {
            "layers": num_blocks,
            "dim": arch["token_embedding_dim"],
            "heads": arch["num_heads"],
            "groups": arch["num_groups"],
            "heads_per_group": heads_per_group,
            "head_dim": arch["head_dim"],
            "q_out_dim": arch["num_heads"] * arch["head_dim"],
            "kv_dim": arch["num_groups"] * arch["head_dim"],
            "expanded_dim": arch["expanded_dim"],
            "vocab": arch["token_count"],
            "max_seq_len": arch["max_seq_len"],
        },
    }

    # --- TRACE_FORWARD: per-step tensor story ---
    steps = []
    prompt_len = len(req.prompt_token_ids)
    for fwd, smp in zip(cap["steps"], cap["samples"]):
        i = fwd["i"]
        assert smp["picked"] == smp["topk_ids"][0], "greedy pick must equal top-1"
        if fwd["phase"] == "decode":
            assert fwd["kv_after"] == prompt_len + i, (
                f"kv position {fwd['kv_after']} != prompt_len {prompt_len} + step {i}"
            )
        steps.append(
            {
                "i": i,
                "phase": fwd["phase"],
                "input_ids": fwd["input_ids"],
                "input_pieces": [piece(t) for t in fwd["input_ids"]],
                "start_pos": fwd["start_pos"],
                "mask": cap["mask_by_forward"].get(i, {}).get("shape"),
                "kv_before": fwd["kv_before"],
                "kv_after": fwd["kv_after"],
                "kv_layer_shape": fwd["kv_layer_shape"],
                "logits_shape": fwd["logits_shape"],
                "topk": {
                    "ids": smp["topk_ids"],
                    "probs": smp["topk_probs"],
                    "pieces": [piece(t) for t in smp["topk_ids"]],
                },
                "picked": {
                    "id": smp["picked"],
                    "piece": piece(smp["picked"]),
                    "prob": smp["picked_prob"],
                },
                "is_stop": smp["picked"] in req.stop_token_ids,
                "ms": fwd["ms"],
            }
        )

    detailed = {}
    for phase_name in ("prefill", "decode"):
        records = cap["detailed"].get(phase_name, [])
        fwd_idx = 0 if phase_name == "prefill" else 1
        detailed[phase_name] = {
            "modules": summarize_modules(records, num_blocks),
            "mask": cap["mask_by_forward"].get(fwd_idx),
            "rope": cap["rope_by_forward"].get(fwd_idx, []),
        }

    decode_steps = [s for s in steps if s["phase"] == "decode"]
    decode_ms = sum(s["ms"] for s in decode_steps)
    emitted = sum(1 for e in run["token_events"] if e["token_id"] is not None)
    trace_forward = {
        "meta": meta,
        "prompt": {
            "ids": req.prompt_token_ids,
            "pieces": [piece(t) for t in req.prompt_token_ids],
            "len": prompt_len,
        },
        "detailed": detailed,
        "steps": steps,
        "totals": {
            "forwards": len(steps),
            "tokens_emitted": emitted,
            "prefill_ms": steps[0]["ms"] if steps else None,
            "decode_ms_avg": round(decode_ms / len(decode_steps), 2) if decode_steps else None,
            "decode_tok_per_s": round(len(decode_steps) / (decode_ms / 1000.0), 1)
            if decode_ms
            else None,
        },
    }

    # --- TRACE_TOKENFLOW: request lifecycle ---
    chatml = next(
        (e for e in cap["encodes"] if e["ids"] == req.prompt_token_ids), None
    )
    thinking_start_id = tokenizer.thinking_start_id
    thinking_end_id = tokenizer.thinking_end_id

    adds = list(cap["decoder_adds"])
    add_pos = 0
    events = []
    in_thinking = False
    for e in run["token_events"]:
        if e["token_id"] is None:
            continue
        tid = e["token_id"]
        marker = None
        emitted_text = None
        buffered = False
        if tid == thinking_start_id:
            marker = "think_start"
            in_thinking = True
            event_phase = "thinking"
        elif tid == thinking_end_id:
            marker = "think_end"
            in_thinking = False
            event_phase = "thinking"
        else:
            event_phase = "thinking" if in_thinking else "text"
            assert add_pos < len(adds) and adds[add_pos]["token_id"] == tid, (
                "decoder_adds out of sync with token events"
            )
            emitted_text = adds[add_pos]["text"]
            buffered = emitted_text == ""
            add_pos += 1
        events.append(
            {
                "idx": len(events),
                "token_id": tid,
                "piece": piece(tid),
                "t_ms": e["t_ms"],
                "phase": event_phase,
                "marker": marker,
                "emitted_text": emitted_text,
                "buffered": buffered,
            }
        )

    state = run["decode_state"]
    thinking_text = "".join(
        s.get("text", "") for s in run["stream_events"] if s["phase"] == "thinking"
    )
    visible_text = "".join(
        s.get("text", "") for s in run["stream_events"] if s["phase"] == "text"
    )
    first_token = next(
        (e for e in run["token_events"] if e["token_id"] is not None), None
    )
    trace_tokenflow = {
        "meta": meta,
        "request": {
            "messages": run["messages"],
            "system": None,
            "chatml_text": chatml["text"] if chatml else None,
            "prompt": {
                "ids": req.prompt_token_ids,
                "pieces": [piece(t) for t in req.prompt_token_ids],
            },
            "request_id": req.request_id,
            "max_tokens": req.max_tokens,
            "stop_token_ids": sorted(req.stop_token_ids),
            "sampling": "greedy (temperature=0)",
        },
        "engine": {
            "kind": "SequentialEngine",
            "queue_maxsize": 256,
            "worker_thread": cap["steps"][0]["worker_thread"] if cap["steps"] else None,
            "consumer_thread": run["token_events"][0]["consumer_thread"]
            if run["token_events"]
            else None,
        },
        "special_tokens": {
            "think_start": {"id": thinking_start_id, "text": "<think>"},
            "think_end": {"id": thinking_end_id, "text": "</think>"},
            "eos": {"id": tokenizer.eos_token_id, "text": piece(tokenizer.eos_token_id)},
            "pad": {"id": tokenizer.pad_token_id, "text": piece(tokenizer.pad_token_id)},
        },
        "events": events,
        "stream_events": run["stream_events"],
        "flush_text": "".join(cap["decoder_flush"]),
        "finish": {
            "finish_reason": state.finish_reason,
            "error": state.error,
            "counts": {
                "thinking": state.thinking,
                "text": state.text,
                "total": state.total,
            },
        },
        "text": {"thinking": thinking_text, "visible": visible_text},
        "timing": {
            "ttft_ms": first_token["t_ms"] if first_token else None,
            "total_ms": round(run["total_ms"], 1),
        },
    }

    return trace_forward, trace_tokenflow


def main():
    device = pick_device()
    log(f"device: {device}")
    t0 = time.perf_counter()
    runtime = load_runtime(device)
    log(f"model loaded in {time.perf_counter() - t0:.1f}s")

    from cantollm.engine.sequential import SequentialEngine

    engine = SequentialEngine(runtime)
    cap = new_capture()
    instrument(runtime, cap)

    max_tokens = MAX_TOKENS
    run = asyncio.run(run_once(engine, runtime.tokenizer, max_tokens))
    if run["decode_state"].text == 0 and run["decode_state"].error is None:
        log(f"thinking never ended within {max_tokens} tokens; retrying at {RETRY_MAX_TOKENS}")
        max_tokens = RETRY_MAX_TOKENS
        reset_capture(cap)
        run = asyncio.run(run_once(engine, runtime.tokenizer, max_tokens))

    if run["decode_state"].error is not None:
        raise RuntimeError(f"generation failed: {run['decode_state'].error}")

    trace_forward, trace_tokenflow = build_traces(runtime, cap, run, device)

    out1 = emit_js("trace_forward.js", "TRACE_FORWARD", trace_forward)
    out2 = emit_js("trace_tokenflow.js", "TRACE_TOKENFLOW", trace_tokenflow)

    # Sanity summary
    tf = trace_forward
    log(
        f"prompt {tf['prompt']['len']} toks | forwards {tf['totals']['forwards']} "
        f"| emitted {tf['totals']['tokens_emitted']} "
        f"| prefill {tf['totals']['prefill_ms']}ms "
        f"| decode avg {tf['totals']['decode_ms_avg']}ms "
        f"({tf['totals']['decode_tok_per_s']} tok/s)"
    )
    fin = trace_tokenflow["finish"]
    log(
        f"finish={fin['finish_reason']} counts={fin['counts']} "
        f"ttft={trace_tokenflow['timing']['ttft_ms']}ms "
        f"total={trace_tokenflow['timing']['total_ms']}ms"
    )
    log(f"thinking: {trace_tokenflow['text']['thinking']!r}")
    log(f"visible:  {trace_tokenflow['text']['visible']!r}")
    ident = tf["detailed"]["prefill"]["modules"]["repeated_layers"]
    log(f"module capture: blocks identical={ident['identical']} x{ident['count']}")
    log(f"wrote {out1}")
    log(f"wrote {out2}")

    assert trace_tokenflow["finish"]["counts"]["text"] > 0, "no visible text produced"
    assert trace_tokenflow["text"]["visible"].strip(), "visible text empty"


if __name__ == "__main__":
    main()

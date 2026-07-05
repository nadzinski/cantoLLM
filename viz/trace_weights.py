"""Weights/parameters trace for the viz Weights tab.

Reads the safetensors HEADERS (8-byte length prefix + JSON — no tensor data,
no torch) of every downloaded Qwen3 checkpoint under models/model_data/, and
mirrors weights.py's HF-name → cantollm-module mapping. Everything in the
output — shapes, dtypes, parameter counts, weight tying, disk sizes, derived
architecture — comes from the real checkpoint files.

Run from the repo root:  .venv/bin/python viz/trace_weights.py   (instant)
Writes:                  viz/data/trace_weights.js  (window.TRACE_WEIGHTS)
"""

import json
import math
import re
import struct
import sys
import time
from pathlib import Path

from trace_common import emit_js

MODEL_DATA = Path(__file__).resolve().parents[1] / "src/cantollm/models/model_data"

DTYPE_BYTES = {"BF16": 2, "F16": 2, "F32": 4, "F64": 8, "I64": 8, "I32": 4}


def read_header(path: Path) -> dict:
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(n))
    header.pop("__metadata__", None)
    return header


def load_tensors(model_dir: Path) -> tuple[dict, int]:
    """Return {name: {dtype, shape}} plus total bytes on disk."""
    single = model_dir / "model.safetensors"
    index = model_dir / "model.safetensors.index.json"
    tensors = {}
    disk = 0
    if single.exists():
        tensors.update(read_header(single))
        disk = single.stat().st_size
    elif index.exists():
        shards = sorted(set(json.load(open(index))["weight_map"].values()))
        for shard in shards:
            p = model_dir / shard
            tensors.update(read_header(p))
            disk += p.stat().st_size
    else:
        return {}, 0
    return tensors, disk


# HF checkpoint name -> (cantollm target, category) — mirrors weights.py
LAYER_PATTERNS = [
    ("self_attn.q_proj.weight", "GQA.W_q.weight", "attention"),
    ("self_attn.k_proj.weight", "GQA.W_k.weight", "attention"),
    ("self_attn.v_proj.weight", "GQA.W_v.weight", "attention"),
    ("self_attn.o_proj.weight", "GQA.out_proj.weight", "attention"),
    ("self_attn.q_norm.weight", "GQA.q_norm.scaling_weight", "norms"),
    ("self_attn.k_norm.weight", "GQA.k_norm.scaling_weight", "norms"),
    ("input_layernorm.weight", "RMSNorm_1.scaling_weight", "norms"),
    ("post_attention_layernorm.weight", "RMSNorm_2.scaling_weight", "norms"),
    ("mlp.gate_proj.weight", "FF.linear_1.weight", "ffn"),
    ("mlp.up_proj.weight", "FF.linear_2.weight", "ffn"),
    ("mlp.down_proj.weight", "FF.linear_3.weight", "ffn"),
]
TOP_PATTERNS = {
    "model.embed_tokens.weight": ("initial_embedding_layer.weight", "embedding"),
    "model.norm.weight": ("output_RMSNorm.scaling_weight", "norms"),
    "lm_head.weight": ("output_layer.weight", "lm_head"),
}

_LAYER_RE = re.compile(r"^model\.layers\.(\d+)\.(.+)$")


def analyze(size: str, model_dir: Path) -> dict | None:
    tensors, disk = load_tensors(model_dir)
    if not tensors:
        return None

    categories = {"embedding": 0, "attention": 0, "ffn": 0, "norms": 0, "lm_head": 0}
    layer_pat_by_suffix = {hf: (canto, cat) for hf, canto, cat in LAYER_PATTERNS}
    unmapped = []
    num_layers = 0
    block_tensors = []  # layer-0 detail table
    per_layer_params = 0
    dtypes = set()

    for name, info in sorted(tensors.items()):
        params = math.prod(info["shape"])
        dtypes.add(info["dtype"])
        m = _LAYER_RE.match(name)
        if m:
            layer, suffix = int(m.group(1)), m.group(2)
            num_layers = max(num_layers, layer + 1)
            hit = layer_pat_by_suffix.get(suffix)
            if hit is None:
                unmapped.append(name)
                continue
            canto, cat = hit
            categories[cat] += params
            if layer == 0:
                per_layer_params += params
                block_tensors.append(
                    {
                        "hf": name,
                        "canto": f"transformer_blocks[i].{canto}",
                        "shape": info["shape"],
                        "params": params,
                    }
                )
        elif name in TOP_PATTERNS:
            canto, cat = TOP_PATTERNS[name]
            categories[cat] += params
        else:
            unmapped.append(name)

    if unmapped:
        # weights.py's mapping would raise on these too — surface loudly
        print(f"WARNING {size}: unmapped tensors: {unmapped[:5]}", file=sys.stderr)

    tied = "lm_head.weight" not in tensors  # weights.py ties output to embedding

    # Architecture derived purely from checkpoint shapes
    embed_shape = tensors["model.embed_tokens.weight"]["shape"]  # [vocab, dim]
    q_shape = tensors["model.layers.0.self_attn.q_proj.weight"]["shape"]  # [heads*hd, dim]
    k_shape = tensors["model.layers.0.self_attn.k_proj.weight"]["shape"]  # [groups*hd, dim]
    head_dim = tensors["model.layers.0.self_attn.q_norm.weight"]["shape"][0]
    gate_shape = tensors["model.layers.0.mlp.gate_proj.weight"]["shape"]  # [expanded, dim]
    arch = {
        "vocab": embed_shape[0],
        "dim": embed_shape[1],
        "layers": num_layers,
        "heads": q_shape[0] // head_dim,
        "groups": k_shape[0] // head_dim,
        "head_dim": head_dim,
        "expanded_dim": gate_shape[0],
    }

    dtype = dtypes.pop() if len(dtypes) == 1 else "mixed"
    bytes_per = DTYPE_BYTES.get(dtype, 2)
    total = sum(categories.values())
    kv_bytes_per_token = 2 * num_layers * arch["groups"] * head_dim * bytes_per
    weight_bytes = total * bytes_per
    return {
        "size": size,
        "total_params": total,
        "disk_bytes": disk,
        "dtype": dtype,
        "tied": tied,
        "categories": categories,
        "per_layer_params": per_layer_params,
        "block_tensors": block_tensors,
        "arch": arch,
        "kv_bytes_per_token": kv_bytes_per_token,
        "kv_crossover_tokens": round(weight_bytes / kv_bytes_per_token),
        "num_tensors": len(tensors),
    }


def main():
    models = {}
    order = []
    for d in sorted(MODEL_DATA.glob("Qwen3-*")):
        size = d.name.replace("Qwen3-", "")
        result = analyze(size, d)
        if result:
            models[size] = result
            order.append(size)
            c = result["categories"]
            print(
                f"{size}: {result['total_params'] / 1e6:.1f}M params, "
                f"{result['num_tensors']} tensors, {result['arch']['layers']} layers, "
                f"tied={result['tied']}, embed {c['embedding'] / result['total_params']:.1%}, "
                f"disk {result['disk_bytes'] / 2**30:.2f} GiB",
                file=sys.stderr,
            )

    order.sort(key=lambda s: float(s.rstrip("B")))
    trace = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source": "safetensors headers under src/cantollm/models/model_data/",
        "order": order,
        "models": models,
    }
    out = emit_js("trace_weights.js", "TRACE_WEIGHTS", trace)
    print(f"wrote {out}", file=sys.stderr)


if __name__ == "__main__":
    main()

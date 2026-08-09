"""Capture real weight statistics for the viz "Quantization" chapter tab.

Like capture_cudagraphs.py, the artifacts are COMMITTED under viz/captures/
as the record the tab's curated inline data is transcribed from; unlike it,
no GPU is needed. It reads tensors straight from the local Qwen3-0.6B
safetensors (no model build, no tokenizer) and runs in seconds on the Mac:

    .venv/bin/python viz/capture_quant.py    -> viz/captures/quant-<date>/

Produces data.json with three exhibits:
  layer      one real projection (layer 0 down_proj): histogram, channel
             spread, outliers, and the quantization error table the
             granularity widget displays (int8/int4 x per-tensor/
             per-channel/per-group-128, RMSE computed here, not in JS)
  block      two real 16-value runs (one typical, one containing the
             tensor's absmax) for the hand-quantized NVFP4 figure
  ranking    absmax / p99.9 per linear of layer 0 (the outlier-hunt
             exercise's expected answer)
"""

import datetime
import json
from pathlib import Path

import torch
from safetensors import safe_open

REPO_ROOT = Path(__file__).resolve().parent.parent
CKPT = REPO_ROOT / "src/cantollm/models/model_data/Qwen3-0.6B/model.safetensors"

LAYER_TENSOR = "model.layers.0.mlp.down_proj.weight"
RANKING_TENSORS = {
    "q_proj": "model.layers.0.self_attn.q_proj.weight",
    "k_proj": "model.layers.0.self_attn.k_proj.weight",
    "v_proj": "model.layers.0.self_attn.v_proj.weight",
    "o_proj": "model.layers.0.self_attn.o_proj.weight",
    "gate_proj": "model.layers.0.mlp.gate_proj.weight",
    "up_proj": "model.layers.0.mlp.up_proj.weight",
    "down_proj": LAYER_TENSOR,
}

HIST_BINS = 121
GROUP = 128


def quantize_rmse(w, bits, granularity):
    """Symmetric absmax round-to-nearest; returns (rmse, codes_used_frac)."""
    qmax = 2 ** (bits - 1) - 1  # 127 for int8, 7 for int4
    if granularity == "tensor":
        s = w.abs().amax() / qmax
        s = s.clamp(min=1e-12)
    elif granularity == "channel":  # one scale per output row
        s = w.abs().amax(dim=1, keepdim=True) / qmax
        s = s.clamp(min=1e-12)
    elif granularity == "group":  # groups of GROUP along the input dim
        rows, cols = w.shape
        assert cols % GROUP == 0, (rows, cols)
        wg = w.reshape(rows, cols // GROUP, GROUP)
        s = (wg.abs().amax(dim=2, keepdim=True) / qmax).clamp(min=1e-12)
        q = (wg / s).round().clamp(-qmax, qmax)
        deq = (q * s).reshape(rows, cols)
        rmse = (deq - w).pow(2).mean().sqrt().item()
        used = q.unique().numel() / (2 * qmax + 1)
        return rmse, used
    q = (w / s).round().clamp(-qmax, qmax)
    deq = q * s
    rmse = (deq - w).pow(2).mean().sqrt().item()
    used = q.unique().numel() / (2 * qmax + 1)
    return rmse, used


def main():
    out_dir = REPO_ROOT / "viz/captures" / f"quant-{datetime.date.today().isoformat()}"
    out_dir.mkdir(parents=True, exist_ok=True)

    with safe_open(CKPT, framework="pt") as f:
        tensors = {name: f.get_tensor(path).to(torch.float32)
                   for name, path in RANKING_TENSORS.items()}

    w = tensors["down_proj"]
    absmax = w.abs().max().item()
    flat = w.flatten()

    # exhibit 1: the layer
    hist = torch.histc(flat, bins=HIST_BINS, min=-absmax, max=absmax)
    row_absmax = w.abs().amax(dim=1)
    qs = torch.tensor([0.0, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0])
    top_idx = flat.abs().argsort(descending=True)[:12]
    error_table = {
        f"int{bits}/{gran}": dict(zip(("rmse", "codes_used"),
                                      quantize_rmse(w, bits, gran)))
        for bits in (8, 4) for gran in ("tensor", "channel", "group")
    }
    layer = {
        "tensor": LAYER_TENSOR,
        "shape": list(w.shape),
        "dtype_on_disk": "bfloat16",
        "absmax": absmax,
        "std": flat.std().item(),
        "p999": flat.abs().quantile(0.999).item(),
        "p9999": flat.abs().quantile(0.9999).item(),
        "hist_bins": HIST_BINS,
        "hist_range": [-absmax, absmax],
        "hist_counts": [int(c) for c in hist],
        "row_absmax_quantiles": {f"q{int(q * 100)}": row_absmax.quantile(q).item() for q in qs},
        "top_values": [{"value": flat[i].item(),
                        "row": int(i // w.shape[1]), "col": int(i % w.shape[1])}
                       for i in top_idx],
        "group_size": GROUP,
        "error_table": error_table,
    }

    # exhibit 2: two real 16-value blocks (bf16-exact values survive the fp32 cast)
    max_flat = flat.abs().argmax().item()
    max_row, max_col = max_flat // w.shape[1], max_flat % w.shape[1]
    ocol = min(max_col - max_col % 16, w.shape[1] - 16)
    block = {
        "typical": {"row": 512, "cols": [0, 16], "values": w[512, :16].tolist()},
        "outlier": {"row": int(max_row), "cols": [ocol, ocol + 16],
                    "values": w[max_row, ocol:ocol + 16].tolist()},
    }

    # exhibit 3: the layer-0 outlier ranking
    ranking = []
    for name, t in tensors.items():
        fl = t.flatten().abs()
        am, p999 = fl.max().item(), fl.quantile(0.999).item()
        ranking.append({"linear": name, "shape": list(t.shape),
                        "absmax": am, "p999": p999, "ratio": am / p999})
    ranking.sort(key=lambda r: -r["ratio"])

    data = {
        "generated": datetime.datetime.now().isoformat(timespec="seconds"),
        "source": str(CKPT.relative_to(REPO_ROOT)),
        "script": "viz/capture_quant.py",
        "layer": layer,
        "block": block,
        "ranking": ranking,
    }
    (out_dir / "data.json").write_text(json.dumps(data, indent=1))

    lines = [f"quantize_rmse {k}: rmse={v['rmse']:.3e} codes_used={v['codes_used']:.3f}"
             for k, v in error_table.items()]
    print(f"wrote {out_dir / 'data.json'}")
    print(f"{LAYER_TENSOR} shape={list(w.shape)} absmax={absmax:.4f} p99.9={layer['p999']:.4f}")
    print("\n".join(lines))
    print("ranking:", " > ".join(f"{r['linear']}({r['ratio']:.1f}x)" for r in ranking))


if __name__ == "__main__":
    main()

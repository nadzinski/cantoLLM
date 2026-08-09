# Quantization capture record: 2026-08-08

Raw artifacts behind the viz "Quantization" tab (the tab embeds a curated
inline copy; this directory is the record it was transcribed from).
Produced by `viz/capture_quant.py` on the Mac, read straight from the
committed local checkpoint's safetensors (no GPU, no model build):
`src/cantollm/models/model_data/Qwen3-0.6B/model.safetensors`, bf16 on
disk, statistics computed in fp32.

One file, `data.json`, three exhibits:

- **layer**: `model.layers.0.mlp.down_proj.weight` (1024 x 3072), the
  chapter's specimen projection. Histogram (121 bins over +-absmax),
  channel-absmax quantiles, the 12 largest-magnitude entries, and the
  quantization error table the granularity widget displays: RMSE and
  codes-used for int8/int4 x per-tensor/per-channel/per-group-128
  (symmetric absmax, round-to-nearest, group size 128 along the input
  dim). Computed here so the page shows real math, not a JS toy.
- **block**: two real 16-value runs from the same tensor, one typical
  (row 512, cols 0-15) and one containing the tensor's absmax, for the
  hand-quantized NVFP4 figure.
- **ranking**: absmax / p99.9 for all seven layer-0 linears, sorted.
  This is also the expected answer to the chapter's outlier-hunt
  exercise; spoiler: down_proj wins at 4.4x, v_proj is tamest at 1.8x.

## Headline numbers

| scheme | RMSE | vs per-tensor |
|---|---:|---:|
| int8 per-tensor | 9.50e-4 | 1.0x |
| int8 per-channel | 2.55e-4 | 3.7x better |
| int8 group-128 | 1.79e-4 | 5.3x better |
| int4 per-tensor | 1.68e-2 | 1.0x |
| int4 per-channel | 4.63e-3 | 3.6x better |
| int4 group-128 | 3.25e-3 | 5.2x better |

absmax 0.418, p99.9 0.0947: one weight 4.4x beyond the 99.9th percentile
sets the per-tensor grid for all 3.1M.

## Regenerating

```
.venv/bin/python viz/capture_quant.py
```

Deterministic (no sampling, no RNG): re-runs reproduce byte-identical
statistics for the same checkpoint.

"""Weight-only FP8 for the Qwen 3.8 PoC.

The official FP8 checkpoint (Qwen/Qwen3.8-27B-FP8) stores each quantized
matrix as float8-e4m3 values plus a fp32 `weight_scale_inv` of one scale
per 128x128 block (dequant multiplies by it). This module keeps exactly
that at rest and dequantizes to the activation dtype on the fly inside
each forward: memory stays ~1 byte/param resident, the transient bf16
copy exists only for the duration of one matmul (largest is down_proj,
~178MB at 27B geometry).

Deliberately NOT here (PoC decision): torch._scaled_mm or any real fp8
GEMM; block-scaled fp8 matmuls need cutlass-style kernels. Every step
therefore re-reads and expands the fp8 weights, which is the accepted
single-digit-tok/s cost of the bring-up.
"""

import torch
import torch.nn.functional as F
from torch import nn


class FP8Linear(nn.Module):
    """Drop-in replacement for a bias-free nn.Linear whose checkpoint
    tensor arrived in float8-e4m3 with block scales."""

    def __init__(
        self,
        weight: torch.Tensor,
        scale_inv: torch.Tensor,
        block_size: tuple[int, int] = (128, 128),
    ):
        super().__init__()
        if weight.dtype != torch.float8_e4m3fn:
            raise ValueError(f"expected float8_e4m3fn weight, got {weight.dtype}")
        out_features, in_features = weight.shape
        expect = (
            -(-out_features // block_size[0]),  # ceil div
            -(-in_features // block_size[1]),
        )
        if tuple(scale_inv.shape) != expect:
            raise ValueError(
                f"scale_inv shape {tuple(scale_inv.shape)} does not match "
                f"{tuple(weight.shape)} at block size {block_size} "
                f"(expected {expect})"
            )
        self.out_features = out_features
        self.in_features = in_features
        self.block_size = block_size
        # Buffers, not Parameters: they move with .to(device) and stay out
        # of optimizers/grads; state_dict round-trips them.
        self.weight = nn.Buffer(weight)
        self.weight_scale_inv = nn.Buffer(scale_inv.to(torch.float32))

    def dequantized(self, dtype: torch.dtype) -> torch.Tensor:
        """Blockwise dequant: expand each scale over its 128x128 tile
        (cropped at the ragged edges), multiply in fp32, cast."""
        scale = self.weight_scale_inv.repeat_interleave(self.block_size[0], dim=0)[
            : self.out_features
        ].repeat_interleave(self.block_size[1], dim=1)[:, : self.in_features]
        return (self.weight.to(torch.float32) * scale).to(dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.dequantized(x.dtype))

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"block_size={self.block_size}, fp8"
        )

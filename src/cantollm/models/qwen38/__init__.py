"""Qwen 3.8 (qwen3_5 architecture): hybrid Gated-DeltaNet + gated-attention
decoder, ported as a PoC alongside the qwen3 package.

Model code here is deliberately self-contained (copying over sharing with
models/qwen3 was an explicit decision for the PoC); only genuinely
model-agnostic infrastructure is imported from the shared modules
(attention methods, the complex-multiply RoPE helpers, the sampler).
"""

"""
Temporary script to verify that our GroupedQueryAttention produces the same
output as the reference implementation.
"""

import torch
import torch.nn as nn

# Our implementation
from qwen3.model import GroupedQueryAttention as OurGQA
from qwen3.rope import precompute_freqs_cis


# Reference implementation
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x / rms) * self.weight


def apply_rope(x, cos, sin):
    """Reference RoPE: x shape is (batch, n_heads, seq_len, head_dim)"""
    batch, n_heads, seq_len, head_dim = x.shape

    x1 = x[..., : head_dim // 2]
    x2 = x[..., head_dim // 2 :]

    cos = cos[:seq_len].view(1, 1, seq_len, head_dim // 2)
    sin = sin[:seq_len].view(1, 1, seq_len, head_dim // 2)

    rotated = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)

    return rotated


class ReferenceGQA(nn.Module):
    def __init__(self, d_in, num_heads, num_kv_groups, head_dim=None, qk_norm=False, dtype=None):
        super().__init__()
        assert num_heads % num_kv_groups == 0

        self.num_heads = num_heads
        self.num_kv_groups = num_kv_groups
        self.group_size = num_heads // num_kv_groups

        if head_dim is None:
            assert d_in % num_heads == 0
            head_dim = d_in // num_heads

        self.head_dim = head_dim
        self.d_out = num_heads * head_dim

        self.W_query = nn.Linear(d_in, self.d_out, bias=False, dtype=dtype)
        self.W_key = nn.Linear(d_in, num_kv_groups * head_dim, bias=False, dtype=dtype)
        self.W_value = nn.Linear(d_in, num_kv_groups * head_dim, bias=False, dtype=dtype)

        self.out_proj = nn.Linear(self.d_out, d_in, bias=False, dtype=dtype)

        if qk_norm:
            self.q_norm = RMSNorm(head_dim, eps=1e-6)
            self.k_norm = RMSNorm(head_dim, eps=1e-6)
        else:
            self.q_norm = self.k_norm = None

    def forward(self, x, mask, cos, sin):
        b, num_tokens, _ = x.shape

        queries = self.W_query(x)
        keys = self.W_key(x)
        values = self.W_value(x)

        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        keys = keys.view(b, num_tokens, self.num_kv_groups, self.head_dim).transpose(1, 2)
        values = values.view(b, num_tokens, self.num_kv_groups, self.head_dim).transpose(1, 2)

        if self.q_norm:
            queries = self.q_norm(queries)
        if self.k_norm:
            keys = self.k_norm(keys)

        queries = apply_rope(queries, cos, sin)
        keys = apply_rope(keys, cos, sin)

        keys = keys.repeat_interleave(self.group_size, dim=1)
        values = values.repeat_interleave(self.group_size, dim=1)

        attn_scores = queries @ keys.transpose(2, 3)
        attn_scores = attn_scores.masked_fill(mask, -torch.inf)
        attn_weights = torch.softmax(attn_scores / self.head_dim**0.5, dim=-1)

        context = (attn_weights @ values).transpose(1, 2).reshape(b, num_tokens, self.d_out)
        return self.out_proj(context)


def compute_cos_sin(head_dim, max_seq_len, theta=100000.0):
    """Compute cos and sin for reference implementation."""
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2)[: (head_dim // 2)].float() / head_dim))
    t = torch.arange(max_seq_len, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    cos = torch.cos(freqs)
    sin = torch.sin(freqs)
    return cos, sin


def copy_weights(ref_gqa, our_gqa):
    """Copy weights from reference to our implementation."""
    our_gqa.W_q.weight.data = ref_gqa.W_query.weight.data.clone()
    our_gqa.W_k.weight.data = ref_gqa.W_key.weight.data.clone()
    our_gqa.W_v.weight.data = ref_gqa.W_value.weight.data.clone()
    our_gqa.out_proj.weight.data = ref_gqa.out_proj.weight.data.clone()

    # Copy norm weights
    our_gqa.q_norm.scaling_weight.data = ref_gqa.q_norm.weight.data.clone()
    our_gqa.k_norm.scaling_weight.data = ref_gqa.k_norm.weight.data.clone()


def test_equivalence():
    torch.manual_seed(42)

    # Config
    batch_size = 2
    seq_len = 8
    d_in = 64
    num_heads = 8
    num_kv_groups = 4
    max_seq_len = 128

    # Create both models
    ref_gqa = ReferenceGQA(
        d_in=d_in,
        num_heads=num_heads,
        num_kv_groups=num_kv_groups,
        qk_norm=True,
    )

    our_gqa = OurGQA(
        token_embedding_dim=d_in,
        num_heads=num_heads,
        num_groups=num_kv_groups,
        head_dim=head_dim,
    )

    # Copy weights
    copy_weights(ref_gqa, our_gqa)

    ref_gqa.eval()
    our_gqa.eval()

    # Input
    x = torch.randn(batch_size, seq_len, d_in)

    # Prepare inputs for reference implementation
    head_dim = d_in // num_heads
    cos, sin = compute_cos_sin(head_dim, max_seq_len)
    ref_mask = torch.ones(seq_len, seq_len).tril() == 0
    ref_mask = ref_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq, seq) for broadcasting
    our_mask = ref_mask.squeeze(0).squeeze(0)
    freqs_cis = precompute_freqs_cis(head_dim, max_seq_len)

    with torch.no_grad():
        ref_output = ref_gqa(x, ref_mask, cos, sin)
        our_output = our_gqa(
            x,
            start_pos=0,
            mask=our_mask,
            freqs_cis=freqs_cis,
            kv_cache=None,
        )

    # Compare
    max_diff = (ref_output - our_output).abs().max().item()
    mean_diff = (ref_output - our_output).abs().mean().item()

    print(f"Max absolute difference: {max_diff:.2e}")
    print(f"Mean absolute difference: {mean_diff:.2e}")

    if torch.allclose(ref_output, our_output, atol=1e-5):
        print("✓ Outputs match!")
        return True
    else:
        print("✗ Outputs differ!")
        print(f"Reference output sample:\n{ref_output[0, 0, :8]}")
        print(f"Our output sample:\n{our_output[0, 0, :8]}")
        return False


if __name__ == "__main__":
    test_equivalence()

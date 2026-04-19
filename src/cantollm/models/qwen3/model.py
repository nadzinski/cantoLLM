import torch
from torch import nn

from cantollm.models.attention import AttentionMethod
from cantollm.models.rope import apply_rotary_emb, precompute_freqs_cis


class RootMeanSquareNorm(nn.Module):
    """
    This is used twice in each transformer block to prevent nasty vanishing or
    exploding gradients during training. We scale to "reasonable" values by
    dividing by the root mean square.

    The input has dimension "token_embedding_dim", because this is going to sit
    at 1) the start of each transformer block and 2) before the feedforward layer.

    It operates on each embedding individually, of course.

    It's similar to the older mean-centering "LayerNorm" but computationally
    more efficient.

    epsilon is a small param to prevent division by zero.
    self.scaling_weight is a learnable vector of scaling params.
    """

    def __init__(self, token_embedding_dim, epsilon=1e-6):
        super().__init__()
        self.epsilon = epsilon
        self.scaling_weight = nn.Parameter(torch.ones(token_embedding_dim))

    def _take_rmsnorm(self, x):
        var = x.pow(2).mean(dim=-1, keepdim=True)
        rms = torch.rsqrt(var + self.epsilon)
        return x * rms

    def forward(self, x):
        # We temporarily cast up to f32 for the rms because squaring can cause
        # overflows in lower float sizes
        x_normed = self._take_rmsnorm(x.to(torch.float32)).type_as(x)
        return x_normed * self.scaling_weight.to(dtype=x.dtype)


class FeedForward(nn.Module):
    """
    This is the neural network part where we think facts get learned!

    Remember: it operates on each embedding vector individually. (The
    cross-vector mixing happens inside the attention heads)

    It has both a Linear->Nonlinear Activation Function->Linear
    path ("gate branch") *and* and Linear->Linear->Linear path
    ("value branch") like so:

                |
           [ Linear 3 ]
                |
               (X) product
              /   \\
      [  SiLU  ]   |
          |        |
    [ Linear 1 ]  [ Linear 2 ]
      (Gate)        (Value)
         \\____________/
                |
              Input

    SiLU is the now-standard "swish" (sigmoid linear unit), a smoothed version
    of RLU that is still fairly low effort to compute.

    The first two linear layers expand the dimension upwards from the normal
    token embedding dimension, and the third one projects it down again.
    """

    def __init__(self, token_embedding_dim, expanded_dim, dtype):
        super().__init__()
        self.linear_1 = nn.Linear(token_embedding_dim, expanded_dim, bias=False, dtype=dtype)
        self.linear_2 = nn.Linear(token_embedding_dim, expanded_dim, bias=False, dtype=dtype)
        self.linear_3 = nn.Linear(expanded_dim, token_embedding_dim, bias=False, dtype=dtype)

    def forward(self, x):
        gate = self.linear_1(x)
        value = self.linear_2(x)
        prod = nn.functional.silu(gate) * value
        return self.linear_3(prod)


class GroupedQueryAttention(nn.Module):
    """
    In multi-head attention, each head projects our embedding vectors down into a
    subspace. Example: embedding dim 786, 12 heads, each subspace has dim 64.
    We then combine the output to get back to our embedding dimension.

    For performance we just add new dimensions to the tensors with view()
    so we can do the whole thing as one big set of matmuls.

    GQA is similar to multi-head attention except that it
    1) divides the heads into groups
    2) groups of heads share K and V (but not Q, which is still unique to each head)

    Example: embedding dim 786, 12 heads, 6 groups => group size 2 heads.

    We also need to do KV caching during inference to avoid N^2 penalty in
    sequence length.

    Note: head_dim can be set explicitly rather than derived from
    token_embedding_dim // num_heads. This means the Q projection output
    dimension (num_heads * head_dim) may differ from the input embedding dimension.
    """

    def __init__(
        self,
        token_embedding_dim,
        num_heads,
        num_groups,
        head_dim,
        attention_method: AttentionMethod,
        dtype=None,
    ):
        super().__init__()
        assert num_heads % num_groups == 0
        self.token_embedding_dim = token_embedding_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_groups = num_groups
        self.heads_per_group = num_heads // num_groups
        self.kv_dim = self.num_groups * self.head_dim
        self.q_out_dim = self.num_heads * self.head_dim

        # These contain the separate weight matrices for the different
        # heads / groups, glued together so we can do W(x) in a single operation
        self.W_q = nn.Linear(token_embedding_dim, self.q_out_dim, bias=False, dtype=dtype)
        self.W_k = nn.Linear(token_embedding_dim, self.kv_dim, bias=False, dtype=dtype)
        self.W_v = nn.Linear(token_embedding_dim, self.kv_dim, bias=False, dtype=dtype)
        self.out_proj = nn.Linear(self.q_out_dim, token_embedding_dim, bias=False, dtype=dtype)

        self.q_norm = RootMeanSquareNorm(self.head_dim)
        self.k_norm = RootMeanSquareNorm(self.head_dim)

        self.attention_method = attention_method

    def forward(self, x, start_pos, mask, freqs_cis, kv_cache=None):
        batches, seq_len, _ = x.shape

        # We can apply the weights across all heads/groups all in one go
        queries_flat = self.W_q(x)
        keys_flat = self.W_k(x)
        values_flat = self.W_v(x)

        # At this point, we start operating separately on each head.
        # We use view() to break out by group and for Q by head number within the group.
        queries = queries_flat.view(
            batches, seq_len, self.num_groups, self.heads_per_group, self.head_dim
        )
        keys = keys_flat.view(batches, seq_len, self.num_groups, self.head_dim)
        values = values_flat.view(batches, seq_len, self.num_groups, self.head_dim)

        # QK Norm, a Qwen3-specific thing to help training stability
        queries_normed = self.q_norm(queries)
        keys_normed = self.k_norm(keys)

        # Apply rotary positional embedding (RoPE) to put in information about
        # relative positions of tokens in the sequence.
        queries_roped = apply_rotary_emb(queries_normed, freqs_cis, offset=start_pos)
        keys_roped = apply_rotary_emb(keys_normed, freqs_cis, offset=start_pos)

        # Delegate attention math + KV update to the attention method.
        # When the cache is populated we're in the decode path (possibly a
        # multi-token speculative chunk); otherwise it's prefill.
        if kv_cache is None or kv_cache["keys"] is None:
            z_context = self.attention_method.forward_prefill(
                queries_roped, keys_roped, values, mask, kv_cache,
            )
        else:
            z_context = self.attention_method.forward_decode(
                queries_roped, keys_roped, values, mask, kv_cache,
            )

        # Stitch heads together
        z_context_flat = z_context.reshape(batches, seq_len, self.q_out_dim)

        # Final projection
        output = self.out_proj(z_context_flat)

        return output


class Transformer(nn.Module):
    def __init__(
        self,
        token_embedding_dim,
        expanded_dim,
        num_heads,
        num_groups,
        head_dim,
        attention_method: AttentionMethod,
        dtype=None,
    ):
        super().__init__()
        self.RMSNorm_1 = RootMeanSquareNorm(token_embedding_dim)
        self.RMSNorm_2 = RootMeanSquareNorm(token_embedding_dim)

        self.GQA = GroupedQueryAttention(
            token_embedding_dim, num_heads, num_groups, head_dim, attention_method, dtype
        )

        self.FF = FeedForward(token_embedding_dim, expanded_dim, dtype)

    def forward(self, x, start_pos, mask, freqs_cis, kv_cache=None):
        gqa_bypass = x
        x = self.RMSNorm_1(x)
        x = self.GQA(x, start_pos, mask, freqs_cis, kv_cache)
        x = x + gqa_bypass

        ff_bypass = x
        x = self.RMSNorm_2(x)
        x = self.FF(x)
        x = x + ff_bypass

        return x


class Qwen3(nn.Module):
    def __init__(self, qwen3_config, attention_method: AttentionMethod):
        super().__init__()

        self.attention_method = attention_method

        self.initial_embedding_layer = nn.Embedding(
            qwen3_config["token_count"],
            qwen3_config["token_embedding_dim"],
            dtype=qwen3_config["dtype"],
        )

        self.transformer_blocks = nn.ModuleList(
            [
                Transformer(
                    qwen3_config["token_embedding_dim"],
                    qwen3_config["expanded_dim"],
                    qwen3_config["num_heads"],
                    qwen3_config["num_groups"],
                    qwen3_config["head_dim"],
                    attention_method,
                    dtype=qwen3_config["dtype"],
                )
                for _ in range(qwen3_config["num_transformers"])
            ]
        )

        self.output_RMSNorm = RootMeanSquareNorm(qwen3_config["token_embedding_dim"])
        self.output_layer = nn.Linear(
            qwen3_config["token_embedding_dim"],
            qwen3_config["token_count"],
            bias=False,
            dtype=qwen3_config["dtype"],
        )

        max_seq_len = qwen3_config["max_seq_len"]
        head_dim = qwen3_config["head_dim"]
        freqs_cis = precompute_freqs_cis(head_dim, max_seq_len)
        self.register_buffer("freqs_cis", freqs_cis)

    def _validate_cache(self, start_pos, kv_cache):
        if kv_cache is not None:
            num_blocks = len(self.transformer_blocks)
            if len(kv_cache) != num_blocks:
                raise ValueError(
                    f"kv_cache has {len(kv_cache)} entries but model has {num_blocks} blocks"
                )
        if start_pos == 0:
            return
        if kv_cache is None:
            raise ValueError(f"start_pos={start_pos} but no kv_cache provided")
        for i, cache in enumerate(kv_cache):
            if cache["keys"] is None:
                raise ValueError(f"start_pos={start_pos} but kv_cache[{i}] is empty")
            cached_len = cache["keys"].shape[1]
            if cached_len != start_pos:
                raise ValueError(
                    f"start_pos={start_pos} but kv_cache[{i}] has {cached_len} positions"
                )

    def forward(self, tokens, start_pos: int, kv_cache=None):
        # We assume we're passed only the tokens we want to process
        self._validate_cache(start_pos, kv_cache)

        x = self.initial_embedding_layer(tokens)

        seq_len = tokens.shape[-1]
        mask = self.attention_method.build_mask(start_pos, seq_len, tokens.device)

        for i, transformer in enumerate(self.transformer_blocks):
            layer_cache = kv_cache[i] if kv_cache is not None else None
            x = transformer(x, start_pos, mask, self.freqs_cis, kv_cache=layer_cache)

        x = self.output_RMSNorm(x)
        output = self.output_layer(x)

        return output

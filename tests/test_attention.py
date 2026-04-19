import torch

from cantollm.models.qwen3.model import GroupedQueryAttention
from cantollm.models.rope import precompute_freqs_cis

MAX_SEQ_LEN = 128


def _causal_mask(start_pos: int, seq_len: int) -> torch.Tensor:
    """Build the pre-sliced causal mask GQA expects: shape [seq_len, start_pos+seq_len]."""
    full_seq_len = start_pos + seq_len
    return torch.ones(seq_len, full_seq_len, dtype=torch.bool).triu(diagonal=start_pos + 1)


def _make_gqa(embedding_dim, num_heads, num_groups, head_dim=None):
    """Create a GQA module and freqs for testing. Mask is built per-call via `_causal_mask`."""
    if head_dim is None:
        head_dim = embedding_dim // num_heads
    gqa = GroupedQueryAttention(
        token_embedding_dim=embedding_dim,
        num_heads=num_heads,
        num_groups=num_groups,
        head_dim=head_dim,
    )
    freqs_cis = precompute_freqs_cis(head_dim, MAX_SEQ_LEN)
    return gqa, freqs_cis


def test_gqa_output_shape():
    """Test that GQA returns correct output shape."""
    batch_size = 2
    seq_len = 8
    embedding_dim = 64
    num_heads = 8
    num_groups = 4

    gqa, freqs_cis = _make_gqa(embedding_dim, num_heads, num_groups)

    x = torch.randn(batch_size, seq_len, embedding_dim)
    output = gqa(x, start_pos=0, mask=_causal_mask(0, seq_len), freqs_cis=freqs_cis)

    assert output.shape == (batch_size, seq_len, embedding_dim)


def test_gqa_different_configurations():
    """Test GQA works with different head/group configurations."""
    batch_size = 1
    seq_len = 4
    embedding_dim = 32
    mask = _causal_mask(0, seq_len)

    # Test 1: 8 heads, 4 groups (2 heads per group)
    gqa1, freqs_cis = _make_gqa(embedding_dim, num_heads=8, num_groups=4)
    x = torch.randn(batch_size, seq_len, embedding_dim)
    out1 = gqa1(x, start_pos=0, mask=mask, freqs_cis=freqs_cis)
    assert out1.shape == x.shape

    # Test 2: 4 heads, 1 group (MQA - multi-query attention)
    gqa2, freqs_cis = _make_gqa(embedding_dim, num_heads=4, num_groups=1)
    out2 = gqa2(x, start_pos=0, mask=mask, freqs_cis=freqs_cis)
    assert out2.shape == x.shape

    # Test 3: 4 heads, 4 groups (standard multi-head attention)
    gqa3, freqs_cis = _make_gqa(embedding_dim, num_heads=4, num_groups=4)
    out3 = gqa3(x, start_pos=0, mask=mask, freqs_cis=freqs_cis)
    assert out3.shape == x.shape


def test_gqa_causal_masking():
    """Test that GQA implements causal masking (future tokens don't affect past)."""
    batch_size = 1
    seq_len = 4
    embedding_dim = 16
    num_heads = 4
    num_groups = 2

    gqa, freqs_cis = _make_gqa(embedding_dim, num_heads, num_groups)
    gqa.eval()
    mask = _causal_mask(0, seq_len)

    # Create input where each position has a unique pattern
    x = torch.zeros(batch_size, seq_len, embedding_dim)
    for i in range(seq_len):
        x[0, i, :] = i + 1

    with torch.no_grad():
        output = gqa(x, start_pos=0, mask=mask, freqs_cis=freqs_cis)

        # For causal attention, changing future tokens shouldn't affect past outputs
        x_modified = x.clone()
        x_modified[0, -1, :] = 999

        output_modified = gqa(x_modified, start_pos=0, mask=mask, freqs_cis=freqs_cis)

    # All positions except the last should be unchanged
    assert torch.allclose(output[0, :-1, :], output_modified[0, :-1, :], atol=1e-5)


def test_gqa_with_kv_cache():
    """Test that GQA can be created with KV cache enabled."""
    embedding_dim = 32
    num_heads = 4
    num_groups = 2

    gqa, freqs_cis = _make_gqa(embedding_dim, num_heads, num_groups)

    x = torch.randn(1, 4, embedding_dim)
    kv_cache = {"keys": None, "values": None}
    output = gqa(x, start_pos=0, mask=_causal_mask(0, 4), freqs_cis=freqs_cis, kv_cache=kv_cache)

    assert output.shape == x.shape
    assert kv_cache["keys"] is not None
    assert kv_cache["values"] is not None


def test_gqa_learnable_parameters():
    """Test that GQA has learnable parameters."""
    gqa, _ = _make_gqa(embedding_dim=32, num_heads=4, num_groups=2)

    assert gqa.W_q.weight.requires_grad
    assert gqa.W_k.weight.requires_grad
    assert gqa.W_v.weight.requires_grad
    assert gqa.out_proj.weight.requires_grad

    params = list(gqa.parameters())
    assert len(params) > 0


def test_kv_cache_incremental_matches_full():
    """Test that incremental decoding with KV cache matches full sequence output."""
    batch_size = 1
    embedding_dim = 32
    num_heads = 4
    num_groups = 2

    gqa, freqs_cis = _make_gqa(embedding_dim, num_heads, num_groups)
    gqa.eval()

    # Create a sequence of 4 tokens
    x_full = torch.randn(batch_size, 4, embedding_dim)

    with torch.no_grad():
        # Run full sequence without cache
        output_full = gqa(x_full, start_pos=0, mask=_causal_mask(0, 4), freqs_cis=freqs_cis, kv_cache=None)

        # Run incrementally with cache
        kv_cache = {"keys": None, "values": None}

        # First: process tokens 0, 1, 2 (the "prompt")
        output_prompt = gqa(x_full[:, :3, :], start_pos=0, mask=_causal_mask(0, 3), freqs_cis=freqs_cis, kv_cache=kv_cache)

        # Then: process just token 3
        output_last = gqa(x_full[:, 3:, :], start_pos=3, mask=_causal_mask(3, 1), freqs_cis=freqs_cis, kv_cache=kv_cache)

    # The last token's output should match between full and incremental
    assert torch.allclose(output_full[:, 3, :], output_last[:, 0, :], atol=1e-5)

    # The prompt outputs should also match
    assert torch.allclose(output_full[:, :3, :], output_prompt, atol=1e-5)


def test_kv_cache_output_shape_incremental():
    """Test that incremental generation returns correct shapes."""
    batch_size = 1
    embedding_dim = 32
    num_heads = 4
    num_groups = 2

    gqa, freqs_cis = _make_gqa(embedding_dim, num_heads, num_groups)
    gqa.eval()

    kv_cache = {"keys": None, "values": None}

    with torch.no_grad():
        # First pass: prompt of 3 tokens
        prompt = torch.randn(batch_size, 3, embedding_dim)
        output1 = gqa(prompt, start_pos=0, mask=_causal_mask(0, 3), freqs_cis=freqs_cis, kv_cache=kv_cache)
        assert output1.shape == (batch_size, 3, embedding_dim)

        # Second pass: just 1 new token
        new_token = torch.randn(batch_size, 1, embedding_dim)
        output2 = gqa(new_token, start_pos=3, mask=_causal_mask(3, 1), freqs_cis=freqs_cis, kv_cache=kv_cache)
        assert output2.shape == (batch_size, 1, embedding_dim)


def test_kv_cache_fresh_cache_matches_no_cache():
    """Test that a fresh empty cache gives same result as no cache."""
    batch_size = 1
    embedding_dim = 32
    num_heads = 4
    num_groups = 2

    gqa, freqs_cis = _make_gqa(embedding_dim, num_heads, num_groups)
    gqa.eval()
    mask = _causal_mask(0, 4)

    x = torch.randn(batch_size, 4, embedding_dim)

    with torch.no_grad():
        output_no_cache = gqa(x, start_pos=0, mask=mask, freqs_cis=freqs_cis, kv_cache=None)

        fresh_cache = {"keys": None, "values": None}
        output_fresh_cache = gqa(x, start_pos=0, mask=mask, freqs_cis=freqs_cis, kv_cache=fresh_cache)

    assert torch.allclose(output_no_cache, output_fresh_cache, atol=1e-5)


def test_kv_cache_multiple_incremental_steps():
    """Test that KV cache works across multiple incremental generation steps."""
    batch_size = 1
    embedding_dim = 32
    num_heads = 4
    num_groups = 2

    gqa, freqs_cis = _make_gqa(embedding_dim, num_heads, num_groups)
    gqa.eval()

    # Create a sequence of 4 tokens
    x_full = torch.randn(batch_size, 4, embedding_dim)

    with torch.no_grad():
        # Run full sequence without cache
        output_full = gqa(x_full, start_pos=0, mask=_causal_mask(0, 4), freqs_cis=freqs_cis, kv_cache=None)

        # Run incrementally: one token at a time
        kv_cache = {"keys": None, "values": None}

        _ = gqa(x_full[:, :1, :], start_pos=0, mask=_causal_mask(0, 1), freqs_cis=freqs_cis, kv_cache=kv_cache)
        _ = gqa(x_full[:, 1:2, :], start_pos=1, mask=_causal_mask(1, 1), freqs_cis=freqs_cis, kv_cache=kv_cache)
        _ = gqa(x_full[:, 2:3, :], start_pos=2, mask=_causal_mask(2, 1), freqs_cis=freqs_cis, kv_cache=kv_cache)
        output_last = gqa(x_full[:, 3:, :], start_pos=3, mask=_causal_mask(3, 1), freqs_cis=freqs_cis, kv_cache=kv_cache)

    assert torch.allclose(output_full[:, 3, :], output_last[:, 0, :], atol=1e-5)


def test_kv_cache_chunk_of_new_tokens():
    """Test processing multiple new tokens at once with KV cache (for speculative decoding)."""
    batch_size = 1
    embedding_dim = 32
    num_heads = 4
    num_groups = 2

    gqa, freqs_cis = _make_gqa(embedding_dim, num_heads, num_groups)
    gqa.eval()

    # Create a sequence of 6 tokens
    x_full = torch.randn(batch_size, 6, embedding_dim)

    with torch.no_grad():
        output_full = gqa(x_full, start_pos=0, mask=_causal_mask(0, 6), freqs_cis=freqs_cis, kv_cache=None)

        kv_cache = {"keys": None, "values": None}

        # Process first 3 tokens
        output_prompt = gqa(x_full[:, :3, :], start_pos=0, mask=_causal_mask(0, 3), freqs_cis=freqs_cis, kv_cache=kv_cache)

        # Process last 3 tokens as a chunk
        output_chunk = gqa(x_full[:, 3:, :], start_pos=3, mask=_causal_mask(3, 3), freqs_cis=freqs_cis, kv_cache=kv_cache)

    assert torch.allclose(output_full[:, :3, :], output_prompt, atol=1e-5)
    assert output_chunk.shape == (batch_size, 3, embedding_dim)
    assert torch.allclose(output_full[:, 3:, :], output_chunk, atol=1e-5)


def test_kv_cache_single_token_prompt():
    """Test KV cache with a single token prompt, then generate more."""
    batch_size = 1
    embedding_dim = 32
    num_heads = 4
    num_groups = 2

    gqa, freqs_cis = _make_gqa(embedding_dim, num_heads, num_groups)
    gqa.eval()

    x_full = torch.randn(batch_size, 4, embedding_dim)

    with torch.no_grad():
        output_full = gqa(x_full, start_pos=0, mask=_causal_mask(0, 4), freqs_cis=freqs_cis, kv_cache=None)

        kv_cache = {"keys": None, "values": None}

        output_0 = gqa(x_full[:, :1, :], start_pos=0, mask=_causal_mask(0, 1), freqs_cis=freqs_cis, kv_cache=kv_cache)
        output_1 = gqa(x_full[:, 1:2, :], start_pos=1, mask=_causal_mask(1, 1), freqs_cis=freqs_cis, kv_cache=kv_cache)
        output_2 = gqa(x_full[:, 2:3, :], start_pos=2, mask=_causal_mask(2, 1), freqs_cis=freqs_cis, kv_cache=kv_cache)
        output_3 = gqa(x_full[:, 3:, :], start_pos=3, mask=_causal_mask(3, 1), freqs_cis=freqs_cis, kv_cache=kv_cache)

    assert torch.allclose(output_full[:, 0, :], output_0[:, 0, :], atol=1e-5)
    assert torch.allclose(output_full[:, 1, :], output_1[:, 0, :], atol=1e-5)
    assert torch.allclose(output_full[:, 2, :], output_2[:, 0, :], atol=1e-5)
    assert torch.allclose(output_full[:, 3, :], output_3[:, 0, :], atol=1e-5)


def test_gqa_multiple_batches():
    """Test that GQA correctly handles multiple sequences in a batch."""
    batch_size = 4
    seq_len = 8
    embedding_dim = 32
    num_heads = 4
    num_groups = 2

    gqa, freqs_cis = _make_gqa(embedding_dim, num_heads, num_groups)
    gqa.eval()
    mask = _causal_mask(0, seq_len)

    x = torch.randn(batch_size, seq_len, embedding_dim)

    with torch.no_grad():
        output = gqa(x, start_pos=0, mask=mask, freqs_cis=freqs_cis)

    assert output.shape == (batch_size, seq_len, embedding_dim)

    with torch.no_grad():
        for i in range(batch_size):
            single_output = gqa(x[i : i + 1, :, :], start_pos=0, mask=mask, freqs_cis=freqs_cis)
            assert torch.allclose(output[i, :, :], single_output[0, :, :], atol=1e-5)


def test_kv_cache_multiple_batches():
    """Test that KV cache works correctly with multiple sequences in a batch."""
    batch_size = 3
    embedding_dim = 32
    num_heads = 4
    num_groups = 2

    gqa, freqs_cis = _make_gqa(embedding_dim, num_heads, num_groups)
    gqa.eval()

    x_full = torch.randn(batch_size, 6, embedding_dim)

    with torch.no_grad():
        output_full = gqa(x_full, start_pos=0, mask=_causal_mask(0, 6), freqs_cis=freqs_cis, kv_cache=None)

        kv_cache = {"keys": None, "values": None}

        output_prompt = gqa(x_full[:, :3, :], start_pos=0, mask=_causal_mask(0, 3), freqs_cis=freqs_cis, kv_cache=kv_cache)
        output_rest = gqa(x_full[:, 3:, :], start_pos=3, mask=_causal_mask(3, 3), freqs_cis=freqs_cis, kv_cache=kv_cache)

    assert torch.allclose(output_full[:, :3, :], output_prompt, atol=1e-5)
    assert output_rest.shape == (batch_size, 3, embedding_dim)
    assert torch.allclose(output_full[:, 3:, :], output_rest, atol=1e-5)

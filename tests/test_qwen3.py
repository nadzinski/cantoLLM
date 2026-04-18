import torch

from cantollm.models.qwen3.model import Qwen3


def make_config():
    return {
        "token_count": 1000,
        "token_embedding_dim": 64,
        "expanded_dim": 128,
        "num_heads": 8,
        "num_groups": 4,
        "head_dim": 8,
        "max_seq_len": 128,
        "num_transformers": 2,
        "dtype": None,
    }


def test_qwen3_output_shape():
    """Test that Qwen3 returns correct output shape."""
    config = make_config()
    model = Qwen3(config)
    model.eval()

    batch_size = 2
    seq_len = 16
    tokens = torch.randint(0, config["token_count"], (batch_size, seq_len))

    kv_cache = [{"keys": None, "values": None} for _ in range(config["num_transformers"])]

    with torch.no_grad():
        output = model(tokens, start_pos=0, kv_cache=kv_cache)

    assert output.shape == (batch_size, seq_len, config["token_count"])


def test_qwen3_with_kv_cache():
    """Test that Qwen3 populates KV cache correctly."""
    config = make_config()
    model = Qwen3(config)
    model.eval()

    batch_size = 1
    prompt_len = 8
    tokens = torch.randint(0, config["token_count"], (batch_size, prompt_len))

    kv_cache = [{"keys": None, "values": None} for _ in range(config["num_transformers"])]

    with torch.no_grad():
        output = model(tokens, start_pos=0, kv_cache=kv_cache)
        assert output.shape == (batch_size, prompt_len, config["token_count"])

        for i, cache in enumerate(kv_cache):
            assert cache["keys"] is not None, f"Block {i} keys not populated"
            assert cache["values"] is not None, f"Block {i} values not populated"
            assert cache["keys"].shape[1] == prompt_len, f"Block {i} keys wrong seq length"


def test_qwen3_incremental_generation():
    """Test incremental generation with KV cache."""
    config = make_config()
    model = Qwen3(config)
    model.eval()

    batch_size = 1
    prompt_len = 4
    tokens = torch.randint(0, config["token_count"], (batch_size, prompt_len))

    kv_cache = [{"keys": None, "values": None} for _ in range(config["num_transformers"])]

    with torch.no_grad():
        # Process prompt
        output1 = model(tokens, start_pos=0, kv_cache=kv_cache)
        assert output1.shape == (batch_size, prompt_len, config["token_count"])

        # Generate one more token
        new_token = torch.randint(0, config["token_count"], (batch_size, 1))
        output2 = model(new_token, start_pos=prompt_len, kv_cache=kv_cache)
        assert output2.shape == (batch_size, 1, config["token_count"])

        # Cache should now have prompt_len + 1 positions
        for cache in kv_cache:
            assert cache["keys"].shape[1] == prompt_len + 1


def test_qwen3_without_kv_cache():
    """Test that Qwen3 works without a KV cache for simple inference."""
    config = make_config()
    model = Qwen3(config)
    model.eval()

    batch_size = 2
    seq_len = 8
    tokens = torch.randint(0, config["token_count"], (batch_size, seq_len))

    with torch.no_grad():
        output = model(tokens, start_pos=0, kv_cache=None)

    assert output.shape == (batch_size, seq_len, config["token_count"])

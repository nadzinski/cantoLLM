import torch

from qwen3.model import FeedForward, RootMeanSquareNorm


def test_rmsnorm_basic():
    """Test that RMSNorm normalizes input and applies scaling."""
    batch_size = 2
    seq_len = 4
    embedding_dim = 8

    # Create module
    rmsnorm = RootMeanSquareNorm(token_embedding_dim=embedding_dim)

    # Create random input
    x = torch.randn(batch_size, seq_len, embedding_dim)

    # Forward pass
    output = rmsnorm(x)

    # Check output shape matches input shape
    assert output.shape == x.shape

    # Check output is not all zeros
    assert not torch.allclose(output, torch.zeros_like(output))

    # Check that scaling_weight is learnable (has grad)
    assert rmsnorm.scaling_weight.requires_grad


def test_feedforward_basic():
    """Test that FeedForward processes input through gate and value branches."""
    batch_size = 2
    seq_len = 4
    embedding_dim = 8
    expanded_dim = 32

    # Create module
    ff = FeedForward(
        token_embedding_dim=embedding_dim, expanded_dim=expanded_dim, dtype=torch.float32
    )

    # Create random input
    x = torch.randn(batch_size, seq_len, embedding_dim)

    # Forward pass
    output = ff(x)

    # Check output shape matches input shape (projects back down)
    assert output.shape == x.shape

    # Check output is not all zeros
    assert not torch.allclose(output, torch.zeros_like(output))

    # Check that weights are learnable
    assert ff.linear_1.weight.requires_grad
    assert ff.linear_2.weight.requires_grad
    assert ff.linear_3.weight.requires_grad

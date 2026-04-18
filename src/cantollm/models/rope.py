import math

import torch


def precompute_freqs_cis(dim: int, max_seq_len: int, theta: float = 100000.0):
    """
    Precompute complex numbers to do RoPE rotations for every combination of 
    1) seq position = 0...max_seq_len - 1
    2) half vector index i=0...dim/2

    Return tensor of shape (max_seq_len, dim / 2)
    """
    seq_positions = torch.arange(max_seq_len).unsqueeze(1)
    indexes = torch.arange(dim // 2).unsqueeze(0)

    angles = seq_positions * torch.exp(-2 * indexes / dim * math.log(theta))

    freqs_cis = torch.exp(1j * angles)

    return freqs_cis

def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor, offset: int = 0):
    """
    map input vectors -> rotated vectors
    i.e. we are mapping a tensor of shape 
    (batches, seq_len, num_groups, head_dim) to a tensor of the same shape
    operating only with the seq_len, head_dim indices
    """
    dtype = x.dtype
    # Cut the input tensor in half along head_dim
    # Transform input to give it an additional dimension on "which half" (0 or 1)
    halved = x.unflatten(-1, (2, -1)).float()

    # Convert to complex number using the last dim of size two for real, img parts
    complexed = torch.complex(halved[..., 0, :], halved[..., 1, :])

    # Apply offset and cut off the bits after seq_len that we won't need
    freqs_cis_reduced = freqs_cis[offset:x.shape[1] + offset, :]

    # Reshape for broadcasting: (seq_len, dim/2) -> (1, seq_len, 1..., dim/2)
    n_middle_dims = x.ndim - 3
    broadcast_shape = (1, -1) + (1,) * n_middle_dims + (freqs_cis_reduced.shape[-1],)
    freqs_cis_reduced = freqs_cis_reduced.view(broadcast_shape)

    # Do the rotation!
    rotated = complexed * freqs_cis_reduced

    # Recombine real and imaginary halves
    result = torch.cat([rotated.real, rotated.imag], dim=-1)

    return result.to(dtype)

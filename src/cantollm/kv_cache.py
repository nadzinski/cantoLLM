"""KV cache management for transformer models."""


class KVCache:
    """KV cache for transformer layers with position tracking."""

    def __init__(self, num_layers: int):
        self.layers = [{"keys": None, "values": None} for _ in range(num_layers)]

    @property
    def position(self) -> int:
        """Current sequence position (0 if empty)."""
        if self.layers[0]["keys"] is None:
            return 0
        return self.layers[0]["keys"].shape[1]

    def truncate(self, pos: int) -> None:
        """Truncate cache to given position."""
        for layer in self.layers:
            if layer["keys"] is not None:
                layer["keys"] = layer["keys"][:, :pos, ...]
                layer["values"] = layer["values"][:, :pos, ...]

    def reset(self) -> None:
        """Clear the cache."""
        for layer in self.layers:
            layer["keys"] = None
            layer["values"] = None

    def __getitem__(self, idx) -> dict:
        """Allow indexing for backward compatibility with model forward."""
        return self.layers[idx]

    def __len__(self) -> int:
        return len(self.layers)

    def __iter__(self):
        """Allow iteration over layers."""
        return iter(self.layers)

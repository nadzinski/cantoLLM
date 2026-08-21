"""Hybrid sequence state for Qwen 3.8: KV on full-attention layers,
recurrent Gated-DeltaNet state on linear-attention layers.

`HybridCache` is the sequential-path analog of kv_cache.KVCache; the
continuous-batching `HybridStatePool` (the KVPool-protocol counterpart
of PaddedKVPool) lands with the batched path.
"""

from cantollm.models.qwen38.model import FULL, LINEAR


class HybridCache:
    """Per-layer dicts, duck-compatible with KVCache where it matters
    (len/index/iterate, `position`, `reset`); layer contents differ by
    kind: {"keys", "values"} on full-attention layers (grow-by-cat,
    owned by the attention method), {"S", "conv", "pos"} on linear ones
    (owned by the model's GDN blocks).
    """

    def __init__(self, layer_types: list[str]):
        assert FULL in layer_types, "position tracking needs a full-attention layer"
        self.layer_types = list(layer_types)
        self._first_full = self.layer_types.index(FULL)
        self.layers = [
            {"keys": None, "values": None}
            if kind == FULL
            else {"S": None, "conv": None, "pos": 0}
            for kind in self.layer_types
        ]

    @property
    def position(self) -> int:
        """Current sequence position (0 if empty), read from the first
        full-attention layer like KVCache reads layer 0."""
        keys = self.layers[self._first_full]["keys"]
        return 0 if keys is None else keys.shape[1]

    def truncate(self, pos: int) -> None:
        # KV could truncate, but the GDN recurrent state cannot rewind:
        # refuse loudly rather than desync the two. This forecloses
        # speculative decoding on this family (rejected drafts rewind).
        raise NotImplementedError(
            "HybridCache cannot truncate: GDN recurrent state has no rewind"
        )

    def reset(self) -> None:
        for kind, layer in zip(self.layer_types, self.layers):
            if kind == FULL:
                layer["keys"] = None
                layer["values"] = None
            else:
                layer["S"] = None
                layer["conv"] = None
                layer["pos"] = 0

    def __getitem__(self, idx) -> dict:
        return self.layers[idx]

    def __len__(self) -> int:
        return len(self.layers)

    def __iter__(self):
        return iter(self.layers)

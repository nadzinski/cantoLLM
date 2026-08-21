"""Hybrid sequence state for Qwen 3.8: KV on full-attention layers,
recurrent Gated-DeltaNet state on linear-attention layers.

`HybridCache` is the sequential-path analog of kv_cache.KVCache;
`HybridStatePool` is the continuous-batching counterpart of
PaddedKVPool, satisfying the structural `KVPool` protocol (num_layers,
max_seq_len, device, layer(i)) while carrying per-slot GDN state for
the linear layers.
"""

import torch

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


class HybridStatePool:
    """Per-slot sequence state for the CB engine, hybrid edition.

    Full-attention layers get PaddedKVPool-style K/V tensors of shape
    (max_batch, max_seq_len + 1, num_groups, head_dim), allocated ONLY
    for those layer indices (the 27B saves 48 of 64 layers' KV memory).
    Linear layers get per-slot GDN state: S (max_batch, num_v_heads,
    Dk, Dv) in fp32 (mamba_ssm_dtype) plus the conv window
    (max_batch, conv_dim, kernel-1) in model dtype.

    Slot lifecycle: the allocator/scheduler are unchanged, so nothing
    tells the pool a slot was freed. Instead `begin_step` zeroes a
    slot's GDN state when a row arrives with start_pos == 0 (a new
    sequence claiming the slot), the same trick the padded pool plays
    with stale KV behind the causal mask. Filler rows (num_new == 0)
    alias slot 0 and must never reset or advance it.

    The per-slot `gdn_pos` counters are host-side validation only: the
    GDN recurrence cannot replay or skip, so a row whose start_pos
    disagrees with its slot's counter fails loudly (a future scheduler
    change that preempts or reorders chunks would surface here, not as
    silent state corruption).
    """

    def __init__(
        self,
        *,
        layer_types: list[str],
        max_batch: int,
        max_seq_len: int,
        num_groups: int,
        head_dim: int,
        gdn_num_v_heads: int,
        gdn_head_k_dim: int,
        gdn_head_v_dim: int,
        gdn_conv_dim: int,
        gdn_conv_kernel: int,
        dtype: torch.dtype,
        device: torch.device,
    ):
        self.layer_types = list(layer_types)
        self.num_layers = len(self.layer_types)
        self.max_batch = max_batch
        self.max_seq_len = max_seq_len
        self.scratch_pos = max_seq_len
        kv_shape = (max_batch, max_seq_len + 1, num_groups, head_dim)
        self.k_layers = {}
        self.v_layers = {}
        self.s_layers = {}
        self.conv_layers = {}
        for i, kind in enumerate(self.layer_types):
            if kind == FULL:
                self.k_layers[i] = torch.zeros(kv_shape, dtype=dtype, device=device)
                self.v_layers[i] = torch.zeros(kv_shape, dtype=dtype, device=device)
            else:
                self.s_layers[i] = torch.zeros(
                    max_batch,
                    gdn_num_v_heads,
                    gdn_head_k_dim,
                    gdn_head_v_dim,
                    dtype=torch.float32,
                    device=device,
                )
                self.conv_layers[i] = torch.zeros(
                    max_batch, gdn_conv_dim, gdn_conv_kernel - 1,
                    dtype=dtype, device=device,
                )
        self.gdn_pos = [0] * max_batch
        # Resolved device, same rationale as PaddedKVPool.
        self.device = torch.empty(0, device=device).device

    @classmethod
    def from_arch(cls, arch: dict, *, max_batch, max_seq_len, dtype, device):
        conv_dim = (
            2 * arch["linear_num_k_heads"] * arch["linear_head_k_dim"]
            + arch["linear_num_v_heads"] * arch["linear_head_v_dim"]
        )
        return cls(
            layer_types=arch["layer_types"],
            max_batch=max_batch,
            max_seq_len=max_seq_len,
            num_groups=arch["num_groups"],
            head_dim=arch["head_dim"],
            gdn_num_v_heads=arch["linear_num_v_heads"],
            gdn_head_k_dim=arch["linear_head_k_dim"],
            gdn_head_v_dim=arch["linear_head_v_dim"],
            gdn_conv_dim=conv_dim,
            gdn_conv_kernel=arch["linear_conv_kernel"],
            dtype=dtype,
            device=device,
        )

    def layer(self, i: int) -> tuple:
        if self.layer_types[i] != FULL:
            raise KeyError(
                f"layer {i} is {self.layer_types[i]}; only full-attention "
                "layers have KV storage"
            )
        return self.k_layers[i], self.v_layers[i]

    def gdn_state(self, i: int) -> tuple:
        if self.layer_types[i] != LINEAR:
            raise KeyError(
                f"layer {i} is {self.layer_types[i]}; only linear-attention "
                "layers have GDN state"
            )
        return self.s_layers[i], self.conv_layers[i]

    def begin_step(self, rows) -> None:
        """Once-per-step host bookkeeping over (slot, start_pos, num_new)
        rows; see the class docstring for the reset/monotone contract."""
        for slot, start, num_new in rows:
            if num_new == 0:
                continue
            if start == 0:
                for s in self.s_layers.values():
                    s[slot].zero_()
                for conv in self.conv_layers.values():
                    conv[slot].zero_()
                self.gdn_pos[slot] = 0
            elif self.gdn_pos[slot] != start:
                raise ValueError(
                    f"slot {slot}: GDN state is at position "
                    f"{self.gdn_pos[slot]} but the row starts at {start}; "
                    "the scan cannot replay or skip"
                )
            self.gdn_pos[slot] += num_new

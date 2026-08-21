"""Weight loading for Qwen 3.8: checkpoint-name mapping onto Qwen38.

Checkpoint layout (Qwen/Qwen3.8-27B and its FP8 variant): the text
decoder lives under `model.language_model.*`, the LM head at top-level
`lm_head.weight`, and the vision tower under `model.visual.*`. The PoC
is text-only, so visual weights are skipped; the release ships no MTP
weights but `mtp.*` is skipped defensively too.

Download + sharded/lazy loading + the FP8 wrap land with the spec
wiring chunk; this module starts at the mapping because the HF-parity
test loads its artifact through here, putting the naming and the
zero-centered-norm semantics under test from the first green run.
"""

import json
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download, snapshot_download
from safetensors import safe_open

from cantollm.models.qwen38.model import FULL

# The FP8 repo is the serving target (fits a 32GB card); it ships the
# same tokenizer as the bf16 release.
REPO_ID = "Qwen/Qwen3.8-27B-FP8"
MODEL_DATA_DIR = Path(__file__).parent.parent / "model_data"

PREFIX = "model.language_model."
SKIP_PREFIXES = ("model.visual.",)


def download_tokenizer() -> str:
    """Download only the tokenizer file and return its local dir (the
    API process needs a tokenizer but must not hold weights)."""
    local_dir = MODEL_DATA_DIR / "Qwen3.8-27B-FP8"
    hf_hub_download(
        repo_id=REPO_ID,
        filename="tokenizer.json",
        local_dir=str(local_dir),
    )
    return str(local_dir)


class ShardedWeights:
    """Lazy mapping over a sharded safetensors checkpoint: one tensor
    resident at a time. The 27B (~31GB in fp8) cannot go through the
    qwen3-style all-in-one dict; here peak host RAM per access is one
    tensor, and the loader materializes params one by one.

    Supports exactly the mapping surface load_weights_into_model needs:
    `in`, `[]`, and key iteration.
    """

    def __init__(self, local_dir: str):
        self.local_dir = Path(local_dir)
        index_path = self.local_dir / "model.safetensors.index.json"
        if index_path.exists():
            self._weight_map = json.loads(index_path.read_text())["weight_map"]
        else:
            # No index (single-file or custom shard naming): scan headers.
            self._weight_map = {}
            for shard in sorted(self.local_dir.glob("*.safetensors")):
                with safe_open(str(shard), framework="pt") as f:
                    for key in f.keys():
                        self._weight_map[key] = shard.name
        if not self._weight_map:
            raise FileNotFoundError(f"no safetensors weights under {self.local_dir}")

    def __contains__(self, name: str) -> bool:
        return name in self._weight_map

    def __iter__(self):
        return iter(self._weight_map)

    def __len__(self) -> int:
        return len(self._weight_map)

    def __getitem__(self, name: str) -> torch.Tensor:
        shard = self._weight_map.get(name)
        if shard is None:
            raise KeyError(name)
        with safe_open(str(self.local_dir / shard), framework="pt") as f:
            return f.get_tensor(name)


def download_weights() -> tuple[str, ShardedWeights]:
    """Download the FP8 checkpoint (~31GB, 48 shards) and return
    (local_dir, lazy weights view). Matches the weights_loader contract:
    the second element only needs dict-style access."""
    local_dir = download_tokenizer()
    snapshot_download(
        repo_id=REPO_ID,
        local_dir=local_dir,
        allow_patterns=["*.safetensors", "*.json"],
    )
    return local_dir, ShardedWeights(local_dir)


def _is_skipped(name: str) -> bool:
    return name.startswith(SKIP_PREFIXES) or ".mtp." in name or name.startswith("mtp.")


class _Entry:
    """Where one checkpoint tensor lands: the module owning the parameter
    attribute, plus (for Linear weights) the Linear's parent and child
    name so the FP8 path can swap the whole module."""

    __slots__ = ("owner", "attr", "parent", "child")

    def __init__(self, owner, attr, parent=None, child=None):
        self.owner = owner
        self.attr = attr
        self.parent = parent
        self.child = child

    @property
    def param(self):
        return getattr(self.owner, self.attr)


def build_param_mapping(model) -> dict:
    """Checkpoint name -> _Entry, driven by the model's own layer_types
    so full/linear layers map their respective mixers."""
    mapping = {}

    def param(name, owner, attr):
        mapping[name] = _Entry(owner, attr)

    def linear(name, parent, child):
        mapping[name] = _Entry(getattr(parent, child), "weight", parent, child)

    param(f"{PREFIX}embed_tokens.weight", model.initial_embedding_layer, "weight")
    param(f"{PREFIX}norm.weight", model.output_RMSNorm, "weight")
    # Untied for the whole family: the checkpoint always ships lm_head.
    linear("lm_head.weight", model, "output_layer")

    for i, kind in enumerate(model.layer_types):
        block = model.transformer_blocks[i]
        hf = f"{PREFIX}layers.{i}"
        param(f"{hf}.input_layernorm.weight", block.input_norm, "weight")
        param(f"{hf}.post_attention_layernorm.weight", block.post_attention_norm, "weight")
        ff = block.feed_forward
        linear(f"{hf}.mlp.gate_proj.weight", ff, "linear_1")
        linear(f"{hf}.mlp.up_proj.weight", ff, "linear_2")
        linear(f"{hf}.mlp.down_proj.weight", ff, "linear_3")
        if kind == FULL:
            attn = block.attention
            linear(f"{hf}.self_attn.q_proj.weight", attn, "W_q")
            linear(f"{hf}.self_attn.k_proj.weight", attn, "W_k")
            linear(f"{hf}.self_attn.v_proj.weight", attn, "W_v")
            linear(f"{hf}.self_attn.o_proj.weight", attn, "out_proj")
            param(f"{hf}.self_attn.q_norm.weight", attn.q_norm, "weight")
            param(f"{hf}.self_attn.k_norm.weight", attn.k_norm, "weight")
        else:
            la = block.linear_attn
            for name in ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj"):
                linear(f"{hf}.linear_attn.{name}.weight", la, name)
            param(f"{hf}.linear_attn.conv1d.weight", la.conv1d, "weight")
            param(f"{hf}.linear_attn.A_log", la, "A_log")
            param(f"{hf}.linear_attn.dt_bias", la, "dt_bias")
            param(f"{hf}.linear_attn.norm.weight", la.norm, "weight")
    return mapping


def load_weights_into_model(model, config: dict, weights_dict) -> None:
    """Install checkpoint tensors into the model, loudly.

    Three cases per mapped tensor:
      - float8-e4m3 with a `<name>_scale_inv` companion: the owning
        nn.Linear is swapped for an FP8Linear (dtype-driven, so the bf16
        checkpoint takes the plain path with zero FP8 modules);
      - meta parameter (arch["init_device"] = "meta"): replaced by a real
        Parameter built from the checkpoint tensor, one at a time, so
        peak host RAM stays ~= checkpoint size;
      - ordinary parameter: in-place copy (the tiny-fixture/test path).

    Loud in every direction: a mapped name missing from the checkpoint
    raises; a checkpoint key that is neither mapped nor a known skip
    (visual/mtp/scale companions) raises; a parameter still meta after
    loading raises. `weights_dict` only needs mapping-style access
    (`in`, `[]`, key iteration), so the lazy ShardedWeights works here.
    """
    from cantollm.models.qwen38.fp8 import FP8Linear

    mapping = build_param_mapping(model)

    with torch.no_grad():
        for hf_name, entry in mapping.items():
            if hf_name not in weights_dict:
                raise KeyError(f"Weight '{hf_name}' not found in checkpoint")
            weight = weights_dict[hf_name]
            current = entry.param
            if current.shape != weight.shape:
                raise ValueError(
                    f"Shape mismatch for '{hf_name}': "
                    f"model expects {tuple(current.shape)}, got {tuple(weight.shape)}"
                )
            if weight.dtype == torch.float8_e4m3fn:
                if entry.parent is None:
                    raise ValueError(
                        f"'{hf_name}' arrived as fp8 but is not a Linear weight"
                    )
                scale_name = f"{hf_name}_scale_inv"
                if scale_name not in weights_dict:
                    raise KeyError(f"fp8 weight '{hf_name}' has no '{scale_name}'")
                setattr(
                    entry.parent, entry.child,
                    FP8Linear(weight, weights_dict[scale_name]),
                )
            elif current.is_meta:
                setattr(
                    entry.owner, entry.attr,
                    torch.nn.Parameter(weight.to(current.dtype), requires_grad=False),
                )
            else:
                current.copy_(weight)

    still_meta = [n for n, p in model.named_parameters() if p.is_meta]
    if still_meta:
        raise RuntimeError(
            f"{len(still_meta)} parameters were never materialized "
            f"(first: {still_meta[:4]}); the mapping missed them"
        )

    unexpected = [
        k for k in weights_dict
        if k not in mapping and not _is_skipped(k) and not k.endswith("_scale_inv")
    ]
    if unexpected:
        preview = ", ".join(sorted(unexpected)[:8])
        raise ValueError(
            f"{len(unexpected)} unexpected checkpoint keys (first: {preview})"
        )

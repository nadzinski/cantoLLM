"""Run-config schema: TOML in, an ordered list of cells out (bench-spec.md §5).

A run config is:

    schema_version = 1
    name = "baseline-5090"

    [server]                      # omitted entirely in --attach mode
    model = "0.6B"
    engine = "batched"            # "sequential" → client metrics only
    device = "auto"
    port = 8377
    health_timeout_s = 900
    respawn = "per-server-config" # or "per-cell"
    max_batch = 8                 # passed through to serve flags
    batch_max_seq_len = 4096
    max_tokens_per_step = 256

    [server.matrix]               # optional cartesian sweep over [server] keys
    max_batch = [8, 16]

    [defaults]                    # per-point knobs, overridable per point
    dialect = "openai"
    max_tokens = 128
    ignore_eos = true
    temperature = 0.0
    top_p = 1.0
    warmup_requests = 8
    repeats = 3
    seed = 0

    [[points]]
    workload = "short_chat"
    mode = "closed"
    concurrency = [1, 2, 4, 8]
    requests_per_level = 64

    [[points]]
    workload = "short_chat"
    mode = "open"
    arrivals = "poisson"          # or "fixed"
    rate_rps = [0.5, 1.0]
    total_requests = 100
    max_inflight = 64

Expansion order: server variants outermost (the executor respawns only on
server-config change — §5 spawn policy), then points in file order, then
levels. Every cell carries its fully-resolved config so results are
self-describing.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path

CONFIG_SCHEMA_VERSION = 1

_POINT_DEFAULTS = {
    "dialect": "openai",
    "max_tokens": 128,
    "ignore_eos": True,
    "temperature": 0.0,
    "top_p": 1.0,
    "warmup_requests": 8,
    "repeats": 3,
    "seed": 0,
    "prompt_limit": None,   # cap prompts drawn from the workload (None = all)
}

_SERVER_DEFAULTS = {
    "model": "0.6B",
    "engine": "batched",
    "device": "auto",
    "port": 8377,
    "health_timeout_s": 900.0,
    "respawn": "per-server-config",
    "in_process": False,
}

# [server] keys that translate into `canto serve` CLI flags.
_SERVE_FLAG_KEYS = (
    "model", "engine", "device", "max_batch", "batch_max_seq_len",
    "max_tokens_per_step", "attention",
)


class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class ServerVariant:
    """One fully-resolved server configuration (one spawn)."""

    values: tuple[tuple[str, object], ...]  # sorted key/value pairs — hashable

    @property
    def as_dict(self) -> dict:
        return dict(self.values)

    @property
    def label(self) -> str:
        skip_defaults = {"port", "health_timeout_s", "respawn", "in_process"}
        parts = [
            f"{k}={v}" for k, v in self.values
            if k not in skip_defaults
        ]
        return ",".join(parts)


@dataclass(frozen=True)
class Cell:
    """One (server variant × point × level) — the unit the executor runs."""

    cell_id: str
    server: ServerVariant | None      # None in attach mode
    workload: str
    mode: str                         # "closed" | "open"
    level: float                      # concurrency (closed) or rate_rps (open)
    arrivals: str | None              # "poisson" | "fixed" (open only)
    requests: int                     # requests_per_level / total_requests
    max_inflight: int | None
    options: dict = field(default_factory=dict)  # resolved per-point knobs

    def to_dict(self) -> dict:
        return {
            "cell_id": self.cell_id,
            "server": self.server.as_dict if self.server else None,
            "workload": self.workload,
            "mode": self.mode,
            "level": self.level,
            "arrivals": self.arrivals,
            "requests": self.requests,
            "max_inflight": self.max_inflight,
            "options": self.options,
        }


@dataclass
class RunConfig:
    name: str
    raw: dict
    cells: list[Cell]
    attach: bool
    stop_on_cell_failure: bool = False

    @property
    def server_variants(self) -> list[ServerVariant]:
        seen: list[ServerVariant] = []
        for cell in self.cells:
            if cell.server is not None and cell.server not in seen:
                seen.append(cell.server)
        return seen


def load_run_config(path: str | Path) -> RunConfig:
    path = Path(path)
    with open(path, "rb") as f:
        raw = tomllib.load(f)
    return parse_run_config(raw, default_name=path.stem)


def parse_run_config(raw: dict, default_name: str = "run") -> RunConfig:
    version = raw.get("schema_version")
    if version != CONFIG_SCHEMA_VERSION:
        raise ConfigError(
            f"config schema_version {version!r} != supported {CONFIG_SCHEMA_VERSION}"
        )
    name = raw.get("name", default_name)
    points = raw.get("points")
    if not points:
        raise ConfigError("config has no [[points]]")

    defaults = {**_POINT_DEFAULTS, **raw.get("defaults", {})}

    server_section = raw.get("server")
    attach = server_section is None
    variants: list[ServerVariant | None]
    if attach:
        variants = [None]
    else:
        matrix = server_section.pop("matrix", {})
        base = {**_SERVER_DEFAULTS, **server_section}
        variants = [
            ServerVariant(values=tuple(sorted(v.items())))
            for v in _expand_matrix(base, matrix)
        ]

    cells: list[Cell] = []
    for variant_idx, variant in enumerate(variants):
        for point_idx, point in enumerate(points):
            for cell in _expand_point(
                point, point_idx, variant, variant_idx, defaults
            ):
                cells.append(cell)

    return RunConfig(
        name=name,
        raw=raw,
        cells=cells,
        attach=attach,
        stop_on_cell_failure=raw.get("stop_on_cell_failure", False),
    )


def serve_argv(variant: ServerVariant, extra: list[str] | None = None) -> list[str]:
    """The `canto serve` argv for a server variant (executor spawns this)."""
    cfg = variant.as_dict
    argv = ["serve", "--host", "127.0.0.1", "--port", str(cfg["port"])]
    for key in _SERVE_FLAG_KEYS:
        if key in cfg and cfg[key] is not None:
            argv += [f"--{key.replace('_', '-')}", str(cfg[key])]
    if cfg.get("in_process"):
        argv.append("--in-process")
    if cfg.get("shape_buckets"):
        argv.append("--shape-buckets")
    if cfg.get("warmup_shapes"):
        argv.append("--warmup-shapes")
    return argv + list(extra or [])


def _expand_matrix(base: dict, matrix: dict) -> list[dict]:
    for key, values in matrix.items():
        if not isinstance(values, list) or not values:
            raise ConfigError(f"[server.matrix] {key} must be a non-empty list")
    variants = [dict(base)]
    for key, values in matrix.items():
        variants = [{**v, key: value} for v in variants for value in values]
    return variants


def _expand_point(
    point: dict,
    point_idx: int,
    variant: ServerVariant | None,
    variant_idx: int,
    defaults: dict,
) -> list[Cell]:
    point = dict(point)
    workload = point.pop("workload", None)
    if not workload:
        raise ConfigError(f"points[{point_idx}] has no workload")
    mode = point.pop("mode", "closed")

    if mode == "closed":
        levels = point.pop("concurrency", None)
        if not levels:
            raise ConfigError(f"points[{point_idx}] (closed) needs concurrency = [...]")
        requests = point.pop("requests_per_level", None)
        if not requests:
            raise ConfigError(f"points[{point_idx}] (closed) needs requests_per_level")
        arrivals = None
        max_inflight = None
    elif mode == "open":
        levels = point.pop("rate_rps", None)
        if not levels:
            raise ConfigError(f"points[{point_idx}] (open) needs rate_rps = [...]")
        requests = point.pop("total_requests", None)
        if not requests:
            raise ConfigError(f"points[{point_idx}] (open) needs total_requests")
        arrivals = point.pop("arrivals", "poisson")
        if arrivals not in ("poisson", "fixed"):
            raise ConfigError(f"points[{point_idx}] arrivals must be poisson|fixed")
        max_inflight = point.pop("max_inflight", 64)
    else:
        raise ConfigError(f"points[{point_idx}] mode must be closed|open, got {mode!r}")

    if not isinstance(levels, list):
        levels = [levels]

    options = {**defaults, **point}
    unknown = set(options) - set(_POINT_DEFAULTS)
    if unknown:
        raise ConfigError(f"points[{point_idx}] unknown keys: {sorted(unknown)}")

    cells = []
    for level in levels:
        cell_id = f"s{variant_idx}-p{point_idx}-{mode[0]}{level:g}"
        cells.append(Cell(
            cell_id=cell_id,
            server=variant,
            workload=workload,
            mode=mode,
            level=float(level),
            arrivals=arrivals,
            requests=int(requests),
            max_inflight=max_inflight,
            options=options,
        ))
    return cells

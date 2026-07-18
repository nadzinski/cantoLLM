"""Run-config parsing + matrix/point expansion (bench-spec.md §5)."""

import pytest

from cantollm.bench.config import (
    ConfigError,
    load_run_config,
    parse_run_config,
    serve_argv,
)

BASE = {
    "schema_version": 1,
    "name": "t",
    "server": {"model": "0.6B", "engine": "batched", "port": 9000},
    "points": [
        {"workload": "short_chat", "mode": "closed",
         "concurrency": [1, 4], "requests_per_level": 8},
    ],
}


def deep(overrides: dict) -> dict:
    import copy
    cfg = copy.deepcopy(BASE)
    cfg.update(copy.deepcopy(overrides))
    return cfg


def test_expansion_order_server_outermost():
    cfg = deep({
        "server": {
            "model": "0.6B", "engine": "batched", "port": 9000,
            "matrix": {"max_batch": [8, 16]},
        },
        "points": [
            {"workload": "a", "mode": "closed", "concurrency": [1, 2],
             "requests_per_level": 8},
            {"workload": "b", "mode": "open", "rate_rps": [0.5],
             "total_requests": 10},
        ],
    })
    run = parse_run_config(cfg)
    assert len(run.cells) == 2 * 3            # 2 server variants × (2 + 1 levels)
    batches = [c.server.as_dict["max_batch"] for c in run.cells]
    assert batches == [8, 8, 8, 16, 16, 16]   # variant-major ordering
    assert [c.workload for c in run.cells[:3]] == ["a", "a", "b"]
    assert [c.level for c in run.cells[:3]] == [1.0, 2.0, 0.5]
    assert len(run.server_variants) == 2


def test_defaults_flow_and_point_overrides_win():
    cfg = deep({
        "defaults": {"max_tokens": 64, "temperature": 0.0},
        "points": [
            {"workload": "w", "mode": "closed", "concurrency": [2],
             "requests_per_level": 8, "temperature": 0.7, "ignore_eos": False},
        ],
    })
    cell = parse_run_config(cfg).cells[0]
    assert cell.options["max_tokens"] == 64        # from defaults
    assert cell.options["temperature"] == 0.7      # point override
    assert cell.options["ignore_eos"] is False
    assert cell.options["dialect"] == "openai"     # built-in default


def test_attach_mode_has_no_server():
    cfg = deep({})
    del cfg["server"]
    run = parse_run_config(cfg)
    assert run.attach and all(c.server is None for c in run.cells)
    assert run.server_variants == []


def test_open_loop_needs_rate_and_validates_arrivals():
    with pytest.raises(ConfigError, match="rate_rps"):
        parse_run_config(deep({"points": [
            {"workload": "w", "mode": "open", "total_requests": 10},
        ]}))
    with pytest.raises(ConfigError, match="arrivals"):
        parse_run_config(deep({"points": [
            {"workload": "w", "mode": "open", "rate_rps": [1], "total_requests": 10,
             "arrivals": "bursty"},
        ]}))


def test_unknown_point_keys_rejected():
    with pytest.raises(ConfigError, match="unknown keys"):
        parse_run_config(deep({"points": [
            {"workload": "w", "mode": "closed", "concurrency": [1],
             "requests_per_level": 8, "temprature": 0.5},
        ]}))


def test_schema_version_gate():
    with pytest.raises(ConfigError, match="schema_version"):
        parse_run_config(deep({"schema_version": 99}))


def test_serve_argv_maps_flags():
    cfg = deep({"server": {
        "model": "4B", "engine": "batched", "port": 9100,
        "max_batch": 16, "batch_max_seq_len": 8192, "max_tokens_per_step": 256,
        "in_process": True,
    }})
    variant = parse_run_config(cfg).cells[0].server
    argv = serve_argv(variant)
    assert argv[:5] == ["serve", "--host", "127.0.0.1", "--port", "9100"]
    joined = " ".join(argv)
    assert "--model 4B" in joined
    assert "--max-batch 16" in joined
    assert "--batch-max-seq-len 8192" in joined
    assert "--max-tokens-per-step 256" in joined
    assert "--in-process" in argv


def test_load_from_toml_file(tmp_path):
    p = tmp_path / "smoke.toml"
    p.write_text(
        'schema_version = 1\n'
        '[server]\nmodel = "0.6B"\n'
        '[[points]]\nworkload = "w"\nmode = "closed"\n'
        'concurrency = [1]\nrequests_per_level = 4\n'
    )
    run = load_run_config(p)
    assert run.name == "smoke"                 # falls back to file stem
    assert run.cells[0].requests == 4

"""Serve config file (3.5 chunk 7): [serve] TOML -> parser defaults,
CLI flags win, unknown keys and bad values are hard errors."""

from __future__ import annotations

import pytest

from cantollm.main import parse_args


def _write(tmp_path, text: str) -> str:
    path = tmp_path / "serve.toml"
    path.write_text(text)
    return str(path)


def test_file_values_apply(tmp_path):
    cfg = _write(tmp_path, """
[serve]
port = 9001
engine = "batched"
max_batch = 4
drain_timeout = 5.5
cuda_graphs = false
""")
    args, _ = parse_args(["serve", "--config", cfg])
    assert args.port == 9001
    assert args.engine == "batched"
    assert args.max_batch == 4
    assert args.drain_timeout == 5.5
    assert args.cuda_graphs is False


def test_cli_overrides_file(tmp_path):
    cfg = _write(tmp_path, """
[serve]
port = 9001
cuda_graphs = false
""")
    args, _ = parse_args([
        "serve", "--config", cfg, "--port", "9500", "--cuda-graphs",
    ])
    assert args.port == 9500
    assert args.cuda_graphs is True


def test_omitted_tristate_stays_none(tmp_path):
    # Absent from file + CLI -> None -> device-based default resolution.
    cfg = _write(tmp_path, "[serve]\nport = 9001\n")
    args, _ = parse_args(["serve", "--config", cfg])
    assert args.cuda_graphs is None
    assert args.torch_compile is None


def test_unknown_key_is_a_hard_error(tmp_path):
    cfg = _write(tmp_path, "[serve]\nmax_bathc = 4\n")
    with pytest.raises(SystemExit) as exc:
        parse_args(["serve", "--config", cfg])
    assert "unknown serve option 'max_bathc'" in str(exc.value)


def test_choices_and_types_validated(tmp_path):
    with pytest.raises(SystemExit) as exc:
        parse_args(["serve", "--config", _write(tmp_path, '[serve]\nattention = "flash"\n')])
    assert "must be one of" in str(exc.value)

    with pytest.raises(SystemExit) as exc:
        parse_args(["serve", "--config", _write(tmp_path, '[serve]\nmax_batch = "eight"\n')])
    assert "must be an integer" in str(exc.value)

    with pytest.raises(SystemExit) as exc:
        parse_args(["serve", "--config", _write(tmp_path, '[serve]\ncuda_graphs = "yes"\n')])
    assert "must be true or false" in str(exc.value)


def test_unknown_table_is_a_hard_error(tmp_path):
    cfg = _write(tmp_path, "[server]\nport = 9001\n")
    with pytest.raises(SystemExit) as exc:
        parse_args(["serve", "--config", cfg])
    assert "unknown table" in str(exc.value)


def test_missing_file_is_a_hard_error(tmp_path):
    with pytest.raises(SystemExit) as exc:
        parse_args(["serve", "--config", str(tmp_path / "nope.toml")])
    assert "not found" in str(exc.value)


def test_example_config_parses():
    args, _ = parse_args(["serve", "--config", "serve.example.toml"])
    assert args.engine == "batched"
    assert args.watchdog_timeout == 60.0

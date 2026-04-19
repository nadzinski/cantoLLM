# Development

This project uses [uv](https://docs.astral.sh/uv/) for Python + dependency management.

## Setup

```
uv sync
```

Installs dependencies and the `cantollm` package itself (editable) into `.venv/`.
After that, activate the venv:

```
source .venv/bin/activate
```

## Running tests

```
python -m pytest tests/ -v
```

## Adding dependencies

```
uv add <package>              # runtime dependency
uv add --dev <package>        # dev-only dependency
```

"""Shared helpers for the viz trace harnesses."""

import json
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "data"


def emit_js(filename: str, varname: str, obj) -> Path:
    """Write `obj` as a JS file assigning `window.<varname>`.

    The visualization (viz/index.html) is opened via file://, where fetch()
    of local JSON is blocked — so traces ship as script files instead.
    """
    DATA_DIR.mkdir(exist_ok=True)
    path = DATA_DIR / filename
    payload = json.dumps(obj, separators=(",", ":"))
    path.write_text(f"window.{varname} = {payload};\n")
    return path

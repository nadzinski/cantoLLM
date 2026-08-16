"""Structured JSON logging shared by the API and engine processes.

One JSON object per line on stderr: ts, level, logger, process, msg, plus
request_id when the emitting context carries one (set by the API's
X-Request-ID middleware via `request_id_var`). The engine child calls
`configure_logging("engine")` in its entrypoint because spawn does not
inherit the parent's logging config.

uvicorn's own loggers keep their handlers (propagate=False in its default
config), so this shapes cantollm's and third-party root-propagated records
without re-styling access logs.
"""

from __future__ import annotations

import json
import logging
import sys
from contextvars import ContextVar
from datetime import datetime, timezone

request_id_var: ContextVar[str | None] = ContextVar("request_id", default=None)


class JsonFormatter(logging.Formatter):
    def __init__(self, process: str):
        super().__init__()
        self._process = process

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc)
            .isoformat(timespec="milliseconds")
            .replace("+00:00", "Z"),
            "level": record.levelname,
            "logger": record.name,
            "process": self._process,
            "msg": record.getMessage(),
        }
        rid = getattr(record, "request_id", None) or request_id_var.get()
        if rid is not None:
            payload["request_id"] = rid
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def configure_logging(process: str, *, level: int = logging.INFO) -> None:
    """Install the JSON handler on the root logger, once per process.

    Idempotent: a second call (e.g. tests building several apps) replaces
    the previous cantollm JSON handler instead of stacking duplicates.
    """
    root = logging.getLogger()
    for handler in list(root.handlers):
        if getattr(handler, "_cantollm_json", False):
            root.removeHandler(handler)
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(JsonFormatter(process))
    handler._cantollm_json = True  # type: ignore[attr-defined]
    root.addHandler(handler)
    root.setLevel(level)

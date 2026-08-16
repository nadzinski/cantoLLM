"""Dialect-correct HTTP error envelopes.

FastAPI's defaults return `{"detail": ...}` and a 422 for body-validation
failures. Neither matches the wire contract of the APIs this server emulates:

  Anthropic: `{"type": "error", "error": {"type": ..., "message": ...}}`
  OpenAI:    `{"error": {"message": ..., "type": ..., "code", "param"}}`

and both use **400** (`invalid_request_error`), not 422, for a malformed
request. These handlers route by request path — the app serves both dialects
plus dialect-less common endpoints (`/health`, `/v1/models`) — and rewrite
`HTTPException` and `RequestValidationError` into the right shape. Common
endpoints keep FastAPI's `{"detail": ...}` default.
"""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from cantollm.lifecycle import NotReadyError

# status → Anthropic error `type`. Unlisted codes fall back to api_error.
_ANTHROPIC_TYPES = {
    400: "invalid_request_error",
    401: "authentication_error",
    403: "permission_error",
    404: "not_found_error",
    413: "request_too_large",
    429: "rate_limit_error",
    500: "api_error",
    503: "overloaded_error",
    529: "overloaded_error",
}

# status → OpenAI error `type`. 5xx collapses to server_error.
_OPENAI_TYPES = {
    400: "invalid_request_error",
    401: "authentication_error",
    403: "permission_error",
    404: "not_found_error",
    429: "rate_limit_error",
}


def _dialect(path: str) -> str | None:
    if path.startswith("/v1/messages"):
        return "anthropic"
    if path.startswith("/v1/chat/completions"):
        return "openai"
    return None


def _anthropic_envelope(status: int, message: str) -> dict:
    return {
        "type": "error",
        "error": {
            "type": _ANTHROPIC_TYPES.get(status, "api_error"),
            "message": message,
        },
    }


def _openai_envelope(status: int, message: str) -> dict:
    return {
        "error": {
            "message": message,
            "type": _OPENAI_TYPES.get(status, "server_error"),
            "code": None,
            "param": None,
        },
    }


def _render(
    path: str, status: int, message: str,
    headers: dict[str, str] | None = None,
) -> JSONResponse | None:
    """Dialect-shaped body for `path`, or None for common endpoints."""
    dialect = _dialect(path)
    if dialect == "anthropic":
        body = _anthropic_envelope(status, message)
    elif dialect == "openai":
        body = _openai_envelope(status, message)
    else:
        return None
    return JSONResponse(status_code=status, content=body, headers=headers)


def _validation_message(exc: RequestValidationError) -> str:
    """One readable line from the first validation error (loc: msg)."""
    errors = exc.errors()
    if not errors:
        return "Invalid request."
    first = errors[0]
    loc = ".".join(str(p) for p in first.get("loc", ()) if p != "body")
    msg = first.get("msg", "invalid")
    return f"{loc}: {msg}" if loc else msg


def install_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(RequestValidationError)
    async def _on_validation(request: Request, exc: RequestValidationError):
        # Both dialects use 400 invalid_request_error for a malformed body,
        # not FastAPI's default 422.
        rendered = _render(request.url.path, 400, _validation_message(exc))
        if rendered is not None:
            return rendered
        return JSONResponse(status_code=422, content={"detail": exc.errors()})

    @app.exception_handler(StarletteHTTPException)
    async def _on_http(request: Request, exc: StarletteHTTPException):
        message = exc.detail if isinstance(exc.detail, str) else str(exc.detail)
        headers = getattr(exc, "headers", None)
        rendered = _render(request.url.path, exc.status_code, message, headers)
        if rendered is not None:
            return rendered
        return JSONResponse(
            status_code=exc.status_code, content={"detail": exc.detail},
            headers=headers,
        )

    @app.exception_handler(NotReadyError)
    async def _on_not_ready(request: Request, exc: NotReadyError):
        # The model exists but cannot serve yet (warming, draining,
        # crashed). 503 with Retry-After on every path, dialect-shaped
        # where the path has a dialect.
        headers = {"Retry-After": str(exc.retry_after_s)}
        rendered = _render(request.url.path, 503, exc.detail, headers)
        if rendered is not None:
            return rendered
        return JSONResponse(
            status_code=503, content={"detail": exc.detail}, headers=headers
        )

"""Browser-based chat client.

Serves a single-page chat UI and proxies /v1/messages and /v1/models to a
running CantoLLM API server (e.g. `cantollm serve` on :8000). The proxy keeps
the SPA same-origin, so no CORS config is needed on the upstream.
"""

import http.client
from pathlib import Path
from urllib.parse import urlparse

from fastapi import FastAPI, Request, Response
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

STATIC_DIR = Path(__file__).parent / "static"


def _parse_upstream(upstream: str) -> tuple[str, int, bool]:
    parsed = urlparse(upstream)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(f"upstream must be http(s) URL, got: {upstream}")
    use_tls = parsed.scheme == "https"
    host = parsed.hostname or "localhost"
    port = parsed.port or (443 if use_tls else 80)
    return host, port, use_tls


def _connect(host: str, port: int, use_tls: bool) -> http.client.HTTPConnection:
    cls = http.client.HTTPSConnection if use_tls else http.client.HTTPConnection
    return cls(host, port, timeout=600)


def create_web_app(upstream: str) -> FastAPI:
    host, port, use_tls = _parse_upstream(upstream)
    app = FastAPI(title="CantoLLM Web Chat")

    @app.middleware("http")
    async def no_cache(request: Request, call_next):
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        return response

    @app.get("/")
    async def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/v1/models")
    async def list_models() -> Response:
        conn = _connect(host, port, use_tls)
        try:
            conn.request("GET", "/v1/models", headers={"Accept": "application/json"})
            resp = conn.getresponse()
            body = resp.read()
            content_type = resp.getheader("Content-Type", "application/json")
            return Response(content=body, status_code=resp.status, media_type=content_type)
        finally:
            conn.close()

    @app.post("/v1/messages")
    async def proxy_messages(request: Request) -> StreamingResponse:
        body_bytes = await request.body()
        client_accept = request.headers.get("accept", "text/event-stream")

        def stream() -> "iter[bytes]":
            conn = _connect(host, port, use_tls)
            try:
                conn.request(
                    "POST",
                    "/v1/messages",
                    body=body_bytes,
                    headers={
                        "Content-Type": "application/json",
                        "Accept": client_accept,
                    },
                )
                resp = conn.getresponse()
                while True:
                    chunk = resp.read1(4096)
                    if not chunk:
                        break
                    yield chunk
            finally:
                conn.close()

        return StreamingResponse(stream(), media_type="text/event-stream")

    return app


def run_server(host: str, port: int, upstream: str) -> None:
    import uvicorn

    app = create_web_app(upstream)
    print(f"\nCantoLLM web chat starting on http://{host}:{port}")
    print(f"  Proxying /v1/* → {upstream}")
    print(f"  Open http://{host}:{port} in your browser\n")
    uvicorn.run(app, host=host, port=port, log_level="info")

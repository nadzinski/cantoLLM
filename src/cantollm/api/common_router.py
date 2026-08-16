"""Dialect-agnostic routes: /health, /ready, /version, /v1/models.

Both Anthropic and OpenAI SDKs hit `GET /v1/models` but with different
response shapes. Since FastAPI can't multiplex two routes on the same
path+method and both SDKs ignore unknown fields, the payload is a union of
both dialects' shapes — one source of truth rather than near-duplicate
endpoints.
"""

from datetime import datetime, timezone

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from cantollm.api.anthropic_types import ModelInfo, ModelListResponse
from cantollm.registry import EngineRegistry

# Severity order for the aggregate /ready status: the highest-ranked state
# across models names the overall condition. Draining dominates (the server
# is going away), then crashed, then the transient build states.
_STATE_RANK = {
    "ready": 0,
    "starting": 1,
    "warming": 1,
    "restarting": 2,
    "crashed": 3,
    "stopped": 4,
    "draining": 4,
}


def build_common_router(registry: EngineRegistry) -> APIRouter:
    router = APIRouter()

    @router.get("/health")
    async def health():
        return {"status": "ok"}

    @router.get("/ready")
    async def ready():
        # 200 with per-model detail when every model is ready; 503 with the
        # same body (progress, last_error, hints) otherwise. Entries without
        # a lifecycle handle (test fakes) count as ready.
        models: dict[str, dict] = {}
        worst = "ready"
        for name, entry in registry.items():
            handle = getattr(entry, "handle", None)
            status = handle.status() if handle is not None else {"state": "ready"}
            models[name] = status
            if _STATE_RANK.get(status["state"], 1) > _STATE_RANK[worst]:
                worst = status["state"]
        body = {"status": worst, "models": models}
        code = 200 if worst == "ready" else 503
        return JSONResponse(status_code=code, content=body)

    # Environment fingerprint, computed once on first request (two git
    # subprocess calls); bench/env.py already nulls fields gracefully on
    # no-git trees (the H100 "nogit" case).
    version_cache: dict | None = None

    @router.get("/version")
    async def version():
        nonlocal version_cache
        if version_cache is None:
            from cantollm.bench.env import fingerprint

            env = fingerprint()
            version_cache = {
                "name": "cantollm",
                "git_sha": env["git_sha"],
                "git_dirty": env["git_dirty"],
                "python": env["python"],
                "torch": env["torch"],
                "device_name": env["device_name"],
                "models": {
                    name: {"engine": type(entry.engine).__name__}
                    for name, entry in registry.items()
                },
            }
        return version_cache

    @router.get("/v1/models", response_model=ModelListResponse)
    async def list_models():
        names = registry.names()
        data = [
            ModelInfo(
                id=name,
                display_name=name,
                created_at=datetime.fromtimestamp(
                    entry.registered_at, tz=timezone.utc,
                ).isoformat().replace("+00:00", "Z"),
                created=int(entry.registered_at),
            )
            for name, entry in registry.items()
        ]
        return ModelListResponse(
            data=data,
            has_more=False,
            first_id=names[0] if names else None,
            last_id=names[-1] if names else None,
        )

    return router

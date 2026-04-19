"""Dialect-agnostic routes: /health, /v1/models.

Both Anthropic and OpenAI SDKs hit `GET /v1/models` but with different
response shapes. Since FastAPI can't multiplex two routes on the same
path+method and both SDKs ignore unknown fields, the payload is a union of
both dialects' shapes — one source of truth rather than near-duplicate
endpoints.
"""

from datetime import datetime, timezone

from fastapi import APIRouter

from cantollm.api.anthropic_types import ModelInfo, ModelListResponse
from cantollm.registry import EngineRegistry


def build_common_router(registry: EngineRegistry) -> APIRouter:
    router = APIRouter()

    @router.get("/health")
    async def health():
        return {"status": "ok"}

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

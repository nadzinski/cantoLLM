"""Admin routes: POST /admin/reload, POST /admin/restart.

Reload = drain-then-rebuild of the same model + config (fresh weights,
re-warm); switching models means restarting the server with a different
config. Restart recovers a CRASHED engine (and from READY behaves exactly
like reload). Both return 202 immediately; progress is watched on /ready.

No auth by design (PLAN.md non-goal: that belongs in a sidecar); these
endpoints mutate server state, so bind the server accordingly.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from cantollm.registry import EngineRegistry


class AdminTarget(BaseModel):
    model: str | None = None


def build_admin_router(registry: EngineRegistry) -> APIRouter:
    router = APIRouter()

    def _resolve(body: AdminTarget | None):
        names = registry.names()
        name = body.model if body is not None and body.model is not None else (
            names[0] if len(names) == 1 else None
        )
        if name is None:
            raise HTTPException(
                status_code=400,
                detail=f"multiple models registered; pass model. Available: {names}",
            )
        try:
            entry = registry.get(name)
        except KeyError:
            raise HTTPException(
                status_code=404,
                detail=f"Model '{name}' is not registered. Available: {names}",
            )
        handle = getattr(entry, "handle", None)
        if handle is None:
            raise HTTPException(
                status_code=409,
                detail=f"Model '{name}' is not lifecycle-managed",
            )
        return name, handle

    def _admin_response(accepted: bool, verb: str, name: str, handle):
        if not accepted:
            raise HTTPException(
                status_code=409,
                detail=f"cannot {verb} '{name}' while {handle.state.value}",
            )
        return {
            "status": "accepted",
            "model": name,
            "state": handle.state.value,
            "watch": "/ready",
        }

    @router.post("/admin/reload", status_code=202)
    async def reload(body: AdminTarget | None = None):
        name, handle = _resolve(body)
        return _admin_response(handle.request_reload(), "reload", name, handle)

    @router.post("/admin/restart", status_code=202)
    async def restart(body: AdminTarget | None = None):
        name, handle = _resolve(body)
        return _admin_response(handle.request_restart(), "restart", name, handle)

    return router

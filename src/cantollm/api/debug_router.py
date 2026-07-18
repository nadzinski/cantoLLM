"""Debug router: the bench harness's engine-stats scrape endpoint.

`GET /debug/engine-stats` exposes the `EngineStatsAccumulator` that both
batching engines maintain (see engine/batching/stats.py). It is a localhost
tool for the bench harness and its live UI, not part of either dialect's
wire contract — Phase 3.5's `/metrics` owns the real observability posture.
Engines without an accumulator (sequential) report `available: false`.

Scrape protocol: pass `since=<seq>` from the previous response's
`next_since` to page through the step ring; a gap in `seq` values means the
ring wrapped between scrapes (the harness records a validity warning).
"""

from fastapi import APIRouter, HTTPException

from cantollm.engine.batching.stats import STATS_SCHEMA_VERSION
from cantollm.registry import EngineRegistry


def build_debug_router(registry: EngineRegistry) -> APIRouter:
    router = APIRouter()

    @router.get("/debug/engine-stats")
    async def engine_stats(model: str | None = None, since: int = -1):
        names = registry.names()
        name = model if model is not None else (names[0] if len(names) == 1 else None)
        if name is None:
            raise HTTPException(
                status_code=400,
                detail=f"multiple models registered; pass ?model=. Available: {names}",
            )
        try:
            entry = registry.get(name)
        except KeyError:
            raise HTTPException(
                status_code=404,
                detail=f"Model '{name}' is not registered. Available: {names}",
            )

        accumulator = getattr(entry.engine, "engine_stats", None)
        if accumulator is None:
            return {
                "schema_version": STATS_SCHEMA_VERSION,
                "model": name,
                "available": False,
            }
        return {
            "schema_version": STATS_SCHEMA_VERSION,
            "model": name,
            "available": True,
            **accumulator.read(since),
        }

    return router

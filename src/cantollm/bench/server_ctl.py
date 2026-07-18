"""Server-under-test lifecycle + engine-stats scraping (bench-spec.md §5).

`ServerProcess` spawns `python -m cantollm.main serve ...` (or any injected
argv — the executor tests spawn a stub), captures its output to a log file,
polls /health until ready, and tears down SIGTERM→SIGKILL. `AttachedServer`
is the --attach shape: same interface, no lifecycle.

`StatsScraper` polls /debug/engine-stats with the since-cursor at ~1 Hz so
the ring can't silently wrap mid-repeat; `take()` drains what accumulated —
the executor calls it at repeat boundaries to window the step records.
"""

from __future__ import annotations

import asyncio
import subprocess
import sys
import time
from pathlib import Path

import httpx

SCRAPE_INTERVAL_S = 1.0
HEALTH_POLL_S = 0.25
TERMINATE_GRACE_S = 5.0


def default_serve_command(serve_argv: list[str]) -> list[str]:
    return [sys.executable, "-m", "cantollm.main", *serve_argv]


class ServerStartupError(RuntimeError):
    pass


class ServerProcess:
    def __init__(self, command: list[str], base_url: str, log_path: Path):
        self.command = command
        self.base_url = base_url.rstrip("/")
        self.log_path = log_path
        self._proc: subprocess.Popen | None = None
        self._log_file = None
        self.spawn_to_ready_s: float | None = None

    @property
    def alive(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    @property
    def exitcode(self) -> int | None:
        return None if self._proc is None else self._proc.poll()

    async def start(self, health_timeout_s: float) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_file = open(self.log_path, "ab")
        t0 = time.perf_counter()
        self._proc = subprocess.Popen(
            self.command, stdout=self._log_file, stderr=subprocess.STDOUT,
        )
        deadline = t0 + health_timeout_s
        async with httpx.AsyncClient(timeout=2.0) as client:
            while True:
                if not self.alive:
                    raise ServerStartupError(
                        f"server died during startup (exit {self.exitcode}); "
                        f"log tail:\n{self.log_tail()}"
                    )
                if time.perf_counter() > deadline:
                    await self.stop()
                    raise ServerStartupError(
                        f"server not healthy after {health_timeout_s:.0f}s; "
                        f"log tail:\n{self.log_tail()}"
                    )
                try:
                    r = await client.get(f"{self.base_url}/health")
                    if r.status_code == 200:
                        break
                except httpx.HTTPError:
                    pass
                await asyncio.sleep(HEALTH_POLL_S)
        self.spawn_to_ready_s = time.perf_counter() - t0

    async def stop(self) -> None:
        if self._proc is None:
            return
        if self.alive:
            self._proc.terminate()
            try:
                await asyncio.to_thread(self._proc.wait, TERMINATE_GRACE_S)
            except subprocess.TimeoutExpired:
                self._proc.kill()
                await asyncio.to_thread(self._proc.wait)
        if self._log_file is not None:
            self._log_file.close()
            self._log_file = None

    def log_tail(self, lines: int = 50) -> str:
        try:
            text = self.log_path.read_text(errors="replace")
        except OSError:
            return "<no log>"
        return "\n".join(text.splitlines()[-lines:])


class AttachedServer:
    """--attach: someone else owns the process (possibly vLLM)."""

    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self.spawn_to_ready_s = None
        self.alive = True
        self.exitcode = None

    async def start(self, health_timeout_s: float) -> None:
        async with httpx.AsyncClient(timeout=5.0) as client:
            r = await client.get(f"{self.base_url}/health")
            if r.status_code != 200:
                raise ServerStartupError(
                    f"attached server unhealthy: {r.status_code}"
                )

    async def stop(self) -> None:
        pass

    def log_tail(self, lines: int = 50) -> str:
        return "<attached server; no log capture>"


class StatsScraper:
    """Background /debug/engine-stats poller.

    `available` turns False permanently on the first available:false or
    404 — sequential engines and non-CantoLLM attach targets simply have
    no engine stats. A seq gap between consecutive scrapes sets
    `overflowed` (ring wrapped → validity warning).
    """

    def __init__(self, client: httpx.AsyncClient, model: str | None = None):
        self._client = client
        self._model = model
        self._since = -1
        self._task: asyncio.Task | None = None
        self.steps: list[dict] = []
        self.itl: list[dict] = []
        self.available: bool | None = None
        self.overflowed = False
        self.load_seconds: float | None = None
        self.capacity: dict = {}

    async def scrape_once(self) -> None:
        if self.available is False:
            return
        params: dict = {"since": self._since}
        if self._model:
            params["model"] = self._model
        try:
            r = await self._client.get("/debug/engine-stats", params=params)
        except httpx.HTTPError:
            return  # transient; server death is detected elsewhere
        if r.status_code != 200:
            self.available = False
            return
        body = r.json()
        if not body.get("available"):
            self.available = False
            return
        self.available = True
        self.load_seconds = body.get("load_seconds")
        self.capacity = body.get("capacity") or {}
        steps = body.get("steps") or []
        if steps:
            if self._since >= 0 and steps[0]["seq"] > self._since + 1:
                self.overflowed = True
            self.steps.extend(steps)
            self.itl.extend(body.get("itl") or [])
            self._since = body.get("next_since", self._since)

    def take(self) -> tuple[list[dict], list[dict]]:
        """Drain accumulated records — the executor's repeat window."""
        steps, self.steps = self.steps, []
        itl, self.itl = self.itl, []
        return steps, itl

    async def start(self) -> None:
        await self.scrape_once()

        async def loop():
            while True:
                await asyncio.sleep(SCRAPE_INTERVAL_S)
                await self.scrape_once()

        if self.available is not False:
            self._task = asyncio.create_task(loop())

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        await self.scrape_once()  # final drain

    async def wait_drained(self, timeout_s: float = 30.0) -> bool:
        """Drain barrier between cells: queue empty and slots free."""
        if self.available is False:
            await asyncio.sleep(0.5)  # sequential: brief settle
            return True
        deadline = time.perf_counter() + timeout_s
        while time.perf_counter() < deadline:
            await self.scrape_once()
            recent = self.steps[-1] if self.steps else None
            if recent is None or (
                recent["queue_depth"] == 0 and recent["occupied_slots"] == 0
            ):
                return True
            await asyncio.sleep(0.25)
        return False

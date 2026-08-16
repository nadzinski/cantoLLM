"""Prometheus /metrics for the API process (Phase 3.5 chunk 4).

One `Metrics` object per app, with its own CollectorRegistry (the global
default registry would break tests that build many apps per process).
Four feeds:

- engine steps: pushed through `EngineStatsAccumulator.on_record` (wired
  per handle via `stats_observer`), so counters/histograms observe every
  step exactly once, restarts included (seqs already rebased upstream);
- requests: per-request observers (TTFT, duration, output tokens, finish
  reasons) driven by the tracked_events wrapper in api/common.py;
- HTTP: a latency histogram observed by the app middleware, labeled by
  route template (never the raw path — cardinality);
- pull-time gauges: a custom collector reads process CPU/RSS (psutil, API
  and engine child PIDs), GPU state (NVML; silently absent without a GPU),
  per-model in-flight / admission waiters, and event-loop lag (sampled by
  a background task the lifespan owns).

Everything is model-labeled where a model is in scope. Metric names all
start `cantollm_`.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, Callable, Iterable

from prometheus_client import (
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from prometheus_client.core import GaugeMetricFamily

logger = logging.getLogger(__name__)

CONTENT_TYPE = "text/plain; version=0.0.4; charset=utf-8"

_TTFT_BUCKETS = (0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0,
                 30.0, 60.0)
_DURATION_BUCKETS = (0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0,
                     120.0, 300.0)
_ITL_BUCKETS = (0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0)
_STEP_BUCKETS = (0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 1.0)
_TOKENS_BUCKETS = (1, 4, 16, 64, 128, 256, 512, 1024, 4096)
_HTTP_BUCKETS = (0.005, 0.025, 0.1, 0.25, 1.0, 2.5, 10.0, 30.0, 60.0, 120.0)


class _RequestObserver:
    """Per-request metrics sink bound to a model label; fed by
    api.common.tracked_events."""

    __slots__ = ("_metrics", "_model")

    def __init__(self, metrics: "Metrics", model: str):
        self._metrics = metrics
        self._model = model

    def first_token(self, ttft_s: float) -> None:
        self._metrics.request_ttft.labels(model=self._model).observe(ttft_s)

    def finished(self, duration_s: float, output_tokens: int,
                 reason: str | None) -> None:
        m = self._metrics
        m.request_duration.labels(model=self._model).observe(duration_s)
        m.request_output_tokens.labels(model=self._model).observe(output_tokens)
        m.requests_finished.labels(
            model=self._model, reason=reason or "disconnect"
        ).inc()


class Metrics:
    def __init__(self, engine_registry: Any):
        self._engine_registry = engine_registry
        self.registry = CollectorRegistry()
        reg = self.registry

        # requests
        self.request_ttft = Histogram(
            "cantollm_request_ttft_seconds",
            "Time from engine submit to the first token event (API side)",
            ["model"], buckets=_TTFT_BUCKETS, registry=reg)
        self.request_duration = Histogram(
            "cantollm_request_duration_seconds",
            "Time from engine submit to stream end",
            ["model"], buckets=_DURATION_BUCKETS, registry=reg)
        self.request_output_tokens = Histogram(
            "cantollm_request_output_tokens",
            "Output tokens per finished request",
            ["model"], buckets=_TOKENS_BUCKETS, registry=reg)
        self.requests_finished = Counter(
            "cantollm_requests_finished_total",
            "Finished requests by terminal reason",
            ["model", "reason"], registry=reg)

        # engine steps
        self.engine_steps = Counter(
            "cantollm_engine_steps_total", "Scheduler steps executed",
            ["model"], registry=reg)
        self.engine_prefill_tokens = Counter(
            "cantollm_engine_prefill_tokens_total",
            "Prompt tokens consumed by forwards", ["model"], registry=reg)
        self.engine_decode_tokens = Counter(
            "cantollm_engine_decode_tokens_total",
            "Decode tokens produced by forwards", ["model"], registry=reg)
        self.engine_step_duration = Histogram(
            "cantollm_engine_step_duration_seconds",
            "Wall time inside scheduler.step()",
            ["model"], buckets=_STEP_BUCKETS, registry=reg)
        self.engine_itl = Histogram(
            "cantollm_engine_itl_seconds",
            "Engine-side inter-token latency (per request, step-derived)",
            ["model"], buckets=_ITL_BUCKETS, registry=reg)
        self.engine_graph_replays = Counter(
            "cantollm_engine_graph_replays_total",
            "Steps served by CUDA graph replay", ["model"], registry=reg)
        self.engine_queue_depth = Gauge(
            "cantollm_engine_queue_depth",
            "Sequences waiting in the scheduler queue (latest step)",
            ["model"], registry=reg)
        self.engine_occupied_slots = Gauge(
            "cantollm_engine_occupied_slots",
            "KV slots held after the latest step", ["model"], registry=reg)
        self.engine_kv_tokens = Gauge(
            "cantollm_engine_kv_tokens",
            "Sum of active sequences' positions (KV utilization proxy)",
            ["model"], registry=reg)

        # http
        self.http_duration = Histogram(
            "cantollm_http_request_duration_seconds",
            "HTTP latency by route template and status",
            ["route", "method", "status"], buckets=_HTTP_BUCKETS,
            registry=reg)

        # event loop
        self.event_loop_lag = Gauge(
            "cantollm_event_loop_lag_seconds",
            "How late a 100 ms sleep fires on the API event loop",
            registry=reg)

        reg.register(_PullCollector(engine_registry))

    # --- wiring ---------------------------------------------------------

    def wire_handles(self) -> None:
        """Attach step + request observers to every lifecycle handle. Runs
        at app build, before launch_all, so generation 1's adoption already
        sees the observer."""
        for name, entry in self._engine_registry.items():
            handle = getattr(entry, "handle", None)
            if handle is None:
                continue
            handle.stats_observer = self._step_observer(name)
            handle.request_observer_factory = (
                lambda model=name: _RequestObserver(self, model)
            )

    def _step_observer(self, model: str) -> Callable:
        def observe(stats, itl_gaps: list[float]) -> None:
            if stats is not None:
                self.engine_steps.labels(model=model).inc()
                self.engine_prefill_tokens.labels(model=model).inc(
                    stats.prefill_tokens)
                self.engine_decode_tokens.labels(model=model).inc(
                    stats.decode_tokens)
                self.engine_step_duration.labels(model=model).observe(
                    stats.dur_s)
                if stats.graph_replayed:
                    self.engine_graph_replays.labels(model=model).inc()
                self.engine_queue_depth.labels(model=model).set(
                    stats.queue_depth)
                self.engine_occupied_slots.labels(model=model).set(
                    stats.occupied_slots)
                self.engine_kv_tokens.labels(model=model).set(stats.kv_tokens)
            itl = self.engine_itl.labels(model=model)
            for gap in itl_gaps:
                itl.observe(gap)

        return observe

    # --- request path ---------------------------------------------------

    def observe_http(self, route: str, method: str, status: int,
                     duration_s: float) -> None:
        self.http_duration.labels(
            route=route, method=method, status=str(status)
        ).observe(duration_s)

    # --- event-loop lag task --------------------------------------------

    async def run_loop_lag_sampler(self, interval_s: float = 0.1) -> None:
        loop = asyncio.get_running_loop()
        while True:
            t0 = loop.time()
            await asyncio.sleep(interval_s)
            self.event_loop_lag.set(max(0.0, loop.time() - t0 - interval_s))

    # --- exposition ------------------------------------------------------

    def render(self) -> bytes:
        return generate_latest(self.registry)


class _PullCollector:
    """Scrape-time gauges: process CPU/RSS for both PIDs, GPU via NVML,
    per-model in-flight and admission waiters."""

    def __init__(self, engine_registry: Any):
        self._engine_registry = engine_registry
        self._psutil = None
        self._procs: dict[int, Any] = {}
        try:
            import psutil

            self._psutil = psutil
        except Exception:  # pragma: no cover - psutil is a hard dep
            logger.warning("psutil unavailable; process metrics disabled")
        self._nvml = None
        try:
            import pynvml

            pynvml.nvmlInit()
            self._nvml = pynvml
        except Exception:
            self._nvml = None  # no GPU / no driver: silently absent

    def _proc(self, pid: int):
        proc = self._procs.get(pid)
        if proc is None:
            proc = self._psutil.Process(pid)
            proc.cpu_percent(interval=None)  # prime the sampler
            self._procs[pid] = proc
        return proc

    def collect(self) -> Iterable:
        cpu = GaugeMetricFamily(
            "cantollm_process_cpu_percent", "Process CPU percent",
            labels=["process"])
        rss = GaugeMetricFamily(
            "cantollm_process_rss_bytes", "Process resident set size",
            labels=["process"])
        if self._psutil is not None:
            targets = [("api", os.getpid())]
            for _, entry in self._engine_registry.items():
                engine = getattr(entry, "engine", None)
                proc = getattr(engine, "_proc", None)
                if proc is not None and proc.pid is not None:
                    targets.append(("engine", proc.pid))
            for label, pid in targets:
                try:
                    p = self._proc(pid)
                    cpu.add_metric([label], p.cpu_percent(interval=None))
                    rss.add_metric([label], p.memory_info().rss)
                except Exception:
                    self._procs.pop(pid, None)  # process went away
        yield cpu
        yield rss

        inflight = GaugeMetricFamily(
            "cantollm_inflight_requests",
            "Admitted requests currently in flight", labels=["model"])
        waiters = GaugeMetricFamily(
            "cantollm_admission_waiters",
            "Requests queued for an admission slot", labels=["model"])
        for name, entry in self._engine_registry.items():
            handle = getattr(entry, "handle", None)
            if handle is None:
                continue
            inflight.add_metric([name], handle.inflight)
            waiters.add_metric([name], getattr(handle, "_admission_waiters", 0))
        yield inflight
        yield waiters

        if self._nvml is not None:
            yield from self._collect_gpu()

    def _collect_gpu(self) -> Iterable:
        nv = self._nvml
        util = GaugeMetricFamily(
            "cantollm_gpu_utilization_percent", "GPU compute utilization",
            labels=["gpu"])
        mem_used = GaugeMetricFamily(
            "cantollm_gpu_memory_used_bytes", "GPU memory used",
            labels=["gpu"])
        mem_total = GaugeMetricFamily(
            "cantollm_gpu_memory_total_bytes", "GPU memory total",
            labels=["gpu"])
        power = GaugeMetricFamily(
            "cantollm_gpu_power_watts", "GPU power draw", labels=["gpu"])
        sm_clock = GaugeMetricFamily(
            "cantollm_gpu_sm_clock_mhz", "GPU SM clock", labels=["gpu"])
        temp = GaugeMetricFamily(
            "cantollm_gpu_temperature_celsius", "GPU temperature",
            labels=["gpu"])
        try:
            for i in range(nv.nvmlDeviceGetCount()):
                h = nv.nvmlDeviceGetHandleByIndex(i)
                idx = str(i)
                util.add_metric([idx], nv.nvmlDeviceGetUtilizationRates(h).gpu)
                mem = nv.nvmlDeviceGetMemoryInfo(h)
                mem_used.add_metric([idx], mem.used)
                mem_total.add_metric([idx], mem.total)
                power.add_metric([idx], nv.nvmlDeviceGetPowerUsage(h) / 1000.0)
                sm_clock.add_metric(
                    [idx], nv.nvmlDeviceGetClockInfo(h, nv.NVML_CLOCK_SM))
                temp.add_metric(
                    [idx],
                    nv.nvmlDeviceGetTemperature(h, nv.NVML_TEMPERATURE_GPU))
        except Exception:  # pragma: no cover - transient NVML hiccup
            logger.debug("NVML scrape failed", exc_info=True)
        yield from (util, mem_used, mem_total, power, sm_clock, temp)

"""CantoLLM bench harness (bench-spec.md is the contract).

Layering, bottom-up:
  records/metrics      — result dataclasses + the math over them
  workloads/config/env — inputs: prompt sets, run configs, environment
  history              — bench/history/<run_id>/ persistence
  sse_clients/loadgen  — HTTP drivers (closed- and open-loop)
  server_ctl/executor  — server lifecycle + the run state machine
  service              — the control-panel FastAPI app (`canto bench ui`)

Everything above sse_clients is torch-free and unit-tests in milliseconds.
"""

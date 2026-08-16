# Observability stack (Phase 3.5)

Prometheus (metrics) + Tempo (traces) + Grafana (one UI for both), for the
5090 box. Host networking: the containers share the host's localhost, so
the server's `/metrics` and the OTLP export need no address plumbing.

## Bring-up

```
cd observability
docker compose up -d
```

Then serve with tracing pointed at Tempo:

```
canto serve --engine batched --otlp-endpoint http://localhost:4318
```

Metrics need nothing: Prometheus scrapes `localhost:8000/metrics` every 5 s
whenever a server is up. Grafana is at `http://localhost:3000` (anonymous
admin, login form disabled — lab box only), with the "CantoLLM serve"
dashboard provisioned from `grafana/dashboards/cantollm.json` and both
datasources preconfigured. Traces: Explore -> Tempo, or search by trace id
from a request's span.

## Notes

- The dashboard file is the source of truth: edit the JSON (or export from
  the UI and copy back) and `docker compose restart grafana`.
- Ports: Grafana 3000, Prometheus 9090, Tempo query 3200, OTLP/HTTP 4318.
- `network_mode: host` is why this stack targets the Linux box; on a Mac,
  host networking works differently — run the bench/serve stack there
  instead and keep observability on the 5090.
- Trace retention is one week (tempo.yaml `block_retention`); Prometheus
  keeps its default 15 d.

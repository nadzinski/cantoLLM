# 5090 traced-arm rerun: OTLP tracing overhead — 2026-08-16

The tracing arm of the Phase 3.5 validation round: same config as
`2026-08-16T141102_3d9acef_ab-5090-compile` (the untraced pair — see its
agent-summary.md for the round's full context) but run with
`OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318` inherited by the
bench-spawned servers, so every request traced at 100% sampling into the
live Tempo container from `observability/`. Sha f8a826a differs from the
untraced runs' 3d9acef only by that run dir's own commit — no code
changes.

**Verdict: tracing at 100% sampling is within noise. PASS.**

Traced vs untraced medians (aggregate tok/s):

| cell | arm | untraced | traced | Δ |
|---|---|---:|---:|---:|
| short_chat c=16 | compile-dynamic | 3595.9 | 3595.0 | −0.0% |
| short_chat c=16 | compile-batch-bucket | 3594.8 | 3526.8 | −1.9% |
| short_chat c=16 | baseline pair | 2470.6 / 2459.0 | 2567.0 / 2445.6 | +3.9% / −0.5% |
| short_chat c=4 | compile-dynamic | 1332.3 | 1255.2 | −5.8% |
| short_chat c=4 | compile-batch-bucket | 1329.4 | 1263.3 | −5.0% |
| short_chat c=4 | baseline pair | 813.1 / 811.4 | 843.3 / 809.6 | +3.7% / −0.2% |
| code c=8 | compile-dynamic | 2083.2 | 2092.1 | +0.4% |
| multi_turn c=8 | compile-dynamic | 1695.4 | 1713.0 | +1.0% |

Reading the noise honestly: the baseline pair — two identical arms —
disagrees with itself by 4-5% inside the traced run (2567 vs 2446 at
c=16), and one baseline arm coming out *faster* under tracing is the
tell that these swings are environmental, not tracing cost. The c=4
compile dips (−5/−5.8%) sit inside their own 7% intra-repeat spreads
(traced repeats [1236, 1255, 1332] vs untraced [1284, 1332, 1332]).
The cells with tight repeats — c=16 dynamic (−0.0%), code (+0.4%),
multi_turn (+1.0%) — show no measurable tracing cost at all. Export
volume was real: Tempo's search API returned its 1000-trace limit for
the bench window, root spans from the bench's openai-dialect requests.

Environment note: unlike the untraced pair, this run had dockerd plus
the Prometheus/Tempo/Grafana compose stack live on the box (Prometheus
scraping :8378 targets is a connection-refused no-op between bench
server spawns; Tempo ingesting at full request rate) — that stack IS
the measured condition, deliberately.

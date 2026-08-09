# 5090 validation summary: torch.compile (Phase 3) — 2026-08-08, BLOCKED

Agent run log for the CUDA-TEST-AGENT-INST.md protocol, against
torch-compile-design.md §6/§7. **The A/B pair did not run.** The §2 step
profile hit the design note's §4 "in-place pool writes under
functionalization" hazard in its worst form: Inductor turns the per-layer
KV-pool scatter into chained pool-scale copy kernels, making a compiled
decode step ~23x *slower* than eager (213 vs 9 ms at the serve geometry)
and OOMing outright at the 48-slot probe geometry. Root cause isolated to
a 6-line standalone repro (`scripts/repro_view_mutation.py`); the fix is a
`PaddedKVPool` layout change (per-layer tensors instead of one stacked
tensor), which is an author decision, so this round stops here per the
protocol's record-and-stop rule. No production code was changed; this
directory is evidence only (no bench run.json — the runs never started).

Environment: torch 2.10.0+cu128, driver 580.173.02 (matched stack
post-reboot; the graphs round's userspace/kernel mismatch is gone),
sm_120, sha c4c7f1e.

## Step 0/1 — box sanity and suite

nvidia-smi clean, `torch.cuda.is_available()` True, no error 804.
Suite: **455 passed, 3 skipped** (all three MPS-only), including the four
`TestCaptureReplayCUDA` graph tests (bit-exact vs eager, graphs without
compile) and all of `tests/test_torch_compile.py`. `fullgraph=True` raised
nowhere — prediction 1's tracing claim holds on CUDA. The failure below is
invisible to the suite by construction: the compile tests run Dynamo with
`backend="eager"`, and AOTAutograd functionalization + Inductor lowering
only run on the real backend.

## Step 2 — profile_step recheck (probe, 0.6B, attention "padded")

Eager arm (unmodified script) reproduces step-profiling.md: 1-row
9.44 ms/step (fwd_call 9.05), 16-row 13.50 ms; **1859** cudaLaunchKernel
calls/step at 16 rows (`logs/profile_eager.log`).

Compile arm, first attempt (scratch copy per protocol: 48-slot Phase A
with `**default_shape_buckets(48, 256)`, `warmup_shapes=True,
torch_compile=True, cuda_graphs=False`): **OOM during the warm-up sweep**
(`logs/profile_compile.log`) — the generated code allocates a
`(28, 48, 4097, 8, 128)` bf16 buffer, i.e. a full copy of the K pool
including the scratch column, 10.50 GiB, on top of the 21 GiB the two
pool tensors already hold.

Compile arm, 16-slot fallback (same knobs, `max_batch=16`, occupancies
1/8/16; `logs/profile_compile16.log`):

| rows | total ms (eager → compiled) | fwd_call ms | fwd_sync ms |
|-----:|----------------------------:|------------:|------------:|
| 1    | 9.44 → 224.3                | 9.05 → 4.41 | 0.20 → 219.5 |
| 8    | 10.40 → 224.3               | 9.51 → 4.28 | 0.22 → 219.3 |
| 16   | 13.50 → 226.7               | 9.44 → 4.23 | 2.77 → 221.2 |

cudaLaunchKernel calls/step at 16 rows: **1859 → 218** — the fusion
itself worked. But the census shows where the time goes: ~13 Triton
pointwise kernels per step at 14.5–17.2 ms *each* (`triton_poi_fused_21/
31/49/58/67`, `triton_poi_fused_copy__76`, …), each moving pool-scale
data. The step is ~225 ms flat regardless of row count — pure pool
bandwidth, zero dispatch dependence. The compiled `fwd_call` is 4.2–4.4 ms
(Dynamo guard evaluation + ~600 `cuLaunchKernel` calls; the protocol asked
for this number — it is itself non-trivial and worth remembering once the
copies are fixed).

## The blocker, isolated

Serve-geometry probe (`scripts/probe_functionalization.py` — 16×4096,
**sdpa**, strategy dynamic, no scheduler, real 16-row and 1-row decode
metas): eager 9.06/9.37 ms (1/16 rows) at 8.45 GiB peak; compiled
**212.9/213.9 ms at 24.47 GiB peak**. So the pathology is not an artifact
of the profile script, the padded einsum method, or the 48-slot geometry —
it is the serve path.

Standalone repro (`scripts/repro_view_mutation.py`, pool sized
28×16×4097×8×128 bf16 ≈ 3.5 GiB, identical scatter code in both forms):

| form | eager | compiled | peak mem |
|---|---:|---:|---:|
| stacked pool, write via `pool[i]` select view | 0.29 ms | **104.7 ms** | 3.5 → **11.5 GiB** |
| per-layer tensors, direct write | 0.29 ms | **0.13 ms** | 3.5 GiB flat |

Mechanism: `PaddedKVPool` holds one stacked `(L, B, S+1, G, D)` tensor per
K/V, and the traced region writes through the per-layer select view
(`layer_k, layer_v = pool.layer(i)`; `layer_k[m.slot, m.pos] = ...`).
AOTAutograd keeps *direct* input mutations in the graph and in place, but
a mutation through a **view of an input** is functionalized — each layer's
`index_put` becomes a rebuild of the base tensor, chained across all 28
layers, and Inductor's re-inplacing pass does not recover it. The same
write against per-layer tensors (each a graph input, mutated directly) is
kept in place and compiles to the expected tiny scatter. This is the
supported static-KV-cache compile pattern used elsewhere (gpt-fast, HF
`StaticCache`, vLLM): **per-layer cache tensors, never layer-views of a
stacked tensor, inside a compiled region.**

The design note's §4 hazard entry predicted "an extra copy of the pool
slice per layer" caught by kernel counts; reality is pool-*scale* copies
caught by a 23x step time and an OOM. The CPU explain-day could not see
it: Dynamo tracing is functionalization-free, and the tests' 2-layer
64-dim fixture makes the copies numerically invisible and microscopically
cheap.

Not attempted here (author's call, in ascending order of invasiveness):
(a) restructure `PaddedKVPool` to hold per-layer tensors (`pool.layer(i)`
interface survives; `pool.k`/`pool.v` stacked-tensor consumers — graphs.py
device lookup, several tests' whole-pool asserts and `pool.k[:, 0]`-style
dirtying — need touching; the repro shows this fixes it outright);
(b) route the write through the base with the layer index baked into the
scatter indices (keeps the stacked layout, changes the attention-method
template's write and the `layer_k/v` plumbing signature);
(c) wait/pin on a torch that re-inplaces select-view mutations (none known).

## Steps 3–5 — not run

Cold/warm Ready bills, recompile tripwire under traffic, capture-under-
compile, and the three-arm greedy equivalence all presuppose a compiled
forward that is at least plausibly competitive; every §7 gate would fail
on the copies alone and measure nothing about fusion. `TORCH_LOGS`
artifact observations from the 16-slot profile, for what they grade:

- Dynamic strategy: 8 recompiles → **9 artifacts** over the 16-slot
  vocabulary sweep plus 200 live steps (prediction 2 said ~2-4; the extra
  artifacts are the expected size-0/1 specializations of the width, batch,
  kv-span, and write-map dims before each goes symbolic).
- **Warm-up does not cover the traffic guard set:** the sweep's all-filler
  metas have length-0 `kv_write_map` columns, and 0/1-sized dims
  specialize, so the first real-traffic steps (map length ≥ 1) trigger
  fresh compiles the sweep cannot prevent (observed: `kv_write_map.off
  size mismatch. expected 0, actual 1` then `actual 64`, promoting to
  symbolic). Once the pool fix lands, the sweep wants a real-rows warm
  pass (or seeded maps, as capture already does) before the zero-
  recompiles-during-traffic gate can hold.

## §6 predictions, graded (kept, not edited)

1. Hoisted forward traces to one graph, fullgraph holds, sdpa pin fine:
   **confirmed on CUDA** (no break anywhere in suite, probe, or profile).
2. ~2-4 artifacts dynamic, zero traffic recompiles: **missed on both
   halves, benignly on the first** — 9 artifacts (specialization
   stepping stones); traffic recompiles occur by construction via the
   filler-meta warm-up gap above.
3. Kernels/step ~1750 → 400-700: **landed below the range (218), but the
   number is hollow** — fusion succeeded and then functionalization spent
   the win on ~13 pool-scale copy kernels per step.
4. 16-row decode 5.93 → 4.6-5.4 ms, 1-row 3.63 → 2.4-3.1: **refuted,
   catastrophically** — ~213 ms at serve geometry, 23x *slower* than
   the un-graphed eager step, from the §4 functionalization hazard.
5. short_chat c=16 +8-24%, longctx +8-23%: **not measured** (A/B aborted).
6. Strategies within 5%: **not measured**.
7. Cold bill 40-150 s / warm under 20 s: **not protocol-measured**; one
   data point: the serve-geometry probe paid 66 s cold for its first
   3 artifacts, consistent with the prediction's range.
8. Greedy equivalence / logits tolerance / replay bit-exact vs
   compiled-eager: **not reached on-box**; the CPU suite's compiled-vs-
   eager equivalence still holds (455 green).

## Gates (§7)

Not evaluated — the blocker fails them all trivially. Compile stays
opt-in, default off, pending the pool-layout decision.

# The H100 day: results (2026-08-14/15, Tokyo)

The Phase-3 close-out measurement session from `h100-plan.md`, run on a
p5.4xlarge (1x H100 80 GB SXM, 16 vCPU, 256 GiB RAM) in ap-northeast-1c.
Verdict up front: **the 32B program ran end to end and the stack's win
did not shrink: the full CUDA serve default beats the everything-off
eager engine by +159-191% at 32B, the same ~2.6-2.9x as at 0.6B, because
the p5's server-class host CPU dispatches kernels so slowly (~6 µs per
launch, 2.5-4x the 5090's Ryzen) that even a ~20 ms GPU decode step
drowns under eager's 4267-launch flood. At 0.6B the 5090 beats the H100
nearly everywhere; production 32B decode lands 27.5 ms against the
19.6 ms bytes-over-bandwidth floor (2.38 TB/s achieved, 71% of spec);
the knee sits at 1.5-2.0 rps; the webchat window worked. Ten of the
twelve predictions missed in at least one clause, and almost every miss
is the same finding: the host CPU, not the GPU, sets the terms.**

Environment: torch 2.10.0+cu128 (sm_90 via the stock cu128 wheels),
driver 595.91.07 / CUDA 13.2 (DLAMI 20260814), Qwen3-0.6B and
Qwen3-32B bf16. Runs: `bench/history/2026-08-15T*_nogit_*` (seven dirs;
`ab-h100-compile/agent-summary.md` is the session log). The `nogit`
stamp is an artifact of running from an rsynced tree with no `.git`;
the engine code is pushed main `7848c6d`, with docs-only local edits.

## 1. Getting a machine at all

The quota saga is its own lesson, recorded in `infra/README.md`:
on-demand p5.4xlarge is sold only in London, Mumbai, Jakarta, Tokyo,
and Sao Paulo. us-west-2 carries the type for Capacity Blocks only, so
the granted us-west-2 quota bought nothing but
`InsufficientInstanceCapacity`, forever, in all four AZs. Tokyo
(ap-northeast-1c is the single AZ with the type) refused a weekday
midday launch, then handed over an instance on the first try on a
Saturday morning JST. The Tokyo pool is saturated in a measurable way:
its spot price sits pinned at exactly the on-demand $8.60/hr around the
clock. Weekend timing appears to be the practical unlock.

Bring-up itself graded clean (prediction 1): suite green on the first
attempt (448 passed, 15 environment skips, 107.8 s), all CUDA
tripwires ran, and the 32B download finished in ~3 minutes at
~350-500 MB/s onto the local NVMe (prediction 7's 4-10 min band,
beaten). The `model_data` NVMe symlink before any download remains
mandatory; nothing else about the box needed touching.

## 2. Tier 1: the 0.6B cross-hardware anchor

Exact re-runs of the 5090 compile round's configs. Medians of 3;
the baseline pair is the cross-product's deliberate repeat-noise
reference and agreed within 0.2% everywhere.

| cell | H100 baseline | H100 +compile dyn | 5090 baseline | 5090 dyn |
|---|---:|---:|---:|---:|
| short_chat c=4 tok/s | 616.9 / 614.4 | 1178.7 (+91%) | 810.7 | 1307.6 |
| short_chat c=16 tok/s | 1855.0 / 1841.0 | **3124.8 (+69%)** | 2461.2 | 3682.6 |
| code c=8 tok/s | 988.3 / 975.6 | 1790.5 (+82%) | 1365.6 | 2097.1 |
| multi_turn c=8 tok/s | 850.7 / 850.3 | 1545.4 (+82%) | 1149.1 | 1698.3 |
| long_context c=1 | 110.0 | 221.5 (+101%) | 179.4 | 294.2 |
| long_context c=2 | 147.0 | 277.2 (+89%) | 207.6 | 285.1 |
| long_context c=4 | 180.7 / 182.3 | **321.1 / 318.7 (+77%)** | 207.7 | 254.7 |
| TTFT p50 (short c=16) | 100.5 ms | 59.2 ms | 66.2 ms | 40.2 ms |

The 5090 wins nearly every absolute cell (prediction 2, confirmed:
small-model decode is a clocks-and-dispatch game and the consumer card
has both). The one H100 win is long_context c=4 under compile, 321 vs
255 (+26%): the first cell of the day where per-kernel work (four long
KV histories per decode step, larger effective batch) is big enough to
run the H100 in its throughput regime instead of its latency regime.
Meanwhile compile's *relative* win is larger on Hopper in every cell
(+69% vs +49.6% at short c=16; +101% vs +64% at longctx c=1), so
PLAN.md's original "compile has more headroom on Hopper" claim reads
true in relative terms while losing in absolute ones. The reason is
unflattering: the eager starting point is worse here (§3), so erasing
overhead buys more.

## 3. The finding under everything: the host CPU

`profile_step` on the 0.6B, eager (flagless, no buckets, no graphs):
the 16-row decode step is **37.2 ms, of which 35.4 ms is `fwd_call`**,
the CPU time for the forward to *return*: Python dispatch plus kernel
launches, flat from 1 to 48 rows, with `fwd_sync` at 0.1 ms. The GPU
finishes everything faster than the CPU can ask. Launches per step are
model-shaped as predicted (1915 vs the 5090's 1859), but each costs
~5.9 µs of this host's CPU against the Ryzen's ~2-3 µs. The compiled
scratch arm (buckets + warm-up + compile, graphs off, the compile
round's protocol) cuts the census to 246 kernels/step (5090: 218,
prediction 4 confirmed, with sm90/Hopper kernel names in the tables)
and the step to 17.3 ms, of which 15.4 ms is still CPU.

The same tax priced every bill of the day at roughly 2.5-3x the 5090
host (prediction 5, missed on magnitude):

| bill | H100 (EPYC host) | 5090 (Ryzen host) |
|---|---:|---:|
| 0.6B baseline Ready (sweep behind it) | 203.7 s | 78.1 s |
| 0.6B compile-dynamic Ready, cold | 646.7 s | 225.6 s |
| 0.6B longctx compiled Ready, cold | 725.2 s | (not directly comparable) |
| 32B default Ready, cold | 1476 s (sweep+compile 1244.1 + capture 9.5) | n/a |
| 32B default Ready, warm cache | 479 s | n/a |

One casualty: the 0.6B compile-batch-bucket arm (which compiles ~2x
the artifacts) never reached Ready inside the config's 900 s health
timeout and lost all four of its points. The longctx config's timeout
was raised to 1800 s on the node before its compiled arm ran (725 s:
it would have been a second squeaker), and both H100 0.6B configs now
carry 1800 s in the repo. The batch-bucket H100 re-run was descoped:
the 5090 round already answered the strategy question (within ~3%,
dynamic default), and the redo would have cost the webchat window.

Also for the record: about a third of the cells carry a repeat-variance
warning on TTFT p50 (CV 5.1-14.3% against the 5% bar; two longctx c=4
cells also flag aggregate CV ~7%). The 5090 rounds had zero such
warnings. Aggregate-throughput medians are unaffected; treat H100 TTFT
numbers as noisier than the 5090's.

## 4. Tier 2: the 32B headline

The largest dense Qwen3 the engine supports, first time hosted: 65.5 GB
of bf16 weights, 8x4096 decode geometry (73.4 GB peak of the 80 GB, no
OOM-ladder step needed, prediction 6 confirmed;
`expandable_segments:True` from the first boot). Full serve default vs
everything-off eager, paired configs:

| cell | eager | default stack | win |
|---|---:|---:|---:|
| short_chat c=4 | 50.1 | 146.0 | +191% |
| short_chat c=8 | 97.0 | **280.2** | +189% |
| code c=8 | 94.4 | 244.5 | +159% |
| long_context c=1 | 11.4 | 30.9 | +171% |
| long_context c=2 | 15.3 | 44.4 | +190% |

Prediction 9 expected the win to shrink to +30-70% because a 32B decode
step is ~20 ms of genuine GPU work that no dispatch fix touches. It
missed in the stack's favor, and `profile_step` at 32B says exactly
why: the eager 8-row step is **78.3 ms with 76.7 ms of CPU dispatch**
across 4267 launches (the ~4200 launch-flood clause of prediction 8,
confirmed within 2%). The GPU's ~20 ms hides completely under the
launch stream; eager 32B on this host measures the CPU, not the model.
Compiled-without-graphs still spends 33.9 ms of CPU over 538 launches.
Only with graphs replaying the step does the measurement flip to the
GPU: production step_dur p50 **27.5 ms** at occupancy 0.98 (p99 41 ms
on prefill-mixed steps, engine ITL p50 27.6 ms).

Against the computable floor: 65.5 GB / 3.35 TB/s = 19.6 ms; measured
27.5 ms is +40% (prediction 8 wanted within 25%), an achieved
**2.38 TB/s, 71% of spec bandwidth**, under the predicted 2.7-3.3.
The gap is attention + sampling + the imperfect overlap of a real
serving step, and it is the number the quantization chapter's
bytes-over-bandwidth lesson now gets to cite from this repo's own
hardware run.

The absolute-throughput clause of prediction 9 (260-380 tok/s at
short_chat c=8) landed at 280.2, and the longctx clause (35-55 at c=1)
near-missed low at 30.9. TTFT p50, default arm: 125 ms (short c=4),
174 ms (c=8), 330 ms (longctx c=1).

## 5. The knee, and the webchat

Open-loop Poisson arrivals, 128-token outputs, repeats 2:

| rps | agg tok/s | TTFT p50 | p90 | p99 |
|---:|---:|---:|---:|---:|
| 1.0 | 112.6 | 0.12 s | 0.19 | 0.76 |
| 1.5 | 166.2 | 0.15 s | 0.82 | 2.03 |
| 2.0 | 215.7 | 1.42 s | 2.92 | 3.67 |
| 2.5 | 233.7 | 4.82 s | 8.34 | 9.12 |
| 3.0 | 238.0 | 7.59 s | 13.74 | 14.55 |
| 4.0 | 240.3 | 11.47 s | 20.73 | 22.40 |

The textbook shape holds (p99 detonates at 1.5 rps, p50 follows at
2.0), but the knee sits at **1.5-2.0 rps**, left of prediction 10's
2.0-3.0, because saturation is ~240 tok/s rather than the ~400 the
prediction derived from the floor: the real 27.5 ms step and the
prefill share price the ceiling, and 240/128 ≈ 1.9 rps is where the
queue must diverge.

The webchat window worked (prediction 11): a fresh default-stack 32B
server (warm Ready 479 s), an SSH tunnel to the Mac, and a coherent
Qwen3-32B thinking-model conversation through this repo's engine with
every Phase-3 piece engaged. Single-stream decode at ~36 tok/s, one
row of a 27.5 ms step.

## 6. Validity gates

- Bench validity warnings: **not zero** (the TTFT CV warnings of §3),
  disclosed above; no other warning class fired.
- Recompiles after Ready on the tripwired 32B default arm
  (`TORCH_LOGS=recompiles`): **zero lines**.
- Pure-decode graph replay (`fwd_width == 1` steps, from
  `engine_steps`): **100.00% in all four graph-enabled runs**
  (0.6B: 21,715 and 25,920 steps; 32B: 5,790 and 6,020), beating the
  100% / >95% gate pair.

## 7. §4 predictions, graded (kept verbatim in h100-plan.md)

1. Bring-up clean, suite green first attempt: **confirmed**.
2. 0.6B decode, H100 does not clearly beat the 5090, dynamic c=16 in
   2900-3700: **confirmed** (3124.8; the 5090 holds by 15%).
3. 0.6B prefill/long-context favor the H100 (longctx c=1 310-420,
   TTFT c=16 28-42 ms): **missed on both clauses** (221.5, and the
   5090 wins the cell; 59.2 ms), with the consolation that the H100
   does win longctx c=4 compiled, 321 vs 255.
4. Census model-shaped ~218/step with Hopper names: **confirmed**
   (246, sm90/Hopper kernels present; eager 1915 vs 1859).
5. Ready bills same-order, slightly worse (cold dynamic 180-300 s,
   warm +20-45 s): **missed on magnitude**: 646.7 s cold, and every
   CPU-side bill ran 2.5-3x the 5090 host's.
6. 32B fits at 8x4096, peak 72-78 GB: **confirmed** (73.4 GB, ladder
   untouched).
7. Download 4-10 min; default cold Ready 6-12 min; eager Ready
   2-4 min: **beaten / missed / confirmed** (~3 min; 24.6 min; ~4 min).
8. 32B default step 20-24 ms within 25% of floor, 2.7-3.3 TB/s; eager
   26-45 ms, ~4200 kernels/step: **missed / missed / missed /
   confirmed** (27.5 ms, +40%; 2.38 TB/s; eager 78.3 ms, the launch
   flood on a slow host; 4267).
9. Stack win shrinks to +30-70% at 32B; absolute 260-380 short c=8,
   35-55 longctx c=1: **missed in the stack's favor** (+159-191%) /
   **confirmed** (280.2) / **near-miss low** (30.9).
10. Knee at 2.0-3.0 rps with the p99-then-p50 shape: **missed low /
    shape confirmed** (1.5-2.0 rps; saturation 240 tok/s).
11. Webchat works: **confirmed**.
12. Session under $35: **missed** (~4 h 55 m at $8.60 ≈ $42; the
    capacity nights added nothing, the compile bills added ~25 min of
    metered CPU time, and the hard stop was never threatened).

## 8. Decisions and residue

- **No flag flips**, as planned: the session decides nothing about the
  serve default, and the default performed as shipped on new hardware
  and a new model size without an engine change.
- The two 0.6B H100 configs now pin `health_timeout_s = 1800` (the
  batch-bucket lesson); the 32B configs already carried it.
- `infra/down.sh` takes `REGION` now: run with the default it
  refreshes against us-west-2, drops the Tokyo instance from state,
  and reports success while the meter runs. The teardown verify step
  (`aws ec2 describe-instances`) caught it; the instance was killed
  via the CLI and the script fixed.
- Deferred/descoped: the batch-bucket H100 arm (strategy question
  already answered on the 5090); tok/s-per-dollar beyond the obvious
  (3125 tok/s/$8.60 vs the 5090's 3683 on a ~$2.3k card makes the
  point without a table); vLLM side-by-side (unchanged from the plan,
  a future H100 day).
- The walk-through lesson of the day, one sentence: **on server-class
  hosts the CPU is slower relative to the GPU than on any desktop, so
  dispatch-erasure (graphs above all) is not an optimization tier, it
  is the difference between measuring the CPU and measuring the GPU**,
  and the bigger the model, the more the eager number lies about what
  the hardware can do.

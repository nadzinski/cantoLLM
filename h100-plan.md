# Session plan: the H100 day (Phase 3 close-out)

**Status (2026-08-15):** Complete. The session ran 2026-08-14/15 (a
Saturday morning JST: the weekday Tokyo pool refused, the weekend pool
gave an instance on the first try). `h100-results.md` is the record;
`bench/history/2026-08-15T*` are the runs, with the session log in
`ab-h100-compile/agent-summary.md`; §4 below is graded verbatim in the
results doc. ~4 h 55 m ≈ $42. Replan history follows.
Replanned to Tokyo (2026-08-11). The us-west-2 quota
(`f6e561bb…`) was approved, but the launch attempt hit
`InsufficientInstanceCapacity` in all four AZs, and the investigation
found why it always will: on-demand p5.4xlarge is sold only in London,
Mumbai, Jakarta, Tokyo, and São Paulo; us-west-2 carries the type for
**Capacity Blocks for ML only**, so its on-demand pool is effectively
empty and the vCPU quota there buys nothing. Pivot: **ap-northeast-1
(Tokyo)**, on-demand $8.60/hr, and exactly one AZ carries the type
(**ap-northeast-1c**: pin it, there is no AZ walk). New quota ticket
PENDING (request `d690a3ee…`, 16 vCPUs of `L-417A185B` in
ap-northeast-1, filed 2026-08-11 evening PT; both prior tickets took
about a day). Pre-verified in Tokyo: DLAMI (20260724 build) and the
default-VPC subnet in 1c. Bring-up uses a fresh terraform workspace
(`terraform workspace new tokyo`) so the us-west-2 key-pair/SG state
stays intact. Budget at the Tokyo rate: ~$30-39 for the 3.5-4.5 h
session, hard stop ~$52. Prep is committed: the 32B spec entry, seven
bench configs, the infra H100 profile, this doc.
**Update 2026-08-13:** Tokyo quota approved and enforced (16). First
launch attempt (PT evening = midday JST) hit
`InsufficientInstanceCapacity` through a ~15 min retry window and was
cancelled; the 1c spot feed sits pinned at exactly the on-demand
$8.60/hr around the clock, i.e. the pool is saturated and launches
ride on-demand's priority over spot when something frees. Plan:
attempt at a **PT morning (~8 AM PT = midnight JST)**, Tokyo's
off-peak. The tokyo workspace already holds the key pair + SG (free),
so the next apply creates only the instance.

This is the last open item of Phase 3. Unlike the sdpa/graphs/compile
rounds it decides no flag flips: it is a measurement session. The house
ritual still applies where it earns its keep: predictions on record
(§4), graded in the results doc.

## 1. Goals

Three tiers, in run order:

1. **0.6B cross-hardware anchor.** Re-run the compile A/B unchanged
   (`ab_h100_compile{,_longctx}.toml`, exact copies of the 5090 configs)
   plus a `profile_step` pass. Every cell gets a direct 5090 sibling:
   that is the H100-vs-5090 read at every tier of the stack, and the
   answer to PLAN.md's original "compile has more headroom on Hopper"
   claim.
2. **32B, the headline.** The largest dense Qwen3 the engine supports
   (spec entry landed with this plan; ~65.5 GB bf16 weights, H100-only).
   Two-arm A/B: full CUDA serve default vs everything-off eager
   (`ab_h100_32b_{default,eager}.toml` + longctx pair). The stack's 0.6B
   wins came mostly from erasing per-step CPU overhead; a 32B decode
   step is ~20 ms of genuine GPU work, so the lesson is how much of the
   win survives when compute dominates. Plus `profile_step` against the
   computable floor: 65.5 GB / 3.35 TB/s = **19.6 ms** per decode step,
   the bytes-over-bandwidth number the quantization chapter teaches.
3. **14B fallback.** Only if 32B fails its OOM ladder (§5). Same
   configs with `model = "14B"`; still a model the 5090 cannot host.

Extras riding along (all cheap): tok/s-per-dollar table, Ready bills on
the server-class host CPU as a second data point (the compile round
showed these bills are CPU-bound), `nvidia-smi` samples during runs,
a Hopper kernel-census diff (same model, 5090 vs H100 kernel names;
analysis happens back home), and the **32B open-loop knee**
(`knee_h100_32b_openloop.toml`), the one purchased extra.

And the fun one: **webchat with the 32B**, served end to end by this
repo's engine with every Phase-3 piece engaged, through an SSH tunnel.
Ten minutes, strictly after the recorded runs.

## 2. The box and the bills

**p5.4xlarge**: 1x H100 80 GB (SXM), 16 vCPU, 256 GiB RAM, 3.8 TB local
NVMe, up to 100 Gbps network. On-demand $8.60/hr in ap-northeast-1
(Tokyo; us-west-2 sells this type via Capacity Blocks only, see
Status). Invocation (details in `infra/README.md`):

```
terraform workspace new tokyo   # once; keeps us-west-2 state intact
REGION=ap-northeast-1 AZ=ap-northeast-1c INSTANCE_TYPE=p5.4xlarge ROOT_VOLUME_GB=150 ./up.sh
```

Weights live on the NVMe via the `model_data` symlink (the repo
downloads into the working tree, whose root EBS volume reads at gp3's
125 MB/s baseline; the symlink step in the README is mandatory, before
anything downloads). After the first read the 256 GiB of RAM page-caches
the safetensors, so per-arm server respawns reload fast.

32B memory budget at the two geometries:

| piece | decode config | longctx config |
|---|---:|---:|
| weights (bf16) | 65.5 GB | 65.5 GB |
| KV pool (256 KB/token) | 8 x 4096 = 8.6 GB | 2 x 10240 = 5.2 GB |
| activations + workspaces + capture pool | ~2-4 GB | ~2-4 GB |
| total vs 80 GB | ~77 GB | ~73 GB |

That is 92%+ occupancy, so `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
is set from the first server, not after the first OOM: at this pressure,
fragmentation is the thing that lies about what fits.

**Session budget:** ~3.5-4.5 hours ≈ $30-39 at the Tokyo rate. Hard
stop at 6 hours / ~$52 unless Nadia extends it live. `./down.sh` is the last command,
verified with `aws ec2 describe-instances`, every time we walk away.

## 3. Runbook

Nadia's standing instruction: Claude drives from the Mac, she checks in
when she likes, nothing waits on her except the webchat window (and even
that has a timeout, step 9).

| # | step | est |
|---|---|---|
| 0 | Pre-flight (Mac): quota approved (`aws service-quotas get-service-quota --service-code ec2 --quota-code L-417A185B --region ap-northeast-1`), tree clean, this plan's prep commits pushed | before the day |
| 1 | `terraform workspace new tokyo` (once), then `REGION=ap-northeast-1 AZ=ap-northeast-1c INSTANCE_TYPE=p5.4xlarge ROOT_VOLUME_GB=150 ./up.sh`; on `InsufficientInstanceCapacity`, retry over ~30 min (1c is the only AZ with the type), else reschedule | 10 min |
| 2 | `./sync.sh`; `./ssh.sh nvidia-smi` (expect H100 80GB, driver from the DLAMI); node: tmux, the `model_data` symlink + `HF_HOME` export, `uv sync` | 15 min |
| 3 | **Kick off the 32B download in a background tmux window immediately** (`snapshot_download` via a one-liner; ~65 GB, predicted 4-10 min on the 100 Gbps NIC); it overlaps step 4-5 | 0 (overlapped) |
| 4 | Suite on the node: `python -m pytest tests/ -v` including the CUDA-marked tripwires | 10 min |
| 5 | 0.6B: `canto bench run bench/configs/ab_h100_compile.toml`, then `ab_h100_compile_longctx.toml`; `profile_step` eager + stack arms per the compile round's pattern (flagless script = eager; scratch copy with buckets+warmup, graphs off, compile on for the kernel census) | 60 min |
| 6 | 32B smoke: default-stack `canto serve --model 32B` with the A/B geometry, record the first Ready bill cold, one curl completion, kill. This is the OOM-ladder gate (§5) | 15 min |
| 7 | 32B A/B: `ab_h100_32b_default`, `ab_h100_32b_eager`, then the longctx pair. `TORCH_LOGS=recompiles` on one default-arm spawn as the recompile tripwire | 60-75 min |
| 8 | 32B `profile_step`, default + eager arms: step decomposition vs the 19.6 ms floor, kernel census, achieved bandwidth | 10 min |
| 9 | Ping Nadia (push notification) with the webchat ETA; run the knee sweep (`knee_h100_32b_openloop.toml`) | 25 min |
| 10 | Webchat window: spin the chat server (default stack, fresh Ready bill), `ssh -L` tunnel, `canto webchat` on the Mac against it, second ping "tunnel is up". While it warms: rsync `bench/history/` + logs home. If she has not appeared ~40 min after the ping, skip: down, and offer a cheap second window another day (one Ready bill, ~$2) | 20 min |
| 11 | Final rsync of anything new, `./down.sh`, verify terminated | 10 min |

Interactive traffic never touches a recorded run: the webchat server is
its own spawn, after all benches.

## 4. Predictions on record

To be graded in `h100-results.md`, kept not edited:

1. **Bring-up is clean**: the cu128 wheels run sm_90 unchanged, full
   suite green on the node on the first attempt.
2. **0.6B decode: the H100 does not clearly beat the 5090.** short_chat
   c=16 compile-dynamic lands 2900-3700 tok/s (5090: 3683). Small-model
   decode under graphs is many small kernels where the 5090's clocks
   fight the H100's width; I expect a narrow 5090 hold or a tie.
3. **0.6B prefill/long-context favor the H100**: long_context c=1
   294 -> 310-420 tok/s; TTFT p50 short_chat c=16 in 28-42 ms (5090:
   40.2). Big matmuls exploit the ~2.4x dense bf16 advantage.
4. **Kernel census structure is model-shaped, not hardware-shaped**:
   ~218 kernels/step at 0.6B under compile, but Hopper-specific names
   appear (wgmma/TMA-flavored GEMMs, sm90 cuDNN attention).
5. **Ready bills on the EPYC host are same-order, slightly worse**:
   0.6B cold compile-dynamic 180-300 s (5090 host: 225.6), warm +20-45 s
   (5090: +21).
6. **32B fits at 8 x 4096**: default arm boots without touching the OOM
   ladder, peak 72-78 GB.
7. **32B bills**: download 4-10 min; default-arm cold Ready 6-12 min;
   eager-arm Ready (pure load) 2-4 min.
8. **32B decode step (8 rows, default arm) 20-24 ms**, within 25% of
   the 19.6 ms floor, achieved bandwidth 2.7-3.3 TB/s. Eager arm
   26-45 ms: the launch flood roughly doubles with 64 layers (~4200
   kernels/step) and the server CPU dispatches no faster than the
   Ryzen did.
9. **The stack's win shrinks by half or more at 32B**: default beats
   eager by +30-70% on short_chat c=8 (the 0.6B equivalent was ~+150%
   from the post-sink-fix eager engine). Absolute: default short_chat
   c=8 at 260-380 tok/s; longctx c=1 at 35-55 tok/s.
10. **The 32B knee sits at 2.0-3.0 rps** at 128-token outputs, with the
    same p99-then-p50 TTFT blow-up shape as the 0.6B knee.
11. **Webchat works**: a coherent 32B thinking-model conversation
    through the tunnel, no engine changes.
12. **The whole session bills under $35.**

## 5. Gates and contingencies

- **Validity**: zero bench validity warnings in every recorded cell;
  zero post-Ready recompiles on the tripwired arm; decode-graph replay
  100% at 0.6B and >95% at 32B.
- **OOM ladder (32B)**: 8 slots -> 6 -> 4 (edit `max_batch` on the node,
  record the deviation in the run notes); if 4 x 4096 still cannot
  boot, fall back to 14B for the whole tier-2 program and say so in the
  results doc. Longctx ladder: 2 x 10240 -> 1 x 10240 -> drop the
  longctx pair.
- **Capacity ladder**: ap-northeast-1c is the only AZ carrying the
  type, so there is no walk: on `InsufficientInstanceCapacity`, retry
  over ~30 min, then stop and reschedule. No 8-GPU fallback:
  p5.48xlarge is ~$55/hr and answers no extra question.
- **Quota stall**: if the ticket is still PENDING after ~2 business
  days, open a support case referencing it.
- **Time**: hard stop at 6 hours. If the 0.6B tier overruns badly,
  the knee is the first cut, then the longctx 32B pair; the 32B
  short_chat A/B and the floor measurement are the last things cut.
- **Money**: `./down.sh` verified at every walk-away, no exceptions.

## 6. Deliverables

- `h100-results.md`: the canonical record, house style, predictions
  graded (§4 kept verbatim).
- **viz `#/h100` tab** (pre-approved 2026-08-09): results and
  conclusions in the chapter kit's look and feel (serif article, def and
  aside boxes, framed figures) but explicitly **not** a chapter: no
  predict gates, no quiz, no exercises. Charts follow the dataviz
  conventions, light mode. `h100-results.md` stays canonical; the tab
  presents it.
- PLAN.md Phase 3 Status -> **Complete** (this is the phase's last open
  item), with the Status-line date fixed; viz Roadmap statusLine synced
  to match (both pre-approved 2026-08-09).
- `bench/history/` runs committed per house convention.
- The `agent-summary.md` habit continues even though Claude drives from
  the Mac: a session log lands next to the runs.

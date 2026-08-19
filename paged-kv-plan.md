# Phase 4: paged KV + preemption + priority + overlap (design + plan)

The merged design note and execution plan for Phase 4, decided 2026-08-18 in a
planning session (decisions the author's; scaffolding and delegated chunks
implemented by Claude in review-sized commits; three chunks hand-written by the
author, red→green against pre-landed suites). PLAN.md's Phase 4 section carries
the forward-looking scope; this doc records the decisions, the alternatives
considered, the architecture, the execution order, and (in §10, as rounds
complete) the validated results with §6's predictions graded.
`flex-spike-results.md` is this doc's gates annex: the kernel route was decided
there, on the 5090, before any of this was designed.

**Status: designed 2026-08-18. Implementation not started; next is chunk 1
(§5).**

## 1. Goal

Four lessons, one phase:

- **Memory management.** The padded pool reserves `max_seq_len + 1` positions
  per slot whether a request uses 40 tokens or 4000. At the standard bench
  geometry (16 slots, 4096 capacity) a short_chat request occupying ~170 tokens
  strands ~96% of its slot. Paging replaces the reservation with fixed-size
  blocks (default 16 tokens) allocated on demand: KV memory tracks actual
  usage, capacity and concurrency decouple, and the same VRAM admits several
  times the requests.
- **The metadata-driven read path.** `F.scaled_dot_product_attention` takes no
  block tables, so the attention call moves to FlexAttention, where raggedness
  and paging are expressed as lengths and index tensors. This is the
  flash-proper restructure deferred from Phase 3 (sdpa-results.md: the
  explicit-mask compromise), landing where the paged pool forces it anyway.
- **Scheduling under scarcity.** With memory allocatable it becomes exhaustible
  on purpose: preemption (evict a victim, recompute on resume), victim
  policies, and per-request priority, judged by a goodput-under-joint-SLO
  metric rather than aggregate tok/s, which barely sees them.
- **Overlap.** The scheduler stops blocking on the forward pass: plan step N+1
  while step N runs, keep sampled tokens GPU-resident as the next step's
  input, move the D2H copy to a side stream. This attacks the same
  CPU-dispatch floor Phase 3 profiled, from the scheduling side.

## 2. Decisions and alternatives

**2.1 FlexAttention first; flash-attn stays a deferred A/B arm.** Decided by
the spike (flex-spike-results.md): all four gates passed, and flash-attn has no
torch-2.10/sm_120 wheel and requires 256-token pages on its paged entry point,
which would forfeit most of the small-block story. The block allocator keeps
block size configurable so the flash arm stays open. Flex makes the author
write the paged index translation herself, which is the educational core of
the phase.

**2.2 The padded stack is the control and stays a separate class.** Padded +
sdpa + buckets + warm-up + graphs + compile remains the always-runnable CUDA
default until the paged stack beats it, and the Mac/CPU default indefinitely.
`FlexAttentionMethod` does not subclass `PaddedAttentionMethod`: the template's
shared mechanics (the 2D `(slot, pos)` scatter, the `[:max_history_len]`
gather, the dense bool mask) are exactly what paging replaces, so inheritance
would share nothing but bugs. Rejected: refactoring padded into a
`block_size = max_seq_len` special case of the paged pool. Unifying them is
tempting and maybe right eventually, but rebuilding the control arm in the
middle of the experiment is how you lose the control; revisit at close-out
(§5 chunk 13) if flex wins outright.

**2.3 The paged pool is flat per-layer tensors; the write map gains a flat
destination column.** Each layer stores K and V as one tensor of shape
`((num_kv_blocks + 1) * block_size, num_groups, head_dim)`: a block is a
16-row span, and the KV scatter writes the base tensor directly through a new
3-column `PagedKVWriteMap(row, off, dst)` with
`dst = table[pos // block_size] * block_size + pos % block_size`. Rejected: a
`(num_blocks, block_size, G, D)` blocked layout (the scatter would write
through a view, which is the exact functionalization failure ab4f438 fixed;
kv_pool.py's docstring is the standing warning), and reusing the padded
`KVWriteMap`'s `(slot, pos)` columns to mean `(block, offset)` (saves nothing,
forces 2D indexing into a layout we don't want, and would entangle the padded
arm's tested invariants). The read side may view the flat tensor however Flex
wants: reads through views are safe; only writes through views are poison.

**2.4 The last block is the scratch block.** Index `num_kv_blocks`, appended
past the allocatable range, never in the free list: the paged analog of the
padded pool's `scratch_pos` column. Filler rows and warm-up writes park there,
and every filler row's table points at it with `kv_num_blocks = 1`, because a
row with zero visible KV is a fully-masked softmax row, which is NaN.

**2.5 Block tables live with the scheduler; `BatchMeta` carries seeded
references.** A scheduler-owned `PagedStepState` (built by
`scheduler_from_runtime`, living beside the allocator) owns the persistent
device-resident int32 tensors (`block_tables`, `kv_num_blocks`, and the
physical→logical inverse the mask needs) plus the per-family `BlockMask`
objects built over them once. Per step, the scheduler writes ints into those
tensors in place and `BatchMeta` carries references via a
`seed_paged_tables(...)` that mirrors `seed_kv_write_map` exactly: seed-once
discipline, and survival through `runtime.forward_batched`'s device-move
`replace()` (the 40fbcf9 family). The pool stays memory-only; kv_pool.py's
docstring promised precisely this split in Phase 2 and it holds. Rejected:
pool-owned tables (the pool would learn about scheduling) and per-step tensor
construction (allocation in the loop, and graph capture bakes in addresses,
so the tensors must be permanent).

**2.6 KV length is a value, not a shape.** The spike's gate 3 is the design's
level: table tensors are allocated once at maximum width, `kv_num_blocks` is
mutated in place, and the same BlockMask through the same compiled callable
computes the truncated result. Consequences, all structural: the kv axis
leaves shape space, the shape vocabulary collapses from (batch × width × kv)
to (batch × width) (~315 → ~20 shapes at the standard bench geometry; exact
counts measured at chunk 6), decode graphs drop from |B|×|KV| (~80) to |B|
(~5), and `kv_bucket` is inert on the paged kernel path. `create_block_mask`
and `from_kv_blocks` never run in the step loop; per-step mask work is
writing ints into preallocated tensors, which is what the scheduler already
does with the KV write map.

**2.7 Capacity knobs decouple.** `num_kv_blocks` sizes memory,
`max_batch` sizes concurrency (rows per step), `max_seq_len` becomes the
per-request logical capacity (block-table length bound, and still the
admission cap `registry.register` derives). Default `num_kv_blocks = None`
means parity capacity, `max_batch * ceil(max_seq_len / block_size)`, at which
exhaustion is impossible and the paged engine is a drop-in; benches
undercommit explicitly to make scarcity happen. Rejected: defaulting to a
VRAM-fraction sizing like vLLM's `gpu_memory_utilization` (right for
production, wrong for a phase whose A/Bs need capacity as a controlled
variable; a fraction-based default can land with the serve polish later).

**2.8 torch.compile is required for the paged path on CUDA.** Flex only
performs compiled (spike §7.4): the compile and attention stacks merge, and
graph capture records Inductor-generated Flex kernels. Enforced at serve
assembly in `main.py` and defensively in `scheduler_from_runtime`; CPU stays
compile-optional so the eager-Flex oracle suite runs everywhere. The cuDNN
pin is sdpa-only: `FlexAttentionMethod.execution_context()` is `nullcontext`.

**2.9 Preemption is whole-request recompute-on-resume.** When blocks run out,
a victim's blocks are freed and the victim re-queues with its known prefix
(prompt + already-emitted tokens); on resume it prefills that prefix and
continues decoding. The client stream is append-only: emitted tokens are
never re-emitted, and under greedy the resumed stream is token-identical to a
never-preempted run, which makes correctness model-free testable. Rejected
for now: partial tail eviction (corrupts the "table covers position"
invariant for marginal gain at these KV sizes) and swap-to-CPU (the stretch
arm; gated on a measured recompute cost from round 3, since swapping only
pays where recompute is expensive).

**2.10 Priority is a body field on both dialects.** `priority: int` (bounded,
default 0, higher wins) on the OpenAI request model (vLLM's de-facto
extension, reachable via `extra_body`) and on the Anthropic request model as
a documented extension. Promotion becomes a stable sort by (priority desc,
arrival), so equal priorities keep today's FCFS exactly. Victim policies:
`lifo` (newest active), `priority` (lowest priority, LIFO tiebreak),
`cost` (fewest KV tokens lost, i.e. cheapest recompute). Rejected: a header
(invisible to SDKs, to request records, and to the OpenAPI surface; can be
added later as an ops-side fallback where the body wins).

**2.11 Goodput is measured client-side.** Goodput = fraction of measured
requests meeting a TTFT SLO and a per-request ITL-tail SLO jointly, SLO pair
configured per bench point. The ITL tail comes from per-chunk client arrival
timestamps (new in `sse_clients.py`): SLOs are client-experienced, and engine
ITL misses mux/API stalls and dies across engine restarts. The engine-side
ITL join (X-Request-ID) stays available as an attribution diagnostic:
when goodput drops, was it scheduling or transport. Open-loop cells are the
ones goodput means anything on (closed-loop under-reports tail latency near
saturation; bench-spec.md §2).

**2.12 Overlap is a flag, orthogonal to layout, and lands last.** The
restructured loop works on both padded and paged arms (it is the same loop);
it lands after preemption so the state machine it restructures is final, and
its A/B is a discrete final round. Finalize (stop tokens, max_tokens, event
emission) runs one step late, gated on a CUDA event from the side-stream
D2H; a row that stopped at step N wasted one decode at N+1 and rolls back:
position decremented, any block allocated for the bonus token freed, no
event emitted. Rejected: landing overlap before preemption (writes the
eviction logic into a loop that is mid-restructure, and the overlap A/B
would run on a mid-phase engine).

## 3. Architecture: the paged step

```
scheduler.step()                                (paged mode)
  _promote_queued()        priority-sorted; allocates first blocks or stops
  _plan_step()             water-fill; RESERVES blocks for start+num_new
                           per row before the row commits; on shortage,
                           shrink the grant, then preempt (chunk 9)
  _build_input_ids()
  build_batch_meta()
  shape_step()             rows/width bucketing unchanged; kv rounding skipped
  PagedStepState.fill()    in-place int writes: block_tables, kv_num_blocks,
                           inverse table, PagedKVWriteMap (fillers → scratch)
  meta.seed_paged_tables() references, never copies
  forward_fn(...)          scatter dst-flat → BlockMask (prebuilt) →
                           flex_attention(enable_gqa=True)   [compiled;
                           graphs replay it after chunk 8]
  sample / finalize        unchanged until overlap (chunk 12) defers it
```

The pool allocates `(num_kv_blocks + 1) * block_size` rows per layer per K/V.
A sequence carries `block_table: list[int]` (host truth); the device tensors
are step-shaped projections of it, filled per step the way the KV write map
already is. `Qwen3._validate_batched`'s `max_history_len <= pool.max_seq_len`
check survives with `max_seq_len` meaning logical capacity.

Preemption inverts admission: `_preempt_one(policy)` picks a victim among
active rows, frees its blocks (and its slot), re-queues it at the front of
its priority class with the concatenated prefix, and emits nothing. Stats
stay coherent because chunk 1 makes the collector consume plan-time counters
instead of deriving from snapshot diffs (stats.py's two dying invariants:
"finish implies past prefill" and "finished rows free slots same-step").

Overlap splits `step()` into launch and reap. Launch: plan (using the
previous step's GPU-resident sampled tokens as decode inputs, scattered
device-side into the next input buffer), forward, sample, enqueue side-stream
D2H, record event. Reap (next iteration, event complete): stop/max_tokens
checks, rollbacks, TokenEvents. `drive_scheduler` and `is_idle` learn about
in-flight steps; abort of an in-flight row defers its block frees to reap.

## 4. Hazards the implementation will meet

Named now so they are recognized as expected, not debugged as surprises:

- **Write-through-view under compile.** Any paged layout or scatter that
  mutates a view of the pool base regresses to ab4f438's pool-scale copies
  (23× steps, OOM). The flat layout exists to prevent this; the Inductor
  CUDA tripwire test extends to the paged pool.
- **`from_kv_blocks` creeping into the step loop.** It allocates, and capture
  bakes addresses. One BlockMask per (batch, width) family over permanent
  tensors; a test counts BlockMask constructions per step (expected: zero).
- **Seeded tables dropped by the meta device move.** `forward_batched`'s
  `replace()` rebuilds `BatchMeta`; the seeded-write-map survival clause must
  cover the paged tensors or warm-up and capture silently un-seed (the
  40fbcf9 family, third appearance).
- **The fully-masked filler row.** A filler with no visible KV is a NaN
  factory (softmax over minus infinity). Every filler points at the scratch
  block with `kv_num_blocks = 1`; its garbage is finite and unread.
- **CPU-eager Flex proves logic, not the kernel.** The CPU oracle validates
  the index translation; the flex-decoding split-KV kernel only exists
  compiled on CUDA. Every CPU-green chunk has a CUDA-skipif twin (compiled
  Flex vs sdpa, bf16 tolerance) plus a Flex-kernel-actually-ran counter: the
  SDPA silent-fallback lesson, generalized.
- **The decode `seq_lengths` wrinkle.** `from_kv_blocks` must get
  `seq_lengths=(q_len, S)` explicitly or q_len=1 raises (spike §2). S is a
  constant per family; if S ever varies per step it is shape churn.
- **Table shapes must stay maximal.** Sizing tables to the step's kv need
  reintroduces per-kv-shape guards and defeats decision 2.6. Lengths are
  values. The recompile counter guards this.
- **Warm-up must take traffic's device path.** Persistent device tables are
  mutated by traffic under inference mode; warm-up reaching them any other
  way splits dispatch-key guards and the first live request recompiles (the
  2026-08-08 ADInplaceOrView lesson).
- **Plan/allocate atomicity.** Block reservation happens inside `_plan_step`
  before a row commits; a step must never launch a row whose writes have no
  destination. Shortage first shrinks the water-fill grant (trim prefill
  chunks before evicting anyone).
- **Preemption vs in-flight state.** A preempted row leaves `active` without
  a finish event (stats), may hold queued client events (mux ordering), and
  under overlap may have a step in flight (defer frees to reap; never
  double-free blocks on abort-during-preemption).
- **Overlap finalize is one step late.** Stop detection lags a step; the
  bonus token's rollback must cover position, block, and event, and
  per-request event order must be provably unchanged (each request's events
  serialize through its own queue).

## 5. Execution order

Chunks are review-sized and land green individually, one commit each, tagged
`(P4 chunk N)`. [HAND] = author-written, red→green against a suite pre-landed
one chunk earlier. Class names below are placeholders until the author
confirms them (§9).

1. **Precondition refactors** (no behavior change; padded/sdpa arms
   byte-identical): `KVPool` structural Protocol over the consumed surface
   and retyped `pool` params (types.py, scheduler.py, graphs.py, warmup.py,
   runtime.py); `BatchingConfig` gains `paged_kv`, `block_size`,
   `num_kv_blocks` (+ validation, incl. paged+CUDA ⇒ torch_compile);
   `new_kv_pool` becomes a layout branch; stats collector switches to
   plan-time counters and `(kv_allocated_tokens, kv_capacity_tokens)`
   (additive schema, STATS_SCHEMA_VERSION bump; bench `kv_fill_mean` prefers
   the capacity field); protocol.py mask types loosen to method-opaque.
2. **Paged pool + [HAND] allocator.** `PagedKVPool` (flat layout, scratch
   block) lands with `tests/test_block_allocator.py` pre-landed; the author
   writes `BlockAllocator` (deque free list, refcounts, double-free guard)
   red→green.
3. **Block tables + BatchMeta extension.** Optional paged fields +
   `seed_paged_tables`; `PagedKVWriteMap`; `PagedStepState` shell. Pre-land
   `tests/test_flex_equivalence.py` (xfail): eager Flex on CPU vs the padded
   oracle (single row, chunked prefill, mixed batch, stale-block reuse,
   block-boundary crossings), write-map vs naive per-token loop.
4. **[HAND] The paged attend.** `models/attention/flex.py` plus the write-map
   destination computation, red→green. Exit: CPU eager green; CUDA-skipif
   twin (compiled Flex vs sdpa on-device) green on the 5090.
5. **Scheduler paged mode.** Promotion/planning allocate and reserve blocks;
   finish/abort free them; `CBSequence.block_table`. Exit: toy-stepper block
   accounting (no leaks), then the engine-level oracle: paged CB engine vs
   padded CB engine, token-for-token, tiny model, CPU greedy.
6. **Warm-up / vocabulary / compile.** Paged vocabulary drops the kv axis;
   warm-up seeds scratch maps + tables (fillers → scratch block);
   `_mark_compile_dims` covers the new tensors. Exit: recompile-counter test
   proves kv-length changes recompile nothing; measured shape/graph counts
   recorded here.
7. **5090 round 1** (§7): `--attention flex` wired; tripwires then A/B.
8. **CUDA graphs on paged decode + 5090 round 2.** Static buffers ARE the
   step tables; graphs keyed (B, 1); `_replayable` gains table bounds checks.
   Exit: bit-exact replay after in-place table permutation (spike gate 4's
   probe), then the full-stack A/B.
9. **[HAND] Preemption.** `tests/test_preemption.py` pre-lands (exhaustion
   triggers eviction; blocks freed; append-only client stream; greedy
   token-identical to an unconstrained run; abort-while-preempted; stats
   coherent). The author writes the evict/resume machine, LIFO first.
10. **Priority + [HAND] policies + goodput.** Claude: the `priority` field
    through both dialects and IPC, priority-sorted promotion, per-chunk
    client timestamps, the goodput metric + SLO config keys, paged KV bench
    fields. The author: `priority` and `cost` victim policies against
    pre-landed policy-comparison tests.
11. **5090 round 3: the goodput round** (§7).
12. **Overlap scheduling** (delegated, like graphs/compile), then **5090
    round 4**. Toy suite for deferred finalize, late-stop rollback,
    abort-in-flight; token-for-token vs the serial scheduler.
13. **Close-out.** H100 scale check; §10 results + §6 graded; PLAN.md status;
    serve.example.toml; the CUDA-default-attention decision (the author's);
    the viz ask (Paged KV tab, Roadmap tab); memory update.

## 6. Predictions (to grade at phase end)

Graded after the 5090 runs; wrong predictions get kept, not edited.
References: padded full stack, 2026-08-16 metrics-on records (short_chat c=16
3595.9 tok/s; long_context c=1 294 via the 2026-08-08 record).

1. **Parity cost is small.** At parity capacity, paged full stack lands
   within −5%..+2% of padded on short_chat c=16 and −8%..+2% on
   long_context c=1. Reasoning: the spike bounds the attention delta at
   ~0.3 ms/step worst case against a ~4 ms step, and Flex reaches cuDNN
   parity by KV 2048 where long-context time actually sits.
2. **The warm bill shrinks.** Sweep shapes ~315 → ~20 and decode graphs
   ~80 → ~5 at the standard geometry (structural, exact counts at chunk 6);
   the paged arm's total warm-up (sweep + capture + warm compile cache)
   comes in under the padded stack's at the same geometry. Cold compile
   stays under 2× padded's cold bill (new Inductor kernels per family, but
   far fewer families).
3. **The capacity headline.** With `num_kv_blocks` sized to what padded's
   16 slots reserve, a paged server at `max_batch = 64` sustains ≥ 48
   concurrent short_chat requests with zero preemptions and reaches ≥ 1.5×
   padded's saturated short_chat aggregate tok/s (decode is
   per-step-overhead-bound at 0.6B; more rows per step amortize it, the
   Phase 2 knee lesson). `kv_fill_mean` on the true block denominator
   reaches ≥ 60% on that cell vs ~4% effective fill for padded slots.
4. **Preemption is correct and priced.** Under a 50%-of-parity pool and
   overload, every request completes with greedy streams token-identical to
   the unconstrained run; a preempted-and-resumed request pays its re-prefill
   at roughly one step (~512-token prefix in one 512-token chunk), so the
   dominant cost is re-queue wait, not compute.
5. **Policies move goodput, not throughput.** Under mixed-priority overload,
   the `priority` victim policy raises high-priority goodput by ≥ 15 points
   over `lifo` while aggregate tok/s stays within ±2% across policies; the
   `cost` policy beats `lifo` on total goodput but not on high-priority
   goodput.
6. **Overlap pays where dispatch does.** Overlap-on improves 5090 short_chat
   c=16 aggregate by +3–10% and decode-cell ITL p50 by 5–15%; on the H100
   32B rerun the gain is larger (server-host dispatch is the floor there:
   27.5 ms steps vs a 19.6 ms bandwidth floor). Event order per request is
   unchanged (gate, not prediction).

## 7. A/B protocol

Every round: predictions and gates written into the run config header before
the run; medians of 3 repeats; records in `bench/history/`; correctness gates
alongside performance gates. Standing correctness gates: greedy token
equivalence vs the appropriate oracle arm, Flex-kernel-ran counter > 0,
recompiles after warm-up = 0, decode replay rate (once graphs land) ≈ 100%,
no NaN events.

- **Round 1 (chunk 7)** `ab_5090_paged{,_longctx}.toml`, three arms so
  attribution is clean: padded full default; padded compile-no-graphs; flex
  compile-no-graphs (graphs don't exist for flex yet). Gate: flex within
  −10% of the padded compile-no-graphs arm on every cell, nothing NaN, warm
  bill recorded.
- **Round 2 (chunk 8)** full stacks head-to-head (padded default vs flex +
  graphs), plus the capacity cells: reduced `num_kv_blocks` at fixed
  `max_batch`, and the wide-slot headline cell (prediction 3). Gate:
  prediction 1's ranges.
- **Round 3 (chunk 11)** `goodput_5090.toml`: open-loop, mixed-priority
  workload, undercommitted blocks, policy matrix {lifo, priority, cost}.
  Judged by goodput; aggregate tok/s is expected flat and the doc says so.
- **Round 4 (chunk 12)** `ab_5090_overlap{,_longctx}.toml`: overlap on/off ×
  {padded, paged}. Gate: prediction 6's floors, event-order equivalence.
- **H100 day (chunk 13)**: rounds 1/2/4 cell subset at 0.6B and 32B, one
  day, same protocol as h100-plan.md.

5090 rounds run through the linux-box-5090 remote session, the Phase 3.5
pattern.

## 8. Division of labor

Settled in the planning session. The author hand-writes, red→green against
pre-landed suites: the **block allocator** (chunk 2), the **paged attend**
including both index-translation sites, write-map destination and mask-side
physical→logical (chunk 4), and **preemption + victim policies** (chunks 9
and 10). Claude writes everything else: pool layout, BatchMeta/table
plumbing, scheduler paged-mode bookkeeping, warm-up/vocabulary/compile/graphs
integration, priority plumbing through the dialects, the goodput metric, the
**overlap restructure** (delegated like graphs and compile were), all tests,
and the docs. Claude's role inside the author's sessions: interpreting test
failures only.

## 9. Open questions (hers to call)

1. **Names.** `PagedKVPool`, `BlockAllocator`, `FlexAttentionMethod`,
   `PagedStepState`, `PagedKVWriteMap`, and the CLI value (`--attention
   flex` vs `paged`). All placeholders.
2. **Priority surface details.** Field name (`priority`?), bounds (−2..2?),
   and whether the Anthropic dialect should also accept it via a beta header
   for SDK ergonomics.
3. **SLO defaults.** The TTFT/ITL pair for the goodput configs (straw
   proposal: TTFT ≤ 500 ms, per-request ITL p99 ≤ 100 ms at 0.6B; H100 32B
   pair chosen after round 2 numbers exist).
4. **Refcounts now or at Phase 5.** The allocator API includes refcounts for
   prefix sharing; land the field now (recommended, it shapes the free-path
   contract) or keep the Phase 4 allocator binary.
5. **`kv_bucket` under paged**: hard error, warn-and-ignore, or silently
   inert (recommended: validated-but-inert, so mixed configs don't lie).
6. **Preempt-on-boundary semantics.** When a decoding row fails to get its
   boundary-crossing block mid-plan, may it victimize itself (drop from this
   step, keep blocks) or must the policy always pick another row?
7. **Mac/CPU flex arm.** Eager Flex works on CPU for tests; do we also allow
   `--attention flex` off-CUDA for development serving, or gate it to CUDA +
   tests only?
8. **The default flip** (chunk 13, after the numbers): does flex + paged
   become the CUDA serve default, and at what `num_kv_blocks` policy?

## 10. Results

Appended per round as they complete.

### Chunk log

Suite green at every step (530 tests + 5 chaos at chunk 1, from 521 at
phase start; Mac/CPU counts).

1. Precondition refactors (2026-08-19). `KVPool` structural Protocol
   (runtime-checkable) with `BatchedForwardFn`, the scheduler ctor, and the
   runtime front retyped against it; the ctor's slot-count check gated on
   the pool actually having slots. `BatchingConfig` gains `paged_kv` /
   `block_size` / `num_kv_blocks` + validation (block-aligned capacity,
   one-max-request minimum, block-aligned `kv_bucket`) and
   `resolved_kv_blocks`; `new_kv_pool` is the layout branch (refusing paged
   until chunk 2); `scheduler_from_runtime` fails the build when paged +
   CUDA lacks torch_compile. Stats: scheduler states `last_step_plan =
   (rows, prefill, decode)` at plan time and the collector prefers it over
   the snapshot-diff derivation (kept as fallback);
   `kv_allocated_tokens` / `kv_capacity_tokens` added (STATS_SCHEMA_VERSION
   2, additive) with the `kv_state` hook reserved for the paged scheduler;
   bench `kv_fill_mean` prefers per-step capacity. `protocol.py` mask types
   loosened to method-opaque. New `tests/test_paged_config.py` plus pins in
   the kv-pool/stats/bench-metrics suites.
   Deviations from §5's wording, all deliberate: `graphs.py` and
   `warmup.py` keep the concrete `PaddedKVPool` type (both consume
   `scratch_pos`, genuinely padded-specific until chunks 8 and 6 rework
   them); the `preemption_policy` / `overlap_scheduling` knobs land with
   their chunks instead of as dead fields; the `main.py` check waits for
   chunk 7, where the CLI flag it guards first exists. One touch inside the
   hand-written scheduler (`step()`): the two-line `last_step_plan`
   addition, following the `last_forward_shape` precedent — flagged for
   author review.

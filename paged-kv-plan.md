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

**Status: chunks 1–9 complete (chunk 9 closed 2026-08-30: the author
hand-wrote the evict/resume machine, 6457e17 (LIFO victim, blocks and
slot freed, requeue at the queue front with the prompt+emitted replay
prefix, eviction emits nothing) and updated the chunk-5 deadlock pin
deliberately in the same commit; all 6 pre-landed preemption tests
went green including the §2.9 greedy-token-identical oracle, strict
xfail removed, suite 624). Chunk 10 complete same day: the delegated
half (priority through both dialects + IPC, sorted promotion, the
`preemption_policy` knob, stats v3 preemption counters, per-chunk
client timestamps + goodput + SLO point keys, policy suite pre-landed
RED), then the author's hand-written `priority` and `cost` selectors
red→green. The red half also caught a MACHINE-LEVEL LIVELOCK latent
since chunk 9 (evict frees one block; next step's promotion re-admits
the victim and hands the block straight back; lifo livelocks
identically under the swapped arrival order), fixed by her call:
same-step evict-and-replan retry in step(), planning as the
can-anyone-advance oracle. Markers off, suite 653 + 5 chaos. Next:
chunk 11, round 3 (goodput_5090.toml; §9.3 SLO pair still hers).**

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
16-token span, and the KV scatter writes the base tensor directly through a new
3-column `PagedKVWriteMap(batch_row, token_offset, pool_index)` with
`pool_index = table[pos // block_size] * block_size + pos % block_size`.
Rejected: a
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

**2.13 (added 2026-08-30) The served `block_size` default is 64.** The chunk-4
5090 probe found the compiled CUDA Flex lowering prunes every mask template
below a 64-token KV block (`NoValidChoicesError`; Q block size and q_len
irrelevant; the full grid is in the chunk log), so 16-token pages cannot be
the mask's KV blocks on the serving path. Decision (the author's): raise the
default to 64 rather than decouple mask granularity from page size. The
`kv_indices`-is-the-block-table identity survives, chunk 6's per-step mask
work stays "write ints into preallocated tensors", and the fragmentation cost
is ~32 tokens per request on average (~192 vs ~176 held for a ~170-token
short_chat request, against the padded pool's ~4096). Enforced at engine
assembly beside the §2.8 compile guard (`MIN_CUDA_KV_BLOCK` in flex.py);
CPU/eager has no floor, so the equivalence suites keep tiny blocks for cheap
boundary crossings. Rejected: 64-token mask blocks spanning four 16-token
pages: `kv_indices` would need per-step derivation, partial mask blocks
over-read up to 4x, and the complexity is permanent while the floor is
Inductor template pruning a torch upgrade may move (revalidate it, and the
Q-block tile multiple, on upgrades).

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
  forward_fn(...)          scatter to flat pool indices → BlockMask (prebuilt) →
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

### Round 1 (chunk 7): first run 2026-08-30, gate FAILED, cause found and fixed, rerun below

Run on the 5090 at 2ac397f (box session; bench/history commit 9e20637:
`ab-5090-paged`, `ab-5090-paged-longctx`, `ab-5090-paged-default`).
Suite on the box 618 + 5 chaos green, both CUDA twins green, no NaN, no
errors, zero recompiles after Ready on every boot.

First-run A/B (median of 3; delta = flex vs same-cell padded
compile-no-graphs): short_chat c=4 +5.4% PASS, c=16 -9.8% PASS; code
c=8 -10.9% FAIL (marginal); multi_turn c=8 -17.8% FAIL; long_context
c=1 -17.2%, c=2 -53.6%, c=4 -30.3%, all FAIL. Default-arm box sanity
PASS (short_chat c=16 3581 vs the 3600-3700 record range).

The box's key diagnostic: a decode step-time CLIFF at 1 -> 2 rows at
long KV, then flat (flex step_p50 4.70 -> 18.65 -> ~21.6 ms across
c=1/2/4 vs sdpa 4.17 -> 5.66 -> ~10.1). Attribution, proven on CPU by
inspecting the traced graphs' placeholder example_values: chunk 6 left
the batch dims unmarked under paged, automatic dynamic promoted
input_ids to a SYMBOLIC batch on the second (batch, width) family, one
symbolic artifact then served every B >= 2 (why 2 -> 4 rows barely
grew), and Inductor's `_use_flex_decoding` gate requires a
statically-known query batch to size its KV splits, so every multi-row
decode silently fell back to the main flex template: a 128-wide Q tile
for one real query row and no KV splitting. B = 1 always specializes
(torch's 0/1 rule), which is why c=1 sat near sdpa, and the damage
scaled with actual KV (short_chat -9.8%, multi_turn -17.8%, longctx
worst). Fix cb1a04a: `_mark_compile_dims` pins the meta-side batch
dims static per family (per-family artifacts are §2.6's vocabulary
anyway) and `PagedStepState.fill` pins the cached mask's own
kv_num_blocks/kv_indices, whose views are born inside the builder;
the regression pin inspects traced-graph placeholders (a plain
backend's example_inputs are real tensors and always look static) and
was verified red on pre-fix code. 5090 rerun of the two pair configs
dispatched; this section gets the confirmed numbers when it reports.

Also on record from the first run:
- Warm bills (0.6B, 16x4096 geometry): flex cold Ready 299 s (sweep
  282.7 s, "20 families x 2 map lengths" confirmed in the log), warm
  72 s (sweep 54.9); padded twin cold 232 s (315 shapes, sweep 216.6),
  warm 104 s (sweep 89.6). Prediction 2's warm-up clauses: warm bill
  under padded's TRUE, cold sweep under padded's FALSE (40 pricier
  individual compiles vs 315 amortized shapes), cold under 2x padded
  cold TRUE (299 < 464). Final §6 grading waits for the phase end.
- Greedy cross-arm spot check (flex vs sdpa servers, 8 prompts, temp
  0): 1/8 token-identical, 7/8 diverge as coherent alternates, no
  derailments; the accepted cross-kernel tie-flip class (compile round
  precedent was 5/7 within one arm), but a high rate worth keeping an
  eye on.
- Harness gap: the §7 "Flex-kernel-ran counter > 0" gate has no
  counter in bench run.json; the suite's profiler test covers the
  silent-fallback risk out-of-band. RESOLVED (the author's call,
  2026-08-30): no in-bench counter. The suite's profiler test plus
  the decode step-time signature (which is what caught this round's
  regression) stand as the gate's working form; the §7 wording reads
  accordingly.

### Round 1 rerun at cb1a04a (2026-08-30): cliff dead, 4 cells still
### outside the gate; residual is a kernel-side flex-vs-cuDNN gap

Box push 1754322 (run dirs `...T171735_cb1a04a_ab-5090-paged`,
`...T172251_cb1a04a_ab-5090-paged-longctx`; round-1 dirs and the
default arm stand as baselines). Suite on the box 619 + 5 chaos green
including the new regression pin; zero recompiles after Ready,
including under B=3 concurrent long-prompt greedy traffic; no NaN.

The fix confirmed behaviorally: longctx decode step_p50 went 4.70 ->
18.65 -> ~21.6 ms (c=1/2/4, first run) to 4.72 -> 8.00 -> ~12.8 ms,
smooth sub-linear scaling restored, and a production-path timing probe
(3000-token prompts) measured B=1 5.63 ms vs B=3 9.53 ms where round 1
had ~4.7 -> ~19 flat. Kernel-name identity turned out UNMEASURABLE on
this build: Inductor 2.10 names fused kernels after the fusion group's
aten ops, not the winning template, so main-vs-decoding cannot be told
apart by name (the §7 counter gate needs a different encoding; the
step-scaling signature is the working substitute).

Rerun A/B (median of 3; delta = flex vs padded compile-no-graphs):
short_chat c=4 +4.6% PASS, c=16 -4.6% PASS (was -9.8), code c=8 -9.1%
PASS (was -10.9), multi_turn c=8 -15.1% FAIL (was -17.8); long_context
c=1 -17.2% FAIL (unchanged; B=1 never had the bug), c=2 -26.6% FAIL
(was -53.6), c=4 -10.4% FAIL by a hair (was -30.3).

Reading of the residual (the box's, endorsed here): flex decode now
carries roughly +0.6 to +2.7 ms per step over sdpa, growing with
rows and KV; c=1's identical -17.2% across both rounds floors it as a
kernel-side gap (compiled Flex decode vs cuDNN at long-KV decode on
this build), not scheduling. Chunk 8's graphs should absorb the launch
share; the kernel share (template tuning, BLOCK sizes, split-KV
parameters at these geometries) is the open attribution target, with a
GPU-busy-vs-host split probe dispatched to divide the two.

Warm-bill follow-up: per-family static artifacts doubled the bills
(cold 299 -> 602 s, warm 72 -> 144 s) because the sweep warmed BOTH
map-length populations per family while bucketing makes exactly one
reachable (length-1 exists only in the (1, 1) family; any b >= 2 step
carries at least two real tokens, and a lone 1-token row lands at
width 1). Fixed 72ccaf0: one forward per family, reachability
enforced by a zero-post-Ready-compile traffic test including the
lone-row decode; trimmed bills measured below.

Round-1 close-out probes (box record
`bench/history/2026-08-31T004900_72ccaf0_paged-followups/`):
- Trimmed bills at 72ccaf0: cold Ready 368 s (sweep 347.4; the ~69 s
  over round 1's 299 is the ~5 per-family static decode artifacts,
  real cold work the shared symbolic artifact never paid), warm 79 s
  (fully recovered from 144). Reachability held on hardware: a
  concurrent burst then a lone (1, 1)-family decode, zero recompile
  lines after Ready.
- Residual split (production path, 3000-token prompts, mean of 10
  decode steps): flex B=1 wall 5.52 ms = GPU 3.91 + host 1.61; B=4
  wall 9.78 = GPU 6.63 + host 3.15; sdpa B=1 4.05 = 2.22 + 1.84; B=4
  5.14 = 3.98 + 1.15. So the B=1 gap is ENTIRELY kernel time (the
  flex attention template runs ~2.12 ms/step vs cuDNN's ~0.47
  including mask prep, ~4x), and the B=4 gap is 57% kernel + 43%
  flex-side host (+2.0 ms/step, dominated by fill's per-element
  device table writes and mask dispatch). Consequence for chunk 8:
  graphs plus a bulk-copy fill absorb the host share at B >= 2 and
  nothing at B=1; the kernel share (template/split tuning at decode
  geometries) is the open target beyond this phase's graphs work.

VERDICT (the author's, 2026-08-30): round-1 gate miss ACCEPTED on the
4 standing cells (multi_turn -15.1%, longctx -17.2/-26.6/-10.4);
proceed to chunk 8. The misses are attributed (kernel-side
flex-vs-cuDNN decode gap) and stand on record for round 2's grading;
kernel tuning is deferred, not dropped.

### Round 2 (chunk 8), 2026-08-30: full stacks + capacity headline

Run on the 5090 at e68bbff (box push e6bc10f; run dirs
`...T182141/T182741/T183950_e68bbff_*`). Suite on the box 631 + 5
chaos green; the chunk-8 exit gate (bit-exact replay including the
in-place physical table permutation, spike gate 4's probe) PASSED at
the bit-exact bar. Boot: 5 decode graphs captured in 0.1 s (the
persistent-buffer design: recordings bake the step state's addresses,
there is nothing to marshal at capture), Ready bills hold the trimmed
sweep's numbers (cold 364 s / warm 81 s; graphs add ~nothing), zero
recompiles after Ready, decode replay 194/194 = 100% with zero false
replays.

Full-stack A/B (median of 3; delta = flex vs padded, both complete
pipelines; §6 prediction 1's ranges):
- short_chat c=4 +1.8% PASS, c=16 +1.3% PASS: flex 3612 tok/s is
  ABOVE the padded default record (3581) with TTFT 9 ms better.
- code c=8 -1.8% PASS; multi_turn c=8 -4.6% PASS (round-1 rerun had
  -15.1: the graphs plus the bulk-copy fill closed the host share as
  attributed).
- long_context c=1 -22.6% MISS, c=2 -17.9% MISS: the pre-accepted
  kernel cells. Both arms improved with graphs; padded improved MORE
  at c=1 (its 1.84 ms host gap was the bigger share there), which
  widened the percentage exactly as probe B's split predicted.
  c=4 +5.0% PASS: flex WINS the saturated cell, with lower step_p99
  (26.8 vs 32.7 ms).
- Decode step_p50, longctx c=1/2/4: flex 3.27/5.27/8.13 ms (was
  4.72/8.00/12.8 without graphs) vs padded 2.26/4.60/8.90. Flex is
  now FASTER per step at c=4; the ~1.0/0.7 ms deficit at c=1/2 is the
  attributed flex-template kernel time.
- No NaN, zero errors everywhere.

Capacity headline (§6 prediction 3: flex full stack, max_batch 64 on
num_kv_blocks 1024, exactly the padded pool's 16-slot reservation):
c=16/32/48/64 gave 3545/5392/5165/5479 tok/s, every request
completed, no paged-KV deadlock, zero errors. Clause grades:
sustained >= 48 concurrent PASS; saturated aggregate 5479 = 1.53x the
3581 padded default record, PASS (>= 1.5x); kv_fill_mean
0.026/0.051/0.071/0.070 FAILS the >= 60% clause as written.
short_chat requests hold ~350 tokens for seconds and free them, so
the mean allocated share of a 65 536-token pool stays small; the
clause needs a longer-held workload or a smaller pool to bind.
Graded, not tuned, per the house rule. (TTFT CV warnings 6.7-14.6% on
the capacity cells: arrival jitter at high fan-in, on record.)

Open from the rounds: the longctx B <= 2 flex decode-kernel gap
(template/split tuning, deferred by the author's round-1 call). The
flex-kernel-ran counter question is settled: no in-bench counter (the
author, 2026-08-30); the suite's profiler test and the step-time
signature are the gate's working form.

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
   author review (accepted 2026-08-19, along with §2's decisions and §9's
   recommendations as written).
2. Paged pool + allocator contract (2026-08-19, prep half). `PagedKVPool`
   landed: flat per-layer `((num_kv_blocks + 1) * block_size, G, D)`
   tensors, scratch block at the tail, protocol-conformant, built by
   `new_kv_pool`'s paged branch at `resolved_kv_blocks` (the RoPE guard
   covers both layouts). `BlockAllocator` landed as an API stub
   (allocate / free / incref / num_free / num_allocated, refcount-1
   contract) with `tests/test_block_allocator.py` PRE-LANDED RED
   (11 xfails: ordering, FIFO reuse, refcount round-trips, loud misuse,
   scratch exclusion). Suite otherwise green (535 + 5 chaos). The chunk
   completes with the author's hand-written session: fill the stub,
   delete the module xfail marker, all 11 green.
   Completed 2026-08-29: the author hand-wrote the deque + refcount core
   (one red→green finding: `allocate()` popped and refcounted but never
   returned the block — 7 of 9 failures from one missing return, and the
   xfail marker's `raises=NotImplementedError` constraint is what made
   the partial implementation loudly visible instead of quietly xfailed);
   the loud-misuse guards (range check, double-free, incref-of-free, in
   SlotAllocator's message style) were handed back to Claude as
   bookkeeping. Marker deleted, all 11 pass unmarked; suite 546 + 5
   chaos.
3. Block tables + BatchMeta extension + pre-landed attend suites
   (2026-08-29). `PagedKVWriteMap` (`batch_row`, `token_offset`,
   `pool_index`) and `PagedTables`
   (block_tables, kv_num_blocks, inverse_tables, write_map — inverse
   carries the past-any-bound sentinel `max_blocks_per_seq` for unowned
   blocks, never 0/-1) land in protocol.py; `BatchMeta.seed_paged_tables`
   mirrors the seed-once discipline with int32/int64 and B-alignment
   validation, and `paged_tables` raises when unseeded (no derivation
   exists). The runtime front's device move was extracted to a testable
   `move_batch_to` and now carries BOTH seeded passengers, with an
   MPS-gated survival test (on CPU the identity gate never opens). New
   `engine/batching/paging.py`: `paged_write_map` as the hand-written
   chunk-4 stub (contract in the docstring) and the `PagedStepState`
   shell (allocation + sentinel convention; fill() lands chunk 5).
   `models/attention/flex.py`: `FlexAttentionMethod` stub — nullcontext
   execution_context (no cuDNN pin), batched-only raises, the two
   hand-written methods red, spike wrinkles inventoried in the module
   docstring. `tests/test_flex_equivalence.py` PRE-LANDED RED (11
   strict xfails: 4 write-map pins incl. boundary crossings and filler
   skipping, scattered-table/chunked/decode/mixed-batch equivalence vs
   the padded oracle, stale-block-reuse fencing, pool-write positions);
   `tests/test_paged_tables.py` green (8). Suite 554 + 5 chaos, 11
   xfailed. The chunk-4 session is the author's: `paged_write_map` +
   the Flex attend, red→green against this suite.
4. Paged attend CPU correctness (2026-08-30). The author implemented both
   index-translation sites: `paged_write_map` maps each new logical token to
   its flat pool destination, while the FlexAttention mask translates
   physical pool positions back to logical positions for causality. The
   attention path scatters new K/V directly into the flat pool, runs grouped-
   query FlexAttention over the shared pool, and restores the model's grouped
   output shape. All 11 Flex equivalence tests now run unmarked and pass,
   including scattered blocks, chunked prefill, decode, mixed batches, stale
   pool contents, and exact pool-write positions. Local suite: 561 passed, 15
   platform skips, with 5 chaos tests deselected. The compiled-CUDA
   equivalence twin on the 5090 remains the chunk's exit gate.
   Gate closed 2026-08-30, via three 5090 rounds (runs through the
   linux-box-5090 session; the box ran and probed, edited nothing).
   Round 1 (twin at CPU-suite geometry): all four CUDA tests died in
   Inductor lowering, and the probe grid isolated two floors eager Flex
   never checks: head_dim >= 16 (tl.dot), and mask KV BLOCK_SIZE >= 64
   (below it every Triton template is pruned, `NoValidChoicesError`; Q
   block size and q_len irrelevant across the whole grid; KV 4/16/32
   fail, 64/128 pass). Twin reshaped to 64-token blocks, head_dim-16 toy
   arch, 150/100-token prompts (c6afbf3). Round 2 exposed two
   conformance gaps in the attend itself, both isolated with standalone
   repros: `from_kv_blocks` needs canonical `(B, 1, 1[, N])` table ranks
   (eager broadcasts squeezed shapes; the CUDA template derives strides
   from actual ndim and emits broken code), and the mask's Q BLOCK_SIZE
   must be a multiple of the 128 prefill tile (a raw width like 150
   fails divisibility; widths <= 12 had routed to a laxer decode
   template, which is why the probe grid missed it). Fixed in flex.py
   (3cb1284, Claude, flagged for author review; index translation and
   `mask_mod` untouched, eager numerics bit-identical; the floors live
   there as `Q_BLOCK_MULTIPLE` / `MIN_CUDA_KV_BLOCK`, revalidate on
   torch upgrades). Round 3: GATE GREEN, 15/15 in ~28 s including all
   Inductor compiles, flex kernels confirmed on the profiler timeline
   (`triton_tem_fused_flex_attention_*`), bf16-vs-sdpa max |diff| one
   bf16 ulp (0.0078 vs atol 3e-2, ~4x margin). Fallout the same day:
   §2.13 (block_size default 64 + the assembly floor guard + its test).
   Chunk 4 complete; suite 566 + 5 chaos.
5. Scheduler paged mode (2026-08-30, delegated). `CBSequence.block_table`
   is the host truth; the scheduler takes a keyword-only paged pair
   (block_allocator + paged_state, validated against config and pool) and
   in paged mode: promotion takes each sequence's first block or stops
   admitting (an admitted sequence always has somewhere to write);
   `_plan_step` reserves blocks through each row's grant after water-fill
   and quantization, trimming grants the pool cannot back (§4 atomicity);
   a boundary-starved row sits the step out with its blocks kept (§9.6)
   and stays active in place; finish and abort release through
   `_release_kv` (slot plus every block); a fully starved active set
   raises "paged KV deadlock" loudly instead of spinning (preemption is
   chunk 9's answer). `PagedStepState.fill` lands: per-step in-place
   rewrite of the persistent buffers with fillers routed to the scratch
   block mapped to logical 0 (finite softmax), returning sliced views
   plus the step's write map, seeded into the meta after `shape_step`;
   shape_step's kv rounding is now skipped under paged (§2.6). `kv_state`
   exposes (allocated, capacity) tokens and the chunk-1 collector hook
   runs against the real property. Both exit gates green on CPU: the
   block-accounting suite (fill unit pins incl. buffer address
   stability; lifecycle, abort, admission-wait, trim, and starve
   scenarios under a per-step no-leak invariant; the deadlock raise) and
   the engine-level oracle (paged CB vs padded CB, weight-shared tiny
   Qwen3, greedy, staggered arrivals): token-for-token at parity AND at
   10 undercommitted blocks, where trims and starvation change the step
   sequence but not one token. One wrinkle: the oracle arms must run
   under inference_mode like the production front; eager Flex refuses
   grad-enabled calls (its mask omits backward-only q-index metadata).
   Touches inside the hand-written scheduler, flagged for author review:
   ctor pair + validation, the promotion gate, reservation in
   `_plan_step`, `_release_kv` at the three free sites, the deadlock
   check, table seeding after shape_step, and the active-list rebuild
   (drop exactly the finished rows so starved rows survive). Suite 589
   + 5 chaos.
6. Warm-up / vocabulary / compile (2026-08-30, delegated). The paged
   vocabulary drops the kv axis: `shape_vocabulary()` yields one entry
   per (batch, width) family under `paged_kv` (third element pinned to
   the logical bound; `shapes_bounded` stops demanding the inert
   `kv_bucket`, which stays validated when set). Measured at the
   standard 5090 geometry and pinned in the gate suite: sweep shapes
   315 -> 20, decode shapes (the future graph keys) 80 -> 5, the
   structural half of §6 prediction 2. Warm-up's paged branch fills the
   SAME `PagedStepState` buffers traffic mutates (the §4 device-path
   rule) with all-filler steps pointing at the scratch block, swaps in
   write maps parked on the scratch block's flat indices, and runs two
   forwards per family (map lengths 1 and max(2, batch): both compile
   populations, the old kv-sweep alternation collapsed to an inner
   pair); the sweep also builds every family's `BlockMask` behind
   Ready. Mask caching per §2.5's ownership: `PagedStepState` gains a
   persistent `start_pos` buffer plus a per-family mask cache over an
   injected builder; the hand-written construction moved intact into
   `FlexAttentionMethod.build_family_mask` (mask_mod byte-identical,
   now reading starts through its argument, the persistent buffer,
   instead of the per-step `meta.start_pos`: a cached mask closing
   over a step's own tensor is a stale-closure trap, which is why the
   buffer exists); `PagedTables` gains an opaque `mask` field, seeded
   by `fill(..., num_new_max)` and returned by `build_batched_mask`,
   so `from_kv_blocks` never runs in the step loop, with hand-built
   test tables (mask=None) falling back to per-step construction so
   the chunk-4 suite is untouched. Compile integration:
   `forward_batched_impl` now traces WHOLE with Flex (the wiring the
   chunk-4 twin deferred here); `BatchMeta.paged_tables` became a
   cached_property so a seeded read is a Dynamo-traceable cache hit
   (the kv_write_map pattern; the plain property's dict.get body drove
   Dynamo into recursive InternalTorchDynamoError); the compiled
   hoists stop forcing the padded write map on paged metas;
   `_mark_compile_dims` marks only the paged map's length dynamic.
   One design finding, hers to veto: flex_attention compares the query
   batch against the mask tensors' batch, so a mark_dynamic batch dim
   against a fixed-batch cached mask specializes on the spot and torch
   hard-errors on the broken promise; resolution is one artifact per
   (batch, width) family under either compile strategy, which is
   §2.6's vocabulary anyway. Engine assembly builds the paged trio
   before the sweep, injects the method's mask builder, and refuses
   cuda_graphs + paged loudly until chunk 8. Exit gates green on CPU
   (tests/test_paged_compile.py): the recompile counter (warm three
   families, then deeper histories, permuted tables, shifted starts:
   zero new artifacts, zero new masks) and the §4 construction counter
   at engine level (a warmed scheduler serves traffic with zero mask
   constructions), plus compiled-vs-eager equality for the whole-impl
   trace; a CUDA-skipif twin (real Inductor, error_on_recompile,
   oracle equivalence at the chunk-4 floor geometry) rides to the box
   with round 1. Touches flagged for review: flex.py (the relocation
   above; index translation and numerics untouched) and scheduler.py
   (the fill call gains `meta.num_new_max`). Suite 600 + 5 chaos.
7. 5090 round 1 (2026-08-30, closed same day; §10 above is the full
   record). Local half da0d76c: `--attention flex` implies the paged
   stack end to end (main.py §2.8 exit, graphs default-off pre-chunk-8,
   --block-size/--num-kv-blocks knobs through CLI/TOML/bench,
   build_runtime block_size plumbing, assembly layout/method mismatch
   guard both ways), round-1 configs
   `ab_5090_paged{,_longctx,_default}.toml` (three arms across three
   files; the bench schema is one matrix per file). The round itself
   ran three times on the box: first run FAILED the -10% gate on 4/7
   cells with a 4x multi-row decode cliff; attribution (proven on CPU
   via traced-graph placeholders) was automatic dynamic promoting the
   batch dim on the second family, disqualifying Inductor's
   flex-decoding kernel (`static_batch`); fixed cb1a04a (per-family
   static batch + mask-tensor pins + a placeholder-inspecting
   regression pin verified red pre-fix). Rerun: cliff dead, 3 cells
   recovered, 4 still out; probes split the residual (B=1 all kernel,
   B=4 57/43 kernel/host) and the warm-bill doubling was fixed by the
   one-forward-per-family sweep (72ccaf0, reachability argued and
   enforced: length-1 maps exist only in (1,1)). Gate miss accepted by
   the author; kernel tuning deferred on record. Also learned: the §7
   flex-kernel-ran counter is unmeasurable by kernel NAME on torch
   2.10 (fused names come from the fusion group, not the winning
   template); the step-scaling signature is the working substitute.
   Suite 609 + 5 chaos.
8. CUDA graphs on paged decode + round 2 (2026-08-30, delegated; §10's
   round-2 section is the record). Graphs key on (batch, 1): kv is a
   value (§2.6), so the capture set is the batch buckets, 5 recordings
   vs padded's ~80. The static buffers ARE the step tables:
   `PagedStepState` gained the persistent padded decode write map §3
   always specified (batch_row/token_offset constants, pool_index
   rewritten per step, fillers parked on the scratch block; prefill
   maps stay exact and fresh), and `fill` was rewritten to assemble
   each step in CPU staging and land it with one bulk `copy_` per
   buffer, which is simultaneously the replay marshal (recordings bake
   the state's addresses; the wrapper copies only the five meta-side
   tensors) and the fix for probe B's +2 ms/step flex-side host gap
   from per-element device table writes. `_replayable` verifies by
   data_ptr that a step's seeded tables and map are the state's
   buffers, so hand-built metas fall through to eager; capture runs
   through `fill` itself with a shared-block dummy batch, the family
   mask riding in. Wiring opened (assembly refusal removed, graphs
   default-on under flex on CUDA). Exit gates: CPU decode-map/staging
   pins plus wrapper guard and capture-setup identity tests; on the
   box, the bit-exact-replay-after-in-place-table-permutation gate
   PASSED, capture cost 0.1 s, replay 194/194, zero recompiles.
   Round 2: prediction 1 PASSES on every cell except the pre-accepted
   longctx kernel cells (and flex now beats the padded default on
   short_chat c=16, 3612 vs 3581); the capacity headline passes its
   concurrency and 1.5x clauses (5479 tok/s at 64 slots on 16-slot KV
   parity, no deadlock) while the kv_fill_mean >= 60% clause fails as
   written (short_chat frees too fast to bind it). Suite 618 + 5
   chaos local, 631 + 5 on the box. Chunk-9 prep rode out with the
   close: tests/test_preemption.py PRE-LANDED RED (6 strict xfails,
   all failing today via the chunk-5 deadlock raise, verified with
   --runxfail): exhaustion-evicts-instead-of-deadlocking, LIFO victim
   requeued at the queue front ahead of waiting arrivals, eviction
   emits nothing with append-only streams (exact token counts),
   abort-while-preempted with no double-free, per-step accounting and
   kv_state coherence through evictions, and the §2.9 bar: greedy
   streams token-identical between an unconstrained arm and a
   4-block-pool arm that must evict (weight-shared tiny Qwen3). The
   chunk-9 session is the author's: the evict/resume machine,
   red→green, LIFO first; note in the suite header that
   test_paged_scheduler's deadlock pin must be updated deliberately in
   that session.
9. [HAND] Preemption (2026-08-30, the author's session, 6457e17
   red→green against the pre-landed suite). The machine:
   `_preempt_sequence` evicts where the chunk-5 deadlock raise stood
   (step() falls through to it only when planning yields no rows with
   rows still active, i.e. after grant trimming has already failed),
   frees the victim's blocks AND slot via the existing `_release_kv`,
   and requeues it with `appendleft` (the queue front, ahead of
   waiting arrivals), emitting no client event. `_preempt_lifo` picks
   `active[-1]`, the newest-admitted row (victim POLICIES are chunk
   10). Resume state lives on `CBSequence.replay_prefix_token_ids`
   (prompt + every token already emitted) with `position` reset to 0;
   a new `prefill_token_ids` property routes `is_prefilling`,
   `remaining_prompt`, `input_tokens_at`, and the step-plan prefill
   accounting through the replay prefix while leaving
   `prompt_token_ids` (the client's input) and `output_token_ids`
   (the append-only ledger) untouched, so re-preemption after further
   decoding recomputes the prefix correctly. The chunk-5 deadlock pin
   was updated deliberately in the same commit:
   test_true_deadlock_raises_loudly became
   test_true_deadlock_preempts_and_completes, asserting no leaks each
   step, the victim reset (position 0, no slot, empty table, correct
   replay prefix), and full completion. Close-out from this side: the
   suite's module-level strict xfail removed, all 6 tests green
   unmarked: exhaustion completes both requests with zero allocated
   blocks at idle, LIFO victim ordering ahead of arrivals, exact
   append-only token counts through eviction, abort-while-preempted
   with the allocator count unmoved, per-step accounting/kv_state
   coherence, and the §2.9 bar: greedy streams token-identical between
   the unconstrained arm and the 4-block arm forced to evict. Suite
   624 + 5 chaos.
10. Priority + policies + goodput, delegated half (2026-08-30). The
    `priority` field (§2.10): `Field(ge=-2, le=2, default=0)` on both
    request models (the §9.2 straw bounds; the Anthropic beta-header
    alternative stays open), threaded through
    tokenize_and_build_request into `InferenceRequest.priority`
    (crosses the IPC pickle for free; the wire test pins it) and onto
    `CBSequence.priority`. Promotion: `_promote_queued` stable-sorts
    the queue by priority alone before admitting, so deque order keeps
    supplying arrival order; equal priorities are byte-for-byte
    today's FCFS (the sort is skipped entirely when every queued
    priority is 0), and the chunk-9 `appendleft` becomes front-of-class
    exactly as §3 words it. Victim dispatch: `step()` now calls
    `_preempt_one` (the one-line touch to the hand-written machine,
    flagged for review), which routes "lifo" to the chunk-9 selector
    and counts every eviction plus its replay-prefix cost into
    `preemptions_total` / `preempted_tokens_total`;
    `_select_victim_priority` / `_select_victim_cost` are contract
    stubs raising NotImplementedError, the chunk-2 pattern.
    `BatchingConfig.preemption_policy` ("lifo"/"priority"/"cost",
    paged-only for non-default values) rides --preemption-policy,
    serve.toml (parser-derived), and the bench `_SERVE_FLAG_KEYS`.
    Stats schema v3 (additive): per-step `preemptions` /
    `preempted_tokens` diffed from the monotonic totals the way graph
    hits are; the debug-endpoint pin consciously bumped. Bench:
    results schema v2 adds `t_chunks` per-chunk arrival timestamps in
    both SSE clients and the derived per-request `client_itl_p99_s`;
    point keys `priority` (sent in bodies only when nonzero, default
    cells byte-identical) and the joint `slo_ttft_s`/`slo_itl_p99_s`
    pair (half a pair is a ConfigError); `summarize_repeat` computes
    goodput per §2.11 (errors in the denominator, single-chunk ITL
    vacuous) plus `preemptions_total`/`preempted_tokens_total`, all
    three in the median roll-up; bench-spec.md §3/§4 amended, incl.
    the deliberate carve-out from the "no client gap distributions"
    rule (goodput's ITL clause is client-experienced BY DESIGN).
    tests/test_victim_policies.py lands with the green half (sorted
    promotion incl. overtake/front-of-class, counters, collector
    diffs) and the policy classes PRE-LANDED RED: 4 strict xfails
    (raises=NotImplementedError so partial work fails loudly),
    verified red via the stubs. Suite 649 + 4 xfail + 5 chaos.
    Completed same day, the author's session: both selectors
    hand-written as min-scans over `active` with `>=` keeping the
    newest (so the LIFO tiebreak falls out of iteration order); one
    typo fix handed back. The pre-landed red half then earned its
    keep twice. (1) The cost test's exhaust-order scenario found the
    test itself over-specified: both rows freeze at block-boundary
    position 8, an exact tie, and the newest-wins tiebreak evicts
    "b"; the test moved to a 5-block geometry where "a" is strictly
    cheaper (8 vs 12), a deliberate pre-landed-suite edit. (2) The
    swapped-arrival cost test exposed a MACHINE-LEVEL LIVELOCK latent
    since chunk 9, which lifo reproduces identically (verified by
    trace: 28 evictions in 60 steps, the survivor pinned at 12): the
    victim frees one block, and the NEXT step's promotion runs before
    planning, re-admits the victim from the queue front, and hands
    the freed block straight back; the starved row never advances.
    The chunk-9 suite never met it because its single arrival order
    makes the newest row the block-rich one. Fix (her call among
    evict-until-progress / same-step retry / hold-back-the-victim,
    implementation handed back): same-step evict-and-replan in
    step(): planning retries after each eviction, so freed blocks go
    to the rows the eviction was meant to unblock and `_plan_step`
    itself is the can-anyone-advance oracle (no duplicated
    reservation arithmetic, no re-admission state). The loop is
    bounded by len(active); a lone survivor always advances. An
    eviction step now also runs the unblocked rows' forward instead
    of being wasted, which is what prediction 4 wants. Class markers
    off, every test unmarked; suite 653 + 5 chaos.

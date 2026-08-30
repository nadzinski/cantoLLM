"""The chunk-5 engine-level oracle (paged-kv-plan.md §5.5): a paged CB
scheduler against a padded CB scheduler, token for token.

Both arms run the same weight-shared tiny Qwen3 on CPU, greedy, over the
same staggered arrival schedule; the paged arm swaps in FlexAttentionMethod,
PagedKVPool, and the scheduler's paged trio. Every emitted token and finish
reason must match exactly. Trust chain: the padded CB arm is proven against
StandardBackend.generate (test_cb_engine equivalence), and the flex attend
against the padded attend (test_flex_equivalence); this gate closes the
loop at the scheduling layer, where block reservation, table seeding, and
frees could silently corrupt either arm's history.

The undercommitted variant is the sharper half: with barely more than one
request's worth of blocks, grants get trimmed and rows starve, so the two
arms take DIFFERENT step sequences, and the streams must still match,
because chunking is content-neutral and starvation only delays.
"""

import torch

from cantollm.engine.batching import BatchingConfig
from cantollm.engine.batching.allocator import BlockAllocator, SlotAllocator
from cantollm.engine.batching.paging import PagedStepState
from cantollm.engine.batching.scheduler import ContinuousBatchingScheduler
from cantollm.kv_pool import PaddedKVPool, PagedKVPool
from cantollm.models.attention import FlexAttentionMethod, PaddedAttentionMethod
from cantollm.models.qwen3.model import Qwen3
from tests.test_cb_scheduler import drain, make_request
from tests.tiny_model import TINY_ARCH

BLOCK = 4
MAX_SEQ = 32
MAX_BATCH = 2
BUDGET = 8


def build_arms() -> tuple[Qwen3, Qwen3]:
    """(padded arm, flex arm), identical weights."""
    torch.manual_seed(1234)
    padded = Qwen3(
        qwen3_config=TINY_ARCH, attention_method=PaddedAttentionMethod()
    )
    flex = Qwen3(
        qwen3_config=TINY_ARCH,
        attention_method=FlexAttentionMethod(block_size=BLOCK),
    )
    flex.load_state_dict(padded.state_dict())
    return padded.eval(), flex.eval()


def forward_fn(model: Qwen3):
    """The production front runs the batched forward under
    `inference_mode` (ModelRuntime.forward_batched); the flex attend
    depends on it: its BlockMask skips the backward-only q-index
    metadata, so a grad-enabled call refuses to run."""
    return torch.inference_mode()(model.forward_batched)


def padded_scheduler(model: Qwen3) -> ContinuousBatchingScheduler:
    config = BatchingConfig(
        max_batch=MAX_BATCH, max_seq_len=MAX_SEQ, max_tokens_per_step=BUDGET
    )
    pool = PaddedKVPool(
        num_layers=TINY_ARCH["num_transformers"], max_batch=MAX_BATCH,
        max_seq_len=MAX_SEQ, num_groups=TINY_ARCH["num_groups"],
        head_dim=TINY_ARCH["head_dim"], dtype=torch.float32,
        device=torch.device("cpu"),
    )
    return ContinuousBatchingScheduler(
        forward_fn=forward_fn(model), pool=pool,
        allocator=SlotAllocator(MAX_BATCH), config=config,
    )


def paged_scheduler(
    model: Qwen3, num_kv_blocks: int | None = None
) -> ContinuousBatchingScheduler:
    config = BatchingConfig(
        max_batch=MAX_BATCH, max_seq_len=MAX_SEQ, max_tokens_per_step=BUDGET,
        paged_kv=True, block_size=BLOCK, num_kv_blocks=num_kv_blocks,
    )
    pool = PagedKVPool(
        num_layers=TINY_ARCH["num_transformers"],
        num_kv_blocks=config.resolved_kv_blocks, block_size=BLOCK,
        max_seq_len=MAX_SEQ, num_groups=TINY_ARCH["num_groups"],
        head_dim=TINY_ARCH["head_dim"], dtype=torch.float32,
        device=torch.device("cpu"),
    )
    return ContinuousBatchingScheduler(
        forward_fn=forward_fn(model), pool=pool,
        allocator=SlotAllocator(MAX_BATCH), config=config,
        block_allocator=BlockAllocator(config.resolved_kv_blocks),
        paged_state=PagedStepState(
            max_rows=MAX_BATCH, max_blocks_per_seq=MAX_SEQ // BLOCK,
            num_kv_blocks=config.resolved_kv_blocks,
            device=torch.device("cpu"),
        ),
    )


def arrivals() -> dict[int, list]:
    """Staggered greedy traffic: chunked prefill (r1 spans two budgets),
    decode-heavy r2, and r3 arriving while both slots are held so it
    waits in the queue."""
    return {
        0: [
            make_request("r1", list(range(11, 23)), max_tokens=6),
            make_request("r2", [31, 32, 33, 34, 35], max_tokens=8),
        ],
        2: [make_request("r3", list(range(51, 60)), max_tokens=5)],
    }


def assert_streams_match(padded_results, paged_results):
    assert padded_results.keys() == paged_results.keys()
    for rid in padded_results:
        assert paged_results[rid]["tokens"] == padded_results[rid]["tokens"], (
            f"{rid}: paged {paged_results[rid]['tokens']} vs "
            f"padded {padded_results[rid]['tokens']}"
        )
        assert paged_results[rid]["finish"] == padded_results[rid]["finish"]
        assert paged_results[rid]["errors"] == padded_results[rid]["errors"]


def test_paged_matches_padded_token_for_token():
    padded_model, flex_model = build_arms()
    padded_results = drain(padded_scheduler(padded_model), arrivals())
    paged_results = drain(paged_scheduler(flex_model), arrivals())
    assert_streams_match(padded_results, paged_results)
    # The run generated something in every request (not vacuous).
    assert all(r["tokens"] for r in padded_results.values())


def test_undercommitted_paged_still_matches_padded():
    # 10 blocks (40 token-positions) against a parity demand of 16: the
    # paged arm must trim and starve its way through, on a different
    # step sequence than the padded arm, with identical streams.
    padded_model, flex_model = build_arms()
    padded_results = drain(padded_scheduler(padded_model), arrivals())
    scheduler = paged_scheduler(flex_model, num_kv_blocks=10)
    paged_results = drain(scheduler, arrivals())
    assert_streams_match(padded_results, paged_results)
    assert scheduler.block_allocator.num_allocated() == 0

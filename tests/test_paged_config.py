"""Phase 4 chunk 1: paged config knobs + the engine-assembly guard.

The knobs land before the paged pool exists (paged-kv-plan.md §5, chunk 1),
so what these pin is validation and the build-time rule the device-blind
config cannot enforce itself: paged on CUDA without torch.compile must fail
the engine build loudly — FlexAttention only performs compiled
(flex-spike-results.md §7), and a silently-eager Flex path is the SDPA
silent-fallback lesson all over again.
"""

from types import SimpleNamespace

import pytest
import torch

from cantollm.engine.batching import BatchingConfig


def config(**overrides) -> BatchingConfig:
    kwargs = dict(max_batch=2, max_seq_len=32, max_tokens_per_step=8)
    kwargs.update(overrides)
    return BatchingConfig(**kwargs)


class TestPagedKnobValidation:
    def test_defaults_are_off_and_valid(self):
        c = config()
        assert c.paged_kv is False
        # 64 = the compiled CUDA kernels' mask KV-block floor
        # (paged-kv-plan.md §2.13); smaller is a CPU-test-only geometry.
        assert c.block_size == 64
        assert c.num_kv_blocks is None

    def test_paged_accepts_block_aligned_capacity(self):
        c = config(paged_kv=True, block_size=16)   # 32 % 16 == 0
        assert c.resolved_kv_blocks == 2 * (32 // 16)

    def test_num_kv_blocks_overrides_parity(self):
        c = config(paged_kv=True, block_size=16, num_kv_blocks=3)
        assert c.resolved_kv_blocks == 3

    def test_resolved_kv_blocks_is_paged_only(self):
        with pytest.raises(ValueError, match="paged_kv"):
            _ = config().resolved_kv_blocks

    def test_block_size_must_be_positive(self):
        with pytest.raises(ValueError, match="block_size"):
            config(block_size=0)

    def test_capacity_must_land_on_a_block_boundary(self):
        with pytest.raises(ValueError, match="multiple of block_size"):
            config(paged_kv=True, block_size=10)

    def test_num_kv_blocks_requires_paged(self):
        with pytest.raises(ValueError, match="paged_kv-only"):
            config(num_kv_blocks=8)

    def test_pool_must_hold_one_max_length_request(self):
        # 32 tokens / 16-token blocks = 2 blocks minimum: an admitted
        # max-size request must be completable alone.
        with pytest.raises(ValueError, match="max-length request"):
            config(paged_kv=True, block_size=16, num_kv_blocks=1)

    def test_kv_bucket_must_be_block_aligned_under_paged(self):
        # The bucket is inert on the paged kernel path (paged-kv-plan.md
        # §2.6) but validated so mixed configs don't lie.
        with pytest.raises(ValueError, match="kv_bucket"):
            config(paged_kv=True, block_size=16, kv_bucket=24)
        config(paged_kv=True, block_size=16, kv_bucket=32)  # aligned: fine


class TestEngineAssemblyGuard:
    def test_paged_cuda_without_compile_fails_the_build(self):
        import cantollm.engine  # noqa: F401  (engine must import before runtime)
        from cantollm.engine.batching.engine import scheduler_from_runtime

        # The guard fires on the device alone, before any pool/model work,
        # so a bare stub runtime is enough — no CUDA needed to pin it.
        runtime = SimpleNamespace(device=torch.device("cuda"))
        with pytest.raises(RuntimeError, match="torch_compile"):
            scheduler_from_runtime(
                runtime, config(paged_kv=True, block_size=16)
            )

    def test_paged_cuda_below_kv_block_floor_fails_the_build(self):
        # The other device-coupled rule: the compiled Flex kernels prune
        # every template below MIN_CUDA_KV_BLOCK (paged-kv-plan.md §2.13),
        # so a small block_size must die at assembly, not as a
        # NoValidChoicesError mid-warm-up. CPU stays free to run tiny
        # blocks; the equivalence suite depends on it.
        import cantollm.engine  # noqa: F401  (engine must import before runtime)
        from cantollm.engine.batching.engine import scheduler_from_runtime

        runtime = SimpleNamespace(device=torch.device("cuda"))
        with pytest.raises(RuntimeError, match="block_size"):
            scheduler_from_runtime(
                runtime,
                config(
                    paged_kv=True, block_size=16,
                    # A minimal compile-valid vocabulary, so the compile
                    # guard ahead of this one is satisfied and the floor
                    # guard is what fires.
                    torch_compile=True, warmup_shapes=True,
                    prefill_widths=(8,), kv_bucket=16, batch_buckets=(1, 2),
                ),
            )

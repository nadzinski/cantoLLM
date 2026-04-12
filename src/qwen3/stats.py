"""Statistics collection for token generation."""

import time
from collections.abc import Iterator
from dataclasses import dataclass


@dataclass
class GenerationStats:
    """Statistics from a generation run."""

    total_tokens: int
    total_time: float
    avg_tokens_per_sec: float
    avg_tokens_per_sec_ex_swap: float  # Excluding 10 slowest (swap stalls)
    min_window_tps: float | None
    max_window_tps: float | None
    min_window_tokens: list[int] | None
    max_window_tokens: list[int] | None


@dataclass
class SpeculativeStats:
    """Statistics from speculative decoding."""

    draft_tokens_proposed: int
    draft_tokens_accepted: int
    iterations: int

    @property
    def acceptance_rate(self) -> float:
        """Fraction of draft tokens that were accepted."""
        if self.draft_tokens_proposed == 0:
            return 0.0
        return self.draft_tokens_accepted / self.draft_tokens_proposed

    @property
    def tokens_per_iteration(self) -> float:
        """Average tokens yielded per speculative iteration.

        Each iteration yields: accepted_drafts + 1 main token.
        Higher is better (more tokens per forward pass on main model).
        """
        if self.iterations == 0:
            return 0.0
        total_yielded = self.draft_tokens_accepted + self.iterations  # +1 main per iter
        return total_yielded / self.iterations


class StatsCollector:
    """Collects timing statistics from a token stream.

    Usage:
        stats = StatsCollector()
        for token_id in stats.wrap(token_stream):
            # process token
        result = stats.get_stats()
    """

    def __init__(self):
        self.timestamps: list[float] = []
        self.token_ids: list[int] = []

    def reset(self):
        """Clear collected data."""
        self.timestamps = []
        self.token_ids = []

    def wrap(self, token_stream: Iterator[int]) -> Iterator[int]:
        """Wrap a token stream, recording timing for each token.

        Yields tokens through unchanged.
        """
        for token_id in token_stream:
            self.timestamps.append(time.perf_counter())
            self.token_ids.append(token_id)
            yield token_id

    def get_stats(self, window_size: int = 5) -> GenerationStats:
        """Compute statistics from collected data.

        Args:
            window_size: Size of sliding window for min/max speed detection

        Returns:
            GenerationStats with computed metrics
        """
        total_tokens = len(self.token_ids)

        if total_tokens < 2:
            return GenerationStats(
                total_tokens=total_tokens,
                total_time=0.0,
                avg_tokens_per_sec=0.0,
                avg_tokens_per_sec_ex_swap=0.0,
                min_window_tps=None,
                max_window_tps=None,
                min_window_tokens=None,
                max_window_tokens=None,
            )

        total_time = self.timestamps[-1] - self.timestamps[0]
        avg_tps = (total_tokens - 1) / total_time if total_time > 0 else 0

        # Calculate per-token times and avg excluding 10 slowest (swap stalls)
        token_times = [
            self.timestamps[i + 1] - self.timestamps[i] for i in range(len(self.timestamps) - 1)
        ]

        if len(token_times) > 10:
            sorted_times = sorted(token_times)
            trimmed_times = sorted_times[:-10]
            trimmed_total = sum(trimmed_times)
            trimmed_avg_tps = len(trimmed_times) / trimmed_total if trimmed_total > 0 else 0
        else:
            trimmed_avg_tps = avg_tps

        # Calculate sliding window stats
        min_window_tps = None
        max_window_tps = None
        min_window_tokens = None
        max_window_tokens = None

        if total_tokens >= window_size:
            window_speeds: list[tuple[float, int]] = []

            for i in range(total_tokens - window_size + 1):
                window_time = self.timestamps[i + window_size - 1] - self.timestamps[i]
                if window_time > 0:
                    tps = (window_size - 1) / window_time
                    window_speeds.append((tps, i))

            if window_speeds:
                min_tps, min_idx = min(window_speeds, key=lambda x: x[0])
                max_tps, max_idx = max(window_speeds, key=lambda x: x[0])

                min_window_tps = min_tps
                max_window_tps = max_tps
                min_window_tokens = self.token_ids[min_idx : min_idx + window_size]
                max_window_tokens = self.token_ids[max_idx : max_idx + window_size]

        return GenerationStats(
            total_tokens=total_tokens,
            total_time=total_time,
            avg_tokens_per_sec=avg_tps,
            avg_tokens_per_sec_ex_swap=trimmed_avg_tps,
            min_window_tps=min_window_tps,
            max_window_tps=max_window_tps,
            min_window_tokens=min_window_tokens,
            max_window_tokens=max_window_tokens,
        )

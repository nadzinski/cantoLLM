"""Unit tests for the water_fill helper.

water_fill(budget, caps) allocates `budget` units across bins whose
heights are `caps`, like pouring water into uneven bins: the level rises
uniformly, capped per-bin, until the budget runs out or every bin is
full. Residue (when `budget` doesn't divide evenly) goes to the
largest-cap, largest-position bins.
"""

from continuous_batching.scheduler import water_fill


def test_empty_caps():
    assert water_fill(10, []) == []


def test_single_bin_budget_under_cap():
    assert water_fill(5, [10]) == [5]


def test_single_bin_budget_over_cap():
    assert water_fill(20, [10]) == [10]


def test_budget_covers_every_bin():
    assert water_fill(20, [3, 5, 4]) == [3, 5, 4]


def test_exact_even_split():
    assert water_fill(9, [10, 10, 10]) == [3, 3, 3]


def test_residue_concentrates_on_late_positions_when_caps_equal():
    # 10 // 4 = 2, residue = 2 → last two positions absorb the +1
    assert water_fill(10, [3, 3, 3, 3]) == [2, 2, 3, 3]


def test_short_bin_takes_its_cap_long_bins_share_the_rest():
    # pos 0 wants 2 (under fair share of 2); pos 1 wants 3 but gets 2;
    # pos 2 and 3 both want >3 and split the leftover at level 3.
    assert water_fill(10, [2, 3, 4, 5]) == [2, 2, 3, 3]


def test_budget_smaller_than_bin_count():
    # 2 across 3 bins of cap 3 → first bin starves, last two get 1 each
    assert water_fill(2, [3, 3, 3]) == [0, 1, 1]


def test_zero_budget():
    assert water_fill(0, [5, 5, 5]) == [0, 0, 0]


def test_total_allocated_never_exceeds_budget():
    # Mixed-shape sanity check across a few inputs
    cases = [
        (10, [2, 3, 4, 5]),
        (16, [5, 5, 5, 4, 4, 4]),
        (13, [5, 4, 5, 4, 5]),
        (1, [3, 3, 3]),
    ]
    for budget, caps in cases:
        out = water_fill(budget, caps)
        assert sum(out) <= budget
        assert all(0 <= a <= c for a, c in zip(out, caps))

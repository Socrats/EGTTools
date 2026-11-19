"""TimingUncertainty from Python: calculate_end / calculate_full_end with an optional seed.

calculate_full_end(min_rounds) = min_rounds + G, with G geometric (failures before the first
success) with parameter p: E[G] = (1 - p) / p, SD[G] = sqrt(1 - p) / p. calculate_end additionally
truncates at max_rounds when max_rounds > 0.
"""
import math

import numpy as np
import pytest

egt = pytest.importorskip("egttools")

from egttools.numerical.numerical_.distributions import TimingUncertainty  # noqa: E402


def test_explicit_seed_is_reproducible():
    tu = TimingUncertainty(1 / 3)
    assert [tu.calculate_full_end(8, seed=42) for _ in range(5)] == [tu.calculate_full_end(8, seed=42)] * 5


def test_global_seed_makes_the_sequence_reproducible():
    tu = TimingUncertainty(1 / 3)
    egt.Random.init_with_seed(3610063510)
    first = [tu.calculate_full_end(8) for _ in range(20)]
    egt.Random.init_with_seed(3610063510)
    second = [tu.calculate_full_end(8) for _ in range(20)]
    assert first == second
    assert len(set(first)) > 1  # successive calls draw different values


def test_calculate_end_is_truncated_at_max_rounds():
    tu = TimingUncertainty(0.1, 12)
    egt.Random.init_with_seed(1)
    values = [tu.calculate_end(8) for _ in range(500)]
    assert min(values) >= 8 and max(values) <= 12
    assert 12 in values  # with p = 0.1 the cap is reached often


def test_full_end_mean_matches_the_geometric_distribution():
    p, min_rounds, n, k = 1 / 3, 8, 10_000, 4.0
    expected = min_rounds + (1 - p) / p
    tolerance = k * math.sqrt(1 - p) / p / math.sqrt(n)
    egt.Random.init_with_seed(3610063510)
    mean = np.mean([TimingUncertainty(p).calculate_full_end(min_rounds) for _ in range(n)])
    assert abs(mean - expected) < tolerance
    # Power: a wrong parameter is rejected by the same check.
    wrong = np.mean([TimingUncertainty(0.3).calculate_full_end(min_rounds) for _ in range(n)])
    assert abs(wrong - expected) > tolerance

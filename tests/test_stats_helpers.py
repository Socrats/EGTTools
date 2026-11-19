"""Tests of tests/stats_helpers.py on processes with known answers.

AR(1): x_t = phi x_{t-1} + e_t, e_t ~ N(0, 1). Stationary mean 0, variance 1 / (1 - phi^2), long-run
variance of the mean sum 1 / (1 - phi)^2, integrated autocorrelation time (1 + phi) / (1 - phi).
"""
import math

import numpy as np
import pytest
from scipy.signal import lfilter

from stats_helpers import (
    assert_detects,
    assert_within_k_se,
    integrated_autocorrelation_time,
    required_run_length,
    standard_error_batch_means,
    standard_error_iid,
)


def ar1(phi, n, rng, burn_in=1000):
    noise = rng.standard_normal(n + burn_in)
    return lfilter([1.0], [1.0, -phi], noise)[burn_in:]


# --- Standard errors and assertions ---------------------------------------------------------------

def test_iid_standard_error(rng):
    mean, se = standard_error_iid(rng.standard_normal(10_000))
    assert se == pytest.approx(0.01, rel=0.05)
    assert_within_k_se(mean, 0.0, se)


def test_batch_means_is_calibrated_on_correlated_data_and_iid_is_not(rng):
    # 200 replicates of an AR(1) with tau_int = 19: the batch-means SE should cover the true mean
    # about 95% of the time at 1.96 SE; the iid SE is ~sqrt(19) times too small and covers far less.
    phi, n, replicates = 0.9, 20_000, 200
    covered_batch, covered_iid = 0, 0
    for _ in range(replicates):
        x = ar1(phi, n, rng)
        batch = standard_error_batch_means(x)
        covered_batch += abs(batch.mean) < 1.96 * batch.se
        covered_iid += abs(x.mean()) < 1.96 * standard_error_iid(x)[1]
    assert 0.88 <= covered_batch / replicates <= 0.99
    assert covered_iid / replicates < 0.6


def test_batch_means_defaults_follow_flegal_and_jones(rng):
    x = ar1(0.5, 10_000, rng)
    result = standard_error_batch_means(x)
    assert result.batch_size == 100 and result.overlapping
    # BM and OBM estimate the same quantity; OBM is less variable but they agree within a factor.
    assert standard_error_batch_means(x, overlapping=False).se == pytest.approx(result.se, rel=0.5)


def test_assert_within_k_se_reports_z_scores():
    assert_within_k_se([1.0, 2.0], [1.01, 2.0], [0.01, 0.1])
    with pytest.raises(AssertionError, match=r"z=\+5\.00"):
        assert_within_k_se([1.0, 2.5], [1.0, 2.0], [0.1, 0.1])
    with pytest.raises(AssertionError):
        assert_within_k_se(1.0, 1.1, 0.0)  # zero SE requires equality


def test_assert_detects():
    assert_detects(1.5, 1.0, 0.1)
    with pytest.raises(AssertionError, match="cannot detect"):
        assert_detects(1.05, 1.0, 0.1)


# --- Run length -----------------------------------------------------------------------------------

@pytest.mark.parametrize("phi", [0.0, 0.5, 0.9])
def test_integrated_autocorrelation_time_of_ar1(phi, rng):
    act = integrated_autocorrelation_time(ar1(phi, 200_000, rng))
    assert act.reliable
    assert act.tau == pytest.approx((1 + phi) / (1 - phi), rel=0.15)


def test_required_run_length_reaches_the_target_se(rng):
    # Size the run from a pilot, then check over replicates that the SE of the mean is on target.
    phi, target_se = 0.8, 0.05
    plan = required_run_length(ar1(phi, 20_000, rng), target_se)
    means = [ar1(phi, plan.nb_steps, rng, burn_in=plan.burn_in).mean() for _ in range(300)]
    assert np.std(means, ddof=1) <= 1.1 * target_se


def test_required_run_length_rejects_a_short_pilot(rng):
    with pytest.raises(ValueError, match="pilot too short"):
        required_run_length(ar1(0.999, 2_000, rng), 0.01)

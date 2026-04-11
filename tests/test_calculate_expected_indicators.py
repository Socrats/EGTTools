# Copyright (c) 2019-2026  Elias Fernandez
#
# This file is part of EGTtools.
#
# EGTtools is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# EGTtools is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with EGTtools.  If not, see <http://www.gnu.org/licenses/>

"""
Tests for calculate_strategies_distribution, calculate_expected_payoff,
calculate_expected_indicator, and calculate_expected_group_success.

All expected values are computed independently in pure Python using
scipy.stats.multivariate_hypergeom so the C++ results are validated against
a reference that does not share any implementation code.
"""

import numpy as np
import pytest
from scipy.sparse import csr_matrix
from scipy.stats import multivariate_hypergeom

import egttools as egt
from egttools import (
    calculate_nb_states,
    sample_simplex,
    calculate_strategies_distribution,
    calculate_expected_payoff,
    calculate_expected_indicator,
    calculate_expected_indicators,
    calculate_expected_indicators_precomputed,
    calculate_expected_group_success,
)


# ---------------------------------------------------------------------------
# Reference implementations (pure Python / scipy)
# ---------------------------------------------------------------------------

def _ref_strategies_distribution(pop_size, nb_strategies, sd_dict):
    """sd_dict: {state_index: probability}"""
    freq = np.zeros(nb_strategies)
    for idx, prob in sd_dict.items():
        state = np.array(sample_simplex(idx, pop_size, nb_strategies), dtype=float)
        freq += (state / pop_size) * prob
    return freq


def _ref_expected_indicator(pop_size, group_size, nb_strategies, sd_dict, indicator_fn):
    """Generic double-loop reference."""
    nb_group_configs = calculate_nb_states(group_size, nb_strategies)
    result = 0.0
    for state_idx, sd_prob in sd_dict.items():
        state = np.array(sample_simplex(state_idx, pop_size, nb_strategies))
        state_contrib = 0.0
        for g_idx in range(nb_group_configs):
            group = np.array(sample_simplex(g_idx, group_size, nb_strategies))
            prob = multivariate_hypergeom.pmf(group, state, group_size)
            state_contrib += prob * indicator_fn(group)
        result += state_contrib * sd_prob
    return result


def _ref_expected_payoff(pop_size, group_size, nb_strategies, sd_dict, payoff_matrix):
    """avg_payoff(g) = sum_j (g[j]/group_size) * payoff_matrix[j, g_idx]"""
    nb_group_configs = calculate_nb_states(group_size, nb_strategies)

    def indicator(group):
        g_idx = egt.calculate_state(group_size, group)
        return float(np.dot(group / group_size, payoff_matrix[:, g_idx]))

    return _ref_expected_indicator(pop_size, group_size, nb_strategies, sd_dict, indicator)


def _ref_group_success(pop_size, group_size, nb_strategies, sd_dict, threshold, contributing):
    def indicator(group):
        return float(sum(int(group[k]) for k in contributing) >= threshold)

    return _ref_expected_indicator(pop_size, group_size, nb_strategies, sd_dict, indicator)


# ---------------------------------------------------------------------------
# Helpers to build toy stationary distributions
# ---------------------------------------------------------------------------

def _make_sd_sparse(nb_states, sd_dict):
    """Build a 1×nb_states scipy sparse row matrix from {index: prob}."""
    data = list(sd_dict.values())
    col = list(sd_dict.keys())
    row = [0] * len(col)
    return csr_matrix((data, (row, col)), shape=(1, nb_states))


# ---------------------------------------------------------------------------
# Tests: calculate_strategies_distribution
# ---------------------------------------------------------------------------

class TestCalculateStrategiesDistribution:

    def test_monomorphic_all_cooperators(self):
        """Population entirely at the all-cooperators state → freq = [1, 0]."""
        pop_size, nb_strategies = 10, 2
        nb_states = calculate_nb_states(pop_size, nb_strategies)
        # State index 0 corresponds to (pop_size, 0) = all defectors for 2 strategies
        # with sample_simplex ordering; all-cooperators is the last state index.
        # Find it explicitly.
        all_coop_idx = None
        for i in range(nb_states):
            s = sample_simplex(i, pop_size, nb_strategies)
            if s[0] == pop_size:
                all_coop_idx = i
                break
        assert all_coop_idx is not None

        sd_dict = {all_coop_idx: 1.0}
        sd = _make_sd_sparse(nb_states, sd_dict)
        freq = calculate_strategies_distribution(pop_size, nb_strategies, sd)

        assert freq.shape == (nb_strategies,)
        assert np.isclose(freq[0], 1.0)
        assert np.isclose(freq[1], 0.0)

    def test_uniform_two_strategies(self):
        """Uniform stationary distribution → freq matches analytical average."""
        pop_size, nb_strategies = 6, 2
        nb_states = calculate_nb_states(pop_size, nb_strategies)
        uniform_prob = 1.0 / nb_states
        sd_dict = {i: uniform_prob for i in range(nb_states)}
        sd = _make_sd_sparse(nb_states, sd_dict)

        freq = calculate_strategies_distribution(pop_size, nb_strategies, sd)

        ref = _ref_strategies_distribution(pop_size, nb_strategies, sd_dict)
        assert np.allclose(freq, ref, atol=1e-12)

    def test_frequencies_sum_to_one(self):
        """Frequencies must sum to 1 for any valid stationary distribution."""
        pop_size, nb_strategies = 10, 3
        nb_states = calculate_nb_states(pop_size, nb_strategies)
        rng = np.random.default_rng(42)
        probs = rng.dirichlet(np.ones(nb_states))
        sd_dict = {i: float(probs[i]) for i in range(nb_states)}
        sd = _make_sd_sparse(nb_states, sd_dict)

        freq = calculate_strategies_distribution(pop_size, nb_strategies, sd)
        assert np.isclose(freq.sum(), 1.0, atol=1e-10)

    def test_sparse_sd_two_states(self):
        """Only two states with known counts; verify exact result."""
        pop_size, nb_strategies = 4, 2
        # find indices for (3,1) and (1,3)
        nb_states = calculate_nb_states(pop_size, nb_strategies)
        idx_3_1 = idx_1_3 = None
        for i in range(nb_states):
            s = sample_simplex(i, pop_size, nb_strategies)
            if list(s) == [3, 1]:
                idx_3_1 = i
            if list(s) == [1, 3]:
                idx_1_3 = i
        assert idx_3_1 is not None and idx_1_3 is not None

        sd_dict = {idx_3_1: 0.6, idx_1_3: 0.4}
        sd = _make_sd_sparse(nb_states, sd_dict)
        freq = calculate_strategies_distribution(pop_size, nb_strategies, sd)

        expected_freq0 = 0.6 * (3 / 4) + 0.4 * (1 / 4)
        expected_freq1 = 0.6 * (1 / 4) + 0.4 * (3 / 4)
        assert np.isclose(freq[0], expected_freq0, atol=1e-12)
        assert np.isclose(freq[1], expected_freq1, atol=1e-12)

    def test_three_strategies_matches_reference(self):
        pop_size, nb_strategies = 8, 3
        nb_states = calculate_nb_states(pop_size, nb_strategies)
        rng = np.random.default_rng(7)
        probs = rng.dirichlet(np.ones(nb_states))
        sd_dict = {i: float(probs[i]) for i in range(nb_states)}
        sd = _make_sd_sparse(nb_states, sd_dict)

        freq = calculate_strategies_distribution(pop_size, nb_strategies, sd)
        ref = _ref_strategies_distribution(pop_size, nb_strategies, sd_dict)
        assert np.allclose(freq, ref, atol=1e-12)


# ---------------------------------------------------------------------------
# Tests: calculate_expected_indicator
# ---------------------------------------------------------------------------

class TestCalculateExpectedIndicator:

    def test_constant_indicator_equals_one(self):
        """f(g) = 1 for all g → E[f] = 1 (probabilities sum to 1)."""
        pop_size, group_size, nb_strategies = 10, 3, 2
        nb_states = calculate_nb_states(pop_size, nb_strategies)
        rng = np.random.default_rng(0)
        probs = rng.dirichlet(np.ones(nb_states))
        sd_dict = {i: float(probs[i]) for i in range(nb_states)}
        sd = _make_sd_sparse(nb_states, sd_dict)

        result = calculate_expected_indicator(pop_size, group_size, nb_strategies, sd,
                                              lambda g: 1.0)
        assert np.isclose(result, 1.0, atol=1e-10)

    def test_constant_indicator_equals_zero(self):
        """f(g) = 0 → E[f] = 0."""
        pop_size, group_size, nb_strategies = 10, 3, 2
        nb_states = calculate_nb_states(pop_size, nb_strategies)
        sd_dict = {0: 1.0}
        sd = _make_sd_sparse(nb_states, sd_dict)

        result = calculate_expected_indicator(pop_size, group_size, nb_strategies, sd,
                                              lambda g: 0.0)
        assert np.isclose(result, 0.0, atol=1e-15)

    def test_matches_reference_two_strategies(self):
        """f(g) = g[0]/group_size (cooperation level), two strategies."""
        pop_size, group_size, nb_strategies = 10, 4, 2
        nb_states = calculate_nb_states(pop_size, nb_strategies)
        rng = np.random.default_rng(1)
        probs = rng.dirichlet(np.ones(nb_states))
        sd_dict = {i: float(probs[i]) for i in range(nb_states)}
        sd = _make_sd_sparse(nb_states, sd_dict)

        indicator = lambda g: g[0] / group_size

        result = calculate_expected_indicator(pop_size, group_size, nb_strategies, sd, indicator)
        ref = _ref_expected_indicator(pop_size, group_size, nb_strategies, sd_dict, indicator)
        assert np.isclose(result, ref, atol=1e-10)

    def test_matches_reference_three_strategies(self):
        """f(g) = 1 if g[0]+g[2] >= 2, three strategies."""
        pop_size, group_size, nb_strategies = 8, 3, 3
        nb_states = calculate_nb_states(pop_size, nb_strategies)
        rng = np.random.default_rng(2)
        probs = rng.dirichlet(np.ones(nb_states))
        sd_dict = {i: float(probs[i]) for i in range(nb_states)}
        sd = _make_sd_sparse(nb_states, sd_dict)

        indicator = lambda g: float(int(g[0]) + int(g[2]) >= 2)

        result = calculate_expected_indicator(pop_size, group_size, nb_strategies, sd, indicator)
        ref = _ref_expected_indicator(pop_size, group_size, nb_strategies, sd_dict, indicator)
        assert np.isclose(result, ref, atol=1e-10)

    def test_monomorphic_state_group_sampling(self):
        """With all cooperators in population, every group is all cooperators."""
        pop_size, group_size, nb_strategies = 10, 4, 2
        nb_states = calculate_nb_states(pop_size, nb_strategies)
        # Find all-cooperators state
        all_coop_idx = next(
            i for i in range(nb_states)
            if list(sample_simplex(i, pop_size, nb_strategies)) == [pop_size, 0]
        )
        sd = _make_sd_sparse(nb_states, {all_coop_idx: 1.0})

        # Cooperation level must be 1.0 when population is all cooperators
        result = calculate_expected_indicator(pop_size, group_size, nb_strategies, sd,
                                              lambda g: g[0] / group_size)
        assert np.isclose(result, 1.0, atol=1e-12)


# ---------------------------------------------------------------------------
# Tests: calculate_expected_group_success
# ---------------------------------------------------------------------------

class TestCalculateExpectedGroupSuccess:

    def test_matches_reference_single_contributing_strategy(self):
        """eta_G with only cooperators (strategy 0) contributing."""
        pop_size, group_size, nb_strategies = 10, 4, 2
        threshold = 2
        nb_states = calculate_nb_states(pop_size, nb_strategies)
        rng = np.random.default_rng(3)
        probs = rng.dirichlet(np.ones(nb_states))
        sd_dict = {i: float(probs[i]) for i in range(nb_states)}
        sd = _make_sd_sparse(nb_states, sd_dict)

        result = calculate_expected_group_success(
            pop_size, group_size, nb_strategies, sd,
            threshold=threshold, contributing_strategies=[0])
        ref = _ref_group_success(pop_size, group_size, nb_strategies, sd_dict,
                                 threshold=threshold, contributing=[0])
        assert np.isclose(result, ref, atol=1e-10)

    def test_matches_reference_two_contributing_strategies(self):
        """eta_G with strategies 0 and 2 both contributing (three-strategy game)."""
        pop_size, group_size, nb_strategies = 8, 3, 3
        threshold = 2
        nb_states = calculate_nb_states(pop_size, nb_strategies)
        rng = np.random.default_rng(4)
        probs = rng.dirichlet(np.ones(nb_states))
        sd_dict = {i: float(probs[i]) for i in range(nb_states)}
        sd = _make_sd_sparse(nb_states, sd_dict)

        result = calculate_expected_group_success(
            pop_size, group_size, nb_strategies, sd,
            threshold=threshold, contributing_strategies=[0, 2])
        ref = _ref_group_success(pop_size, group_size, nb_strategies, sd_dict,
                                 threshold=threshold, contributing=[0, 2])
        assert np.isclose(result, ref, atol=1e-10)

    def test_threshold_zero_always_succeeds(self):
        """threshold=0 → every group succeeds → eta_G = 1."""
        pop_size, group_size, nb_strategies = 10, 3, 2
        nb_states = calculate_nb_states(pop_size, nb_strategies)
        rng = np.random.default_rng(5)
        probs = rng.dirichlet(np.ones(nb_states))
        sd_dict = {i: float(probs[i]) for i in range(nb_states)}
        sd = _make_sd_sparse(nb_states, sd_dict)

        result = calculate_expected_group_success(
            pop_size, group_size, nb_strategies, sd,
            threshold=0, contributing_strategies=[0])
        assert np.isclose(result, 1.0, atol=1e-10)

    def test_threshold_exceeds_group_size_never_succeeds(self):
        """threshold > group_size → no group can ever succeed → eta_G = 0."""
        pop_size, group_size, nb_strategies = 10, 3, 2
        nb_states = calculate_nb_states(pop_size, nb_strategies)
        rng = np.random.default_rng(6)
        probs = rng.dirichlet(np.ones(nb_states))
        sd_dict = {i: float(probs[i]) for i in range(nb_states)}
        sd = _make_sd_sparse(nb_states, sd_dict)

        result = calculate_expected_group_success(
            pop_size, group_size, nb_strategies, sd,
            threshold=group_size + 1, contributing_strategies=[0])
        assert np.isclose(result, 0.0, atol=1e-12)

    def test_monomorphic_all_defectors_zero_success(self):
        """All-defectors population → zero cooperators in any group → eta_G = 0 for threshold > 0."""
        pop_size, group_size, nb_strategies = 10, 4, 2
        nb_states = calculate_nb_states(pop_size, nb_strategies)
        # strategy 0 = cooperator; find all-defectors state (s[0] = 0)
        all_def_idx = next(
            i for i in range(nb_states)
            if list(sample_simplex(i, pop_size, nb_strategies)) == [0, pop_size]
        )
        sd = _make_sd_sparse(nb_states, {all_def_idx: 1.0})

        result = calculate_expected_group_success(
            pop_size, group_size, nb_strategies, sd,
            threshold=1, contributing_strategies=[0])
        assert np.isclose(result, 0.0, atol=1e-12)

    def test_monomorphic_all_cooperators_full_success(self):
        """All-cooperators population → all groups full cooperators → eta_G = 1 for any threshold ≤ group_size."""
        pop_size, group_size, nb_strategies = 10, 4, 2
        nb_states = calculate_nb_states(pop_size, nb_strategies)
        all_coop_idx = next(
            i for i in range(nb_states)
            if list(sample_simplex(i, pop_size, nb_strategies)) == [pop_size, 0]
        )
        sd = _make_sd_sparse(nb_states, {all_coop_idx: 1.0})

        result = calculate_expected_group_success(
            pop_size, group_size, nb_strategies, sd,
            threshold=group_size, contributing_strategies=[0])
        assert np.isclose(result, 1.0, atol=1e-12)

    def test_result_bounded_in_zero_one(self):
        """eta_G must always be in [0, 1]."""
        pop_size, group_size, nb_strategies = 12, 5, 3
        threshold = 3
        nb_states = calculate_nb_states(pop_size, nb_strategies)
        rng = np.random.default_rng(99)
        probs = rng.dirichlet(np.ones(nb_states))
        sd_dict = {i: float(probs[i]) for i in range(nb_states)}
        sd = _make_sd_sparse(nb_states, sd_dict)

        result = calculate_expected_group_success(
            pop_size, group_size, nb_strategies, sd,
            threshold=threshold, contributing_strategies=[0, 1])
        assert 0.0 <= result <= 1.0 + 1e-10


# ---------------------------------------------------------------------------
# Tests: calculate_expected_payoff
# ---------------------------------------------------------------------------

class TestCalculateExpectedPayoff:

    def _crd_payoff_matrix(self, group_size, nb_strategies, endowment, cost, risk, threshold):
        """Build a OneShotCRD-style payoff matrix analytically for 2 strategies (C, D)."""
        from egttools import calculate_nb_states, sample_simplex
        nb_group_configs = calculate_nb_states(group_size, nb_strategies)
        # rows: strategy, cols: group composition index
        payoff_matrix = np.zeros((nb_strategies, nb_group_configs))
        for g_idx in range(nb_group_configs):
            group = sample_simplex(g_idx, group_size, nb_strategies)
            nb_coop = int(group[0])
            if nb_coop >= threshold:
                payoff_matrix[0, g_idx] = endowment * (1 - cost)   # cooperator success
                payoff_matrix[1, g_idx] = endowment                 # defector success
            else:
                payoff_matrix[0, g_idx] = endowment * (1 - risk) - endowment * cost  # coop failure
                payoff_matrix[1, g_idx] = endowment * (1 - risk)                     # def failure
        return payoff_matrix

    def test_matches_reference_two_strategies(self):
        pop_size, group_size, nb_strategies = 10, 4, 2
        endowment, cost, risk, threshold = 1.0, 0.1, 0.5, 2
        nb_states = calculate_nb_states(pop_size, nb_strategies)
        rng = np.random.default_rng(10)
        probs = rng.dirichlet(np.ones(nb_states))
        sd_dict = {i: float(probs[i]) for i in range(nb_states)}
        sd = _make_sd_sparse(nb_states, sd_dict)

        payoff_matrix = self._crd_payoff_matrix(group_size, nb_strategies, endowment, cost, risk, threshold)

        result = calculate_expected_payoff(pop_size, group_size, nb_strategies, sd, payoff_matrix)
        ref = _ref_expected_payoff(pop_size, group_size, nb_strategies, sd_dict, payoff_matrix)
        assert np.isclose(result, ref, atol=1e-10)

    def test_monomorphic_all_defectors(self):
        """All-defectors: every group has 0 cooperators → always failure payoff for defector."""
        pop_size, group_size, nb_strategies = 10, 4, 2
        endowment, cost, risk, threshold = 1.0, 0.1, 0.5, 1
        nb_states = calculate_nb_states(pop_size, nb_strategies)
        all_def_idx = next(
            i for i in range(nb_states)
            if list(sample_simplex(i, pop_size, nb_strategies)) == [0, pop_size]
        )
        sd = _make_sd_sparse(nb_states, {all_def_idx: 1.0})
        payoff_matrix = self._crd_payoff_matrix(group_size, nb_strategies, endowment, cost, risk, threshold)

        result = calculate_expected_payoff(pop_size, group_size, nb_strategies, sd, payoff_matrix)
        # All-defectors: group is always (0, group_size), nb_coop=0 < threshold → failure
        expected = endowment * (1 - risk)
        assert np.isclose(result, expected, atol=1e-12)

    def test_monomorphic_all_cooperators_above_threshold(self):
        """All-cooperators: every group has group_size cooperators ≥ threshold → success payoff."""
        pop_size, group_size, nb_strategies = 10, 4, 2
        endowment, cost, risk, threshold = 1.0, 0.1, 0.5, 2
        nb_states = calculate_nb_states(pop_size, nb_strategies)
        all_coop_idx = next(
            i for i in range(nb_states)
            if list(sample_simplex(i, pop_size, nb_strategies)) == [pop_size, 0]
        )
        sd = _make_sd_sparse(nb_states, {all_coop_idx: 1.0})
        payoff_matrix = self._crd_payoff_matrix(group_size, nb_strategies, endowment, cost, risk, threshold)

        result = calculate_expected_payoff(pop_size, group_size, nb_strategies, sd, payoff_matrix)
        # All-cooperators: group is (group_size, 0), all cooperators get success payoff
        expected = endowment * (1 - cost)
        assert np.isclose(result, expected, atol=1e-12)

    def test_consistency_with_expected_indicator(self):
        """calculate_expected_payoff must equal calculate_expected_indicator with the same f(g)."""
        pop_size, group_size, nb_strategies = 8, 3, 2
        nb_states = calculate_nb_states(pop_size, nb_strategies)
        rng = np.random.default_rng(11)
        probs = rng.dirichlet(np.ones(nb_states))
        sd_dict = {i: float(probs[i]) for i in range(nb_states)}
        sd1 = _make_sd_sparse(nb_states, sd_dict)
        sd2 = _make_sd_sparse(nb_states, sd_dict)

        payoff_matrix = np.array([
            [0.9, 0.5, 0.2, 0.0],   # cooperator payoffs for each group config
            [1.0, 0.8, 0.6, 0.3],   # defector payoffs for each group config
        ])

        result_payoff = calculate_expected_payoff(pop_size, group_size, nb_strategies, sd1, payoff_matrix)

        def indicator(g):
            g_idx = egt.calculate_state(group_size, g)
            return float(np.dot(np.array(g, dtype=float) / group_size, payoff_matrix[:, g_idx]))

        result_indicator = calculate_expected_indicator(pop_size, group_size, nb_strategies, sd2, indicator)

        assert np.isclose(result_payoff, result_indicator, atol=1e-10)


# ---------------------------------------------------------------------------
# Tests: calculate_expected_indicators (vectorized multi-indicator)
# ---------------------------------------------------------------------------

class TestCalculateExpectedIndicators:

    def test_single_indicator_matches_scalar_version(self):
        """One-element list must give the same result as calculate_expected_indicator."""
        pop_size, group_size, nb_strategies = 10, 4, 2
        nb_states = calculate_nb_states(pop_size, nb_strategies)
        rng = np.random.default_rng(20)
        probs = rng.dirichlet(np.ones(nb_states))
        sd_dict = {i: float(probs[i]) for i in range(nb_states)}
        sd1 = _make_sd_sparse(nb_states, sd_dict)
        sd2 = _make_sd_sparse(nb_states, sd_dict)

        fn = lambda g: g[0] / group_size

        scalar = calculate_expected_indicator(pop_size, group_size, nb_strategies, sd1, fn)
        vec = calculate_expected_indicators(pop_size, group_size, nb_strategies, sd2, [fn])

        assert vec.shape == (1,)
        assert np.isclose(vec[0], scalar, atol=1e-12)

    def test_each_element_matches_independent_scalar_call(self):
        """Every element of the result must equal the corresponding scalar call."""
        pop_size, group_size, nb_strategies = 8, 3, 2
        threshold = 2
        nb_states = calculate_nb_states(pop_size, nb_strategies)
        rng = np.random.default_rng(21)
        probs = rng.dirichlet(np.ones(nb_states))
        sd_dict = {i: float(probs[i]) for i in range(nb_states)}

        fns = [
            lambda g: g[0] / group_size,
            lambda g: float(int(g[0]) >= threshold),
            lambda _: 1.0,
            lambda _: 0.0,
        ]

        # build four independent sparse matrices (each call consumes its own)
        sds = [_make_sd_sparse(nb_states, sd_dict) for _ in range(len(fns) + 1)]

        vec = calculate_expected_indicators(pop_size, group_size, nb_strategies, sds[0], fns)

        assert vec.shape == (len(fns),)
        for k, fn in enumerate(fns):
            expected = calculate_expected_indicator(pop_size, group_size, nb_strategies, sds[k + 1], fn)
            assert np.isclose(vec[k], expected, atol=1e-12), \
                f"indicator {k}: vectorized={vec[k]:.15f}, scalar={expected:.15f}"

    def test_three_strategies_multiple_indicators(self):
        """Three strategies, three indicators including a multi-contributing-strategy one."""
        pop_size, group_size, nb_strategies = 8, 3, 3
        threshold = 2
        nb_states = calculate_nb_states(pop_size, nb_strategies)
        rng = np.random.default_rng(22)
        probs = rng.dirichlet(np.ones(nb_states))
        sd_dict = {i: float(probs[i]) for i in range(nb_states)}

        fns = [
            lambda g: g[0] / group_size,                           # cooperation level (strategy 0)
            lambda g: float(int(g[0]) + int(g[2]) >= threshold),   # success: strategies 0 and 2
            lambda g: (g[1]) / group_size,                         # frequency strategy 1
        ]

        sds = [_make_sd_sparse(nb_states, sd_dict) for _ in range(len(fns) + 1)]
        vec = calculate_expected_indicators(pop_size, group_size, nb_strategies, sds[0], fns)

        assert vec.shape == (len(fns),)
        for k, fn in enumerate(fns):
            expected = calculate_expected_indicator(
                pop_size, group_size, nb_strategies, sds[k + 1], fn)
            assert np.isclose(vec[k], expected, atol=1e-12), \
                f"indicator {k}: vectorized={vec[k]:.15f}, scalar={expected:.15f}"

    def test_constant_indicators(self):
        """f_0=1, f_1=0 → results must be [1, 0] regardless of SD."""
        pop_size, group_size, nb_strategies = 10, 3, 2
        nb_states = calculate_nb_states(pop_size, nb_strategies)
        rng = np.random.default_rng(23)
        probs = rng.dirichlet(np.ones(nb_states))
        sd_dict = {i: float(probs[i]) for i in range(nb_states)}
        sd = _make_sd_sparse(nb_states, sd_dict)

        vec = calculate_expected_indicators(
            pop_size, group_size, nb_strategies, sd,
            [lambda _: 1.0, lambda _: 0.0])

        assert np.isclose(vec[0], 1.0, atol=1e-10)
        assert np.isclose(vec[1], 0.0, atol=1e-15)

    def test_empty_indicator_list_returns_empty_array(self):
        """Passing an empty list should return a zero-length array without error."""
        pop_size, group_size, nb_strategies = 6, 2, 2
        nb_states = calculate_nb_states(pop_size, nb_strategies)
        sd = _make_sd_sparse(nb_states, {0: 1.0})

        vec = calculate_expected_indicators(pop_size, group_size, nb_strategies, sd, [])
        assert vec.shape == (0,)


# ---------------------------------------------------------------------------
# Helpers shared by precomputed tests
# ---------------------------------------------------------------------------

def _build_indicator_matrix(group_size, nb_strategies, fns):
    """Build the (nb_group_configs, K) indicator matrix from a list of callables."""
    nb_group_configs = calculate_nb_states(group_size, nb_strategies)
    mat = np.zeros((nb_group_configs, len(fns)), dtype=float)
    for g_idx in range(nb_group_configs):
        group = sample_simplex(g_idx, group_size, nb_strategies)
        for k, fn in enumerate(fns):
            mat[g_idx, k] = fn(group)
    return mat


# ---------------------------------------------------------------------------
# Tests: calculate_expected_indicators_precomputed
# ---------------------------------------------------------------------------

class TestCalculateExpectedIndicatorsPrecomputed:

    def test_matches_callable_version_single_indicator(self):
        """Precomputed path must agree with the callable version for one indicator."""
        pop_size, group_size, nb_strategies = 10, 4, 2
        nb_states = calculate_nb_states(pop_size, nb_strategies)
        rng = np.random.default_rng(30)
        probs = rng.dirichlet(np.ones(nb_states))
        sd_dict = {i: float(probs[i]) for i in range(nb_states)}
        sd1 = _make_sd_sparse(nb_states, sd_dict)
        sd2 = _make_sd_sparse(nb_states, sd_dict)

        fns = [lambda g: float(g[0] >= 2)]
        mat = _build_indicator_matrix(group_size, nb_strategies, fns)

        result_pre = calculate_expected_indicators_precomputed(
            pop_size, group_size, nb_strategies, sd1, mat)
        result_call = calculate_expected_indicators(
            pop_size, group_size, nb_strategies, sd2, fns)

        assert result_pre.shape == (1,)
        assert np.isclose(result_pre[0], result_call[0], atol=1e-12)

    def test_matches_callable_version_multiple_indicators(self):
        """All K elements must agree with the callable version."""
        pop_size, group_size, nb_strategies = 8, 3, 3
        threshold = 2
        nb_states = calculate_nb_states(pop_size, nb_strategies)
        rng = np.random.default_rng(31)
        probs = rng.dirichlet(np.ones(nb_states))
        sd_dict = {i: float(probs[i]) for i in range(nb_states)}
        sd1 = _make_sd_sparse(nb_states, sd_dict)
        sd2 = _make_sd_sparse(nb_states, sd_dict)

        fns = [
            lambda g: g[0] / group_size,
            lambda g: float(int(g[0]) + int(g[2]) >= threshold),
            lambda _: 1.0,
        ]
        mat = _build_indicator_matrix(group_size, nb_strategies, fns)

        result_pre = calculate_expected_indicators_precomputed(
            pop_size, group_size, nb_strategies, sd1, mat)
        result_call = calculate_expected_indicators(
            pop_size, group_size, nb_strategies, sd2, fns)

        assert result_pre.shape == result_call.shape
        assert np.allclose(result_pre, result_call, atol=1e-12)

    def test_matches_reference_implementation(self):
        """Precomputed path must agree with the pure-Python reference."""
        pop_size, group_size, nb_strategies = 10, 4, 2
        threshold = 2
        nb_states = calculate_nb_states(pop_size, nb_strategies)
        rng = np.random.default_rng(32)
        probs = rng.dirichlet(np.ones(nb_states))
        sd_dict = {i: float(probs[i]) for i in range(nb_states)}
        sd = _make_sd_sparse(nb_states, sd_dict)

        fns = [
            lambda g: float(int(g[0]) >= threshold),
            lambda g: g[0] / group_size,
        ]
        mat = _build_indicator_matrix(group_size, nb_strategies, fns)

        result = calculate_expected_indicators_precomputed(
            pop_size, group_size, nb_strategies, sd, mat)

        for k, fn in enumerate(fns):
            ref = _ref_expected_indicator(pop_size, group_size, nb_strategies, sd_dict, fn)
            assert np.isclose(result[k], ref, atol=1e-10), \
                f"indicator {k}: precomputed={result[k]:.15f}, ref={ref:.15f}"

    def test_constant_columns(self):
        """Column of all-ones → result 1.0; column of all-zeros → result 0.0."""
        pop_size, group_size, nb_strategies = 10, 3, 2
        nb_states = calculate_nb_states(pop_size, nb_strategies)
        nb_group_configs = calculate_nb_states(group_size, nb_strategies)
        rng = np.random.default_rng(33)
        probs = rng.dirichlet(np.ones(nb_states))
        sd_dict = {i: float(probs[i]) for i in range(nb_states)}
        sd = _make_sd_sparse(nb_states, sd_dict)

        mat = np.column_stack([
            np.ones(nb_group_configs),
            np.zeros(nb_group_configs),
        ])
        result = calculate_expected_indicators_precomputed(
            pop_size, group_size, nb_strategies, sd, mat)

        assert np.isclose(result[0], 1.0, atol=1e-10)
        assert np.isclose(result[1], 0.0, atol=1e-15)

    def test_empty_matrix_returns_empty_array(self):
        """Zero-column matrix should return a zero-length array."""
        pop_size, group_size, nb_strategies = 6, 2, 2
        nb_states = calculate_nb_states(pop_size, nb_strategies)
        nb_group_configs = calculate_nb_states(group_size, nb_strategies)
        sd = _make_sd_sparse(nb_states, {0: 1.0})

        mat = np.zeros((nb_group_configs, 0))
        result = calculate_expected_indicators_precomputed(
            pop_size, group_size, nb_strategies, sd, mat)
        assert result.shape == (0,)

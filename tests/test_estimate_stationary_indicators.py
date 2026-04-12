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
Tests for PairwiseComparisonNumerical.estimate_stationary_indicators
and .estimate_stationary_indicators_precomputed.

Accuracy tests validate numerical estimates against analytical expectations
computed from the exact stationary distribution obtained via
``egt.analytical.PairwiseComparison.calculate_transition_matrix`` +
``egt.utils.calculate_stationary_distribution``.

The analytical reference is independent of any C++ indicator estimation code;
it is a plain sum  ``sum_s sd[s] * f(state_s)`` (state-level) or a
double loop with hypergeometric marginalisation (group-level), computed using
``egt.calculate_expected_state_indicator`` and ``egt.calculate_expected_indicator``
respectively.
"""

import warnings

import numpy as np
import pytest
import scipy.sparse as sp

egt = pytest.importorskip("egttools")

PairwiseComparisonNumerical = egt.numerical.PairwiseComparisonNumerical
Matrix2PlayerGameHolder = egt.games.Matrix2PlayerGameHolder
MatrixNPlayerGameHolder = egt.games.MatrixNPlayerGameHolder
StationaryIndicatorResult = egt.numerical.StationaryIndicatorResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def hawk_dove_game():
    """Hawk-Dove payoff matrix (b=4, c=6)."""
    b, c = 4.0, 6.0
    return np.array([[(b - c) / 2, b], [0, b / 2]])


@pytest.fixture(scope="module")
def hd_game_obj(hawk_dove_game):
    return Matrix2PlayerGameHolder(2, hawk_dove_game)


@pytest.fixture(scope="module")
def hd_solver(hd_game_obj):
    """PairwiseComparisonNumerical for Hawk-Dove with Z=20."""
    return PairwiseComparisonNumerical(20, hd_game_obj, 10_000)


@pytest.fixture(scope="module")
def hd_analytical(hd_game_obj):
    """Exact stationary distribution for Hawk-Dove (Z=20, beta=1, mu=0.05).

    Returns a dict with keys: sd (dense array), sd_sparse (csr_matrix),
    Z, beta, mu, nb_strategies.
    """
    Z, beta, mu = 20, 1.0, 0.05
    evolver = egt.analytical.PairwiseComparison(Z, hd_game_obj)
    T = evolver.calculate_transition_matrix(beta, mu)
    sd = egt.utils.calculate_stationary_distribution(T.T)   # shape (nb_states,)
    return {
        "sd": sd,
        "sd_sparse": sp.csr_matrix(sd),
        "Z": Z,
        "beta": beta,
        "mu": mu,
        "nb_strategies": 2,
    }


@pytest.fixture(scope="module")
def hd_numerical_result(hd_solver, hd_analytical):
    """Single simulation run shared across accuracy tests.

    Two state-level indicators: fraction of Hawks (s0) and Doves (s1).
    Uses enough runs for tight confidence intervals.
    """
    Z = hd_analytical["Z"]
    beta = hd_analytical["beta"]
    mu = hd_analytical["mu"]
    return hd_solver.estimate_stationary_indicators(
        [lambda s: float(s[0]) / Z, lambda s: float(s[1]) / Z],
        nb_runs=500,
        nb_generations=3_000,
        transitory=300,
        beta=beta,
        mu=mu,
        verbose=True,
    )


@pytest.fixture(scope="module")
def npg_payoff_matrix():
    """Linear public-goods payoff matrix: 2 strategies (D=0, C=1), group_size=3.

    With r=2, cost=1, N=3, the payoffs for j cooperators in the group are:
      Defector (row 0): r * j / N           =  2*j/3
      Cooperator (row 1): r * j / N - cost  =  2*j/3 - 1

    Columns follow sample_simplex(i, 3, 2):
      i=0 → [0,3] (j=3): D=2.0,   C=1.0
      i=1 → [1,2] (j=2): D=4/3,   C=1/3
      i=2 → [2,1] (j=1): D=2/3,   C=-1/3
      i=3 → [3,0] (j=0): D=0.0,   C=-1.0
    """
    r, cost, N = 2.0, 1.0, 3
    nb_configs = egt.calculate_nb_states(3, 2)   # = 4
    payoffs = np.zeros((2, nb_configs))
    for i in range(nb_configs):
        gc = egt.sample_simplex(i, 3, 2)          # [n_D, n_C]
        j = float(gc[1])                           # number of cooperators
        payoffs[0, i] = r * j / N                 # defector payoff
        payoffs[1, i] = r * j / N - cost          # cooperator payoff
    return payoffs


@pytest.fixture(scope="module")
def npg_game_obj(npg_payoff_matrix):
    """MatrixNPlayerGameHolder for the linear PGG (group_size=3, 2 strategies)."""
    return MatrixNPlayerGameHolder(2, 3, npg_payoff_matrix)


@pytest.fixture(scope="module")
def npg_solver(npg_game_obj):
    """PairwiseComparisonNumerical for the N-player PGG with Z=20."""
    return PairwiseComparisonNumerical(20, npg_game_obj, 10_000)


@pytest.fixture(scope="module")
def npg_analytical(npg_game_obj):
    """Exact stationary distribution for the N-player PGG (Z=20, beta=1, mu=0.05).

    Returns the same dict structure as hd_analytical.
    """
    Z, beta, mu = 20, 1.0, 0.05
    evolver = egt.analytical.PairwiseComparison(Z, npg_game_obj)
    T = evolver.calculate_transition_matrix(beta, mu)
    sd = egt.utils.calculate_stationary_distribution(T.T)
    return {
        "sd": sd,
        "sd_sparse": sp.csr_matrix(sd),
        "Z": Z,
        "beta": beta,
        "mu": mu,
        "nb_strategies": 2,
    }


@pytest.fixture(scope="module")
def npg_numerical_result(npg_solver, npg_analytical):
    """State-level indicator estimates for the N-player PGG.

    Two indicators: fraction of Defectors (s0) and Cooperators (s1).
    """
    Z = npg_analytical["Z"]
    beta = npg_analytical["beta"]
    mu = npg_analytical["mu"]
    return npg_solver.estimate_stationary_indicators(
        [lambda s: float(s[0]) / Z, lambda s: float(s[1]) / Z],
        nb_runs=500,
        nb_generations=3_000,
        transitory=300,
        beta=beta,
        mu=mu,
        verbose=True,
    )


@pytest.fixture(scope="module")
def npg_group_result(npg_solver, npg_analytical):
    """Group-level indicator estimates for the N-player PGG (group_size=3).

    Indicator: fraction of Defectors (strategy 0) in the sampled group.
    """
    beta = npg_analytical["beta"]
    mu = npg_analytical["mu"]
    return npg_solver.estimate_stationary_indicators(
        lambda g: float(g[0]) / 3,
        nb_runs=500,
        nb_generations=3_000,
        transitory=300,
        beta=beta,
        mu=mu,
        indicator_type="group",
        group_size=3,
    )


# ---------------------------------------------------------------------------
# Interface tests
# ---------------------------------------------------------------------------

class TestReturnType:
    def test_returns_stationary_indicator_result(self, hd_solver, hd_analytical):
        result = hd_solver.estimate_stationary_indicators(
            lambda s: float(s[0]) / hd_analytical["Z"],
            nb_runs=10,
            nb_generations=500,
            transitory=50,
            beta=hd_analytical["beta"],
            mu=hd_analytical["mu"],
        )
        assert isinstance(result, StationaryIndicatorResult)

    def test_mean_shape_single_indicator(self, hd_solver, hd_analytical):
        result = hd_solver.estimate_stationary_indicators(
            lambda s: float(s[0]) / hd_analytical["Z"],
            nb_runs=10,
            nb_generations=500,
            transitory=50,
            beta=hd_analytical["beta"],
            mu=hd_analytical["mu"],
        )
        assert result.mean.shape == (1,)

    def test_mean_shape_two_indicators(self, hd_numerical_result):
        assert hd_numerical_result.mean.shape == (2,)

    def test_ci_shape(self, hd_numerical_result):
        ci_low, ci_high = hd_numerical_result.confidence_interval
        assert ci_low.shape == (2,)
        assert ci_high.shape == (2,)

    def test_ci_ordering(self, hd_numerical_result):
        ci_low, ci_high = hd_numerical_result.confidence_interval
        assert np.all(ci_low <= ci_high)

    def test_nb_runs_used(self, hd_numerical_result):
        assert hd_numerical_result.nb_runs_used == 500

    def test_per_run_values_verbose_true(self, hd_numerical_result):
        assert hd_numerical_result.per_run_values is not None
        assert hd_numerical_result.per_run_values.shape == (500, 2)

    def test_per_run_values_verbose_false(self, hd_solver, hd_analytical):
        result = hd_solver.estimate_stationary_indicators(
            lambda s: float(s[0]) / hd_analytical["Z"],
            nb_runs=10,
            nb_generations=500,
            transitory=50,
            beta=hd_analytical["beta"],
            mu=hd_analytical["mu"],
            verbose=False,
        )
        assert result.per_run_values is None

    def test_converged_false_without_tolerance(self, hd_numerical_result):
        assert hd_numerical_result.converged is False

    def test_mean_in_unit_interval(self, hd_numerical_result):
        assert np.all(hd_numerical_result.mean >= 0.0)
        assert np.all(hd_numerical_result.mean <= 1.0)

    def test_accepts_list_of_callables(self, hd_solver, hd_analytical):
        Z = hd_analytical["Z"]
        result = hd_solver.estimate_stationary_indicators(
            [lambda s: float(s[0]) / Z, lambda s: float(s[1]) / Z],
            nb_runs=10,
            nb_generations=500,
            transitory=50,
            beta=hd_analytical["beta"],
            mu=hd_analytical["mu"],
        )
        assert result.mean.shape == (2,)

    def test_accepts_tuple_of_callables(self, hd_solver, hd_analytical):
        Z = hd_analytical["Z"]
        result = hd_solver.estimate_stationary_indicators(
            (lambda s: float(s[0]) / Z,),
            nb_runs=10,
            nb_generations=500,
            transitory=50,
            beta=hd_analytical["beta"],
            mu=hd_analytical["mu"],
        )
        assert result.mean.shape == (1,)


class TestInvalidInputs:
    def test_invalid_indicator_type_raises(self, hd_solver, hd_analytical):
        with pytest.raises(Exception):
            hd_solver.estimate_stationary_indicators(
                lambda s: 0.0,
                nb_runs=5,
                nb_generations=100,
                transitory=10,
                beta=hd_analytical["beta"],
                mu=hd_analytical["mu"],
                indicator_type="invalid",
            )

    def test_group_type_without_group_size_raises(self, hd_solver, hd_analytical):
        with pytest.raises(Exception):
            hd_solver.estimate_stationary_indicators(
                lambda g: float(g[0]),
                nb_runs=5,
                nb_generations=100,
                transitory=10,
                beta=hd_analytical["beta"],
                mu=hd_analytical["mu"],
                indicator_type="group",
                # group_size omitted → should raise
            )

    def test_group_type_with_2player_game_raises(self, hd_solver, hd_analytical):
        """A 2-player game (square payoff matrix) is incompatible with group_size>2.

        The payoff matrix of a 2-player game has nb_strategies columns, while a
        proper N-player game needs stars_bars(group_size, nb_strategies) columns.
        Passing group_size=3 to a 2-player solver must raise an error.
        """
        with pytest.raises(Exception):
            hd_solver.estimate_stationary_indicators(
                lambda g: float(g[0]) / 3,
                nb_runs=5,
                nb_generations=100,
                transitory=10,
                beta=hd_analytical["beta"],
                mu=hd_analytical["mu"],
                indicator_type="group",
                group_size=3,   # incompatible with 2-player game → must raise
            )

    def test_non_callable_raises(self, hd_solver, hd_analytical):
        with pytest.raises(Exception):
            hd_solver.estimate_stationary_indicators(
                42,   # not a callable
                nb_runs=5,
                nb_generations=100,
                transitory=10,
                beta=hd_analytical["beta"],
                mu=hd_analytical["mu"],
            )

    def test_small_mu_warns(self, hd_game_obj):
        solver = PairwiseComparisonNumerical(20, hd_game_obj, 1000)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            solver.estimate_stationary_indicators(
                lambda s: float(s[0]),
                nb_runs=2,
                nb_generations=50,
                transitory=5,
                beta=1.0,
                mu=1e-3,  # expected mutations = 1e-3 * 45 = 0.045 << 10
            )
        assert any(issubclass(warning.category, UserWarning) for warning in w)


# ---------------------------------------------------------------------------
# Accuracy tests: numerical vs analytical
# ---------------------------------------------------------------------------

class TestAccuracyStateLevel:
    """Compare numerical estimates to exact analytical expectations.

    Tolerance: 3 % absolute.  The analytical value must also lie within the
    reported 95 % bootstrap CI (may fail ~5 % of the time by construction,
    but the simulation parameters are chosen to give very tight intervals so
    in practice this should be robust).
    """

    def test_hawk_fraction_close_to_analytical(self, hd_numerical_result, hd_analytical):
        sd_sparse = hd_analytical["sd_sparse"]
        Z = hd_analytical["Z"]
        analytical = egt.calculate_expected_state_indicator(
            Z, 2, sd_sparse, lambda s: float(s[0]) / Z
        )
        numerical = hd_numerical_result.mean[0]
        assert abs(numerical - analytical) < 0.03, (
            f"Hawk fraction estimate {numerical:.4f} deviates from analytical "
            f"{analytical:.4f} by more than 3 %"
        )

    def test_dove_fraction_close_to_analytical(self, hd_numerical_result, hd_analytical):
        sd_sparse = hd_analytical["sd_sparse"]
        Z = hd_analytical["Z"]
        analytical = egt.calculate_expected_state_indicator(
            Z, 2, sd_sparse, lambda s: float(s[1]) / Z
        )
        numerical = hd_numerical_result.mean[1]
        assert abs(numerical - analytical) < 0.03, (
            f"Dove fraction estimate {numerical:.4f} deviates from analytical "
            f"{analytical:.4f} by more than 3 %"
        )

    def test_hawk_fraction_ci_contains_analytical(self, hd_numerical_result, hd_analytical):
        sd_sparse = hd_analytical["sd_sparse"]
        Z = hd_analytical["Z"]
        analytical = egt.calculate_expected_state_indicator(
            Z, 2, sd_sparse, lambda s: float(s[0]) / Z
        )
        ci_low, ci_high = hd_numerical_result.confidence_interval
        assert ci_low[0] <= analytical <= ci_high[0], (
            f"Analytical value {analytical:.4f} not within 95% CI "
            f"[{ci_low[0]:.4f}, {ci_high[0]:.4f}]"
        )

    def test_two_fractions_sum_to_one(self, hd_numerical_result):
        """Hawk + Dove fractions must sum to exactly 1 for any state."""
        total = hd_numerical_result.mean.sum()
        assert abs(total - 1.0) < 1e-10, (
            f"Sum of strategy fractions {total} differs from 1"
        )

    def test_per_run_values_mean_matches_reported_mean(self, hd_numerical_result):
        """Grand mean computed from per_run_values must equal result.mean."""
        computed = hd_numerical_result.per_run_values.mean(axis=0)
        np.testing.assert_allclose(computed, hd_numerical_result.mean, rtol=1e-10)


class TestAccuracyGroupLevel:
    """Group-level indicator: fraction of Defectors in a sampled group of size 3.

    Uses a linear public-goods game (2 strategies D/C, group_size=3) so that
    the solver's payoff matrix is consistent with the requested group_size.
    """

    def test_group_defector_fraction_close_to_analytical(self, npg_group_result, npg_analytical):
        sd_sparse = npg_analytical["sd_sparse"]
        Z = npg_analytical["Z"]
        analytical = egt.calculate_expected_indicator(
            Z, 3, 2, sd_sparse, lambda g: float(g[0]) / 3
        )
        numerical = npg_group_result.mean[0]
        assert abs(numerical - analytical) < 0.03, (
            f"Group-level Defector fraction estimate {numerical:.4f} deviates from "
            f"analytical {analytical:.4f} by more than 3 %"
        )

    def test_group_defector_fraction_ci_contains_analytical(self, npg_group_result, npg_analytical):
        sd_sparse = npg_analytical["sd_sparse"]
        Z = npg_analytical["Z"]
        analytical = egt.calculate_expected_indicator(
            Z, 3, 2, sd_sparse, lambda g: float(g[0]) / 3
        )
        ci_low, ci_high = npg_group_result.confidence_interval
        assert ci_low[0] <= analytical <= ci_high[0], (
            f"Analytical group value {analytical:.4f} not within 95% CI "
            f"[{ci_low[0]:.4f}, {ci_high[0]:.4f}]"
        )

    def test_state_and_group_expectations_agree(self, npg_group_result, npg_numerical_result):
        """E[Defector fraction in group] should equal E[Defector fraction in population].

        This is a fundamental property of the multivariate hypergeometric distribution:
        for any population state s, E_g[g[i]/n | s] = s[i]/N.  Both estimates come
        from the same N-player PGG dynamics, so their expectations under the shared
        stationary distribution must match.
        """
        state_defector = npg_numerical_result.mean[0]
        group_defector = npg_group_result.mean[0]
        assert abs(group_defector - state_defector) < 0.03, (
            f"State-level ({state_defector:.4f}) and group-level ({group_defector:.4f}) "
            "Defector fractions differ by more than 3 %"
        )


# ---------------------------------------------------------------------------
# Precomputed method tests
# ---------------------------------------------------------------------------

class TestPrecomputedMethod:
    def test_precomputed_returns_correct_shape(self, hd_solver, hd_analytical):
        Z = hd_analytical["Z"]
        nb_strategies = hd_analytical["nb_strategies"]
        nb_states = hd_solver.nb_states

        # Build indicator matrix manually
        indicator_matrix = np.empty((nb_states, 2))
        for s in range(nb_states):
            state = egt.sample_simplex(s, Z, nb_strategies)
            indicator_matrix[s, 0] = float(state[0]) / Z
            indicator_matrix[s, 1] = float(state[1]) / Z

        per_run = hd_solver.estimate_stationary_indicators_precomputed(
            nb_runs=50,
            nb_generations=1_000,
            transitory=100,
            beta=hd_analytical["beta"],
            mu=hd_analytical["mu"],
            indicator_values=indicator_matrix,
        )
        assert per_run.ndim == 2
        assert per_run.shape == (50, 2)

    def test_precomputed_values_in_unit_interval(self, hd_solver, hd_analytical):
        Z = hd_analytical["Z"]
        nb_strategies = hd_analytical["nb_strategies"]
        nb_states = hd_solver.nb_states

        indicator_matrix = np.array([
            [float(egt.sample_simplex(s, Z, nb_strategies)[0]) / Z]
            for s in range(nb_states)
        ])
        per_run = hd_solver.estimate_stationary_indicators_precomputed(
            nb_runs=30,
            nb_generations=500,
            transitory=50,
            beta=hd_analytical["beta"],
            mu=hd_analytical["mu"],
            indicator_values=indicator_matrix,
        )
        assert np.all(per_run >= 0.0)
        assert np.all(per_run <= 1.0)

    def test_precomputed_mean_agrees_with_highlevel(self, hd_solver, hd_analytical):
        """Mean of precomputed method should be within 3 % of high-level method."""
        Z = hd_analytical["Z"]
        nb_strategies = hd_analytical["nb_strategies"]
        nb_states = hd_solver.nb_states
        beta = hd_analytical["beta"]
        mu = hd_analytical["mu"]

        indicator_matrix = np.array([
            [float(egt.sample_simplex(s, Z, nb_strategies)[0]) / Z]
            for s in range(nb_states)
        ])

        nb_runs = 300
        per_run = hd_solver.estimate_stationary_indicators_precomputed(
            nb_runs=nb_runs,
            nb_generations=2_000,
            transitory=200,
            beta=beta,
            mu=mu,
            indicator_values=indicator_matrix,
        )
        high_level = hd_solver.estimate_stationary_indicators(
            lambda s: float(s[0]) / Z,
            nb_runs=nb_runs,
            nb_generations=2_000,
            transitory=200,
            beta=beta,
            mu=mu,
        )
        assert abs(per_run.mean() - high_level.mean[0]) < 0.03


# ---------------------------------------------------------------------------
# Tolerance / early stopping tests
# ---------------------------------------------------------------------------

class TestEarlyStopping:
    def test_no_tolerance_uses_all_runs(self, hd_solver, hd_analytical):
        nb_runs = 40
        result = hd_solver.estimate_stationary_indicators(
            lambda s: float(s[0]) / hd_analytical["Z"],
            nb_runs=nb_runs,
            nb_generations=500,
            transitory=50,
            beta=hd_analytical["beta"],
            mu=hd_analytical["mu"],
            tolerance=0.0,
        )
        assert result.nb_runs_used == nb_runs
        assert result.converged is False

    def test_tight_tolerance_may_stop_early(self, hd_solver, hd_analytical):
        """With a very tight tolerance and many allowed runs the simulation may
        (likely will) converge before all runs are consumed."""
        result = hd_solver.estimate_stationary_indicators(
            lambda s: float(s[0]) / hd_analytical["Z"],
            nb_runs=2_000,
            nb_generations=1_000,
            transitory=100,
            beta=hd_analytical["beta"],
            mu=hd_analytical["mu"],
            tolerance=1e-4,
        )
        # We can't assert convergence deterministically, but at least check
        # the result is self-consistent.
        assert result.nb_runs_used <= 2_000
        if result.converged:
            assert result.nb_runs_used < 2_000

    def test_check_every_respected(self, hd_solver, hd_analytical):
        """check_every controls batch granularity.  With check_every=10,
        nb_runs_used must be a multiple of 10 (unless it equals nb_runs)."""
        nb_runs = 100
        check_every = 10
        result = hd_solver.estimate_stationary_indicators(
            lambda s: float(s[0]) / hd_analytical["Z"],
            nb_runs=nb_runs,
            nb_generations=500,
            transitory=50,
            beta=hd_analytical["beta"],
            mu=hd_analytical["mu"],
            tolerance=1e-4,
            check_every=check_every,
        )
        assert result.nb_runs_used % check_every == 0 or result.nb_runs_used == nb_runs

"""Check the fixation and absorption implementations against ``docs/theory/fixation.md``.

The references are the transcriptions in ``theory_reference.py``: F 2.2 from the transition
probabilities of the pairwise comparison process, its log form, and the absorbing-chain results
of F 3.3. Monte Carlo estimators are compared with them within k standard errors
(``stats_helpers.py``), with fixed seeds, and must reject the exact value for a different beta.
"""
import itertools

import numpy as np
import pytest

import egttools as egt
from egttools.analytical import PairwiseComparison, StochDynamics
from egttools.numerical import PairwiseComparisonNumerical
from stats_helpers import DEFAULT_K, assert_detects, assert_within_k_se
from theory_reference import (absorbing_chain, edge_state, fitness_n_player, fitness_two_player,
                              fixation_probability, log_fixation_probability)

Z = 9
BETAS = [0.0, 0.5, 1.3, 5.0]
GROUP_SIZE = 3


# ---------------------------------------------------------------------------
# Games: a two-player and a three-player game with three strategies
# ---------------------------------------------------------------------------

def two_player_game(k=3, seed=3):
    a = np.random.default_rng(seed).uniform(-1, 2, size=(k, k))
    return a, egt.games.Matrix2PlayerGameHolder(k, a), (lambda n: fitness_two_player(a, n))


def n_player_game(k=3, seed=3):
    table = np.random.default_rng(seed).uniform(-1, 2, size=(k, egt.calculate_nb_states(GROUP_SIZE, k)))
    game = egt.games.MatrixNPlayerGameHolder(k, GROUP_SIZE, table)
    return table, game, (lambda n: fitness_n_player(table, GROUP_SIZE, n))


def stoch_dynamics_group_payoffs(table, k):
    """StochDynamics' group-game format: payoffs[i, j](x, N) is the payoff of an i-strategist in a
    group of N with x i-strategists (focal included) and N - x j-strategists."""

    def payoff(i, j):
        def value(x, group_size, *args):
            group = np.zeros(k, dtype=int)
            group[i], group[j] = x, group_size - x
            return table[i, egt.calculate_state(group_size, group)]
        return value

    payoffs = np.empty((k, k), dtype=object)
    for i, j in itertools.product(range(k), repeat=2):
        payoffs[i, j] = payoff(i, j)
    return payoffs


GAMES = {"2p": two_player_game, "3p": n_player_game}
PAIRS = list(itertools.permutations(range(3), 2))


# ---------------------------------------------------------------------------
# F 2.2 and F 2.3: exact fixation probabilities
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", GAMES)
@pytest.mark.parametrize("beta", BETAS)
def test_pairwise_comparison_fixation_matches_definition_2_2(name, beta):
    _, game, fit = GAMES[name]()
    pc = PairwiseComparison(Z, game)
    for a, b in PAIRS:
        expected = fixation_probability(fit, Z, 3, a, b, beta)
        assert pc.calculate_fixation_probability(a, b, beta) == pytest.approx(expected, rel=1e-12)
        assert pc.calculate_fixation_probability(a, b, beta, high_precision=True) == pytest.approx(expected, rel=1e-12)
        assert pc.calculate_log_fixation_probability(a, b, beta) == pytest.approx(np.log(expected), rel=1e-12, abs=1e-14)


@pytest.mark.parametrize("beta", BETAS)
def test_stoch_dynamics_fixation_matches_definition_2_2(beta):
    a_matrix, _, fit = two_player_game()
    sd = StochDynamics(3, a_matrix, Z)
    table, _, fit_group = n_player_game()
    sd_group = StochDynamics(3, stoch_dynamics_group_payoffs(table, 3), Z, group_size=GROUP_SIZE)
    for a, b in PAIRS:
        for dynamics, f in ((sd, fit), (sd_group, fit_group)):
            expected = fixation_probability(f, Z, 3, a, b, beta)
            assert dynamics.fixation_probability(a, b, beta) == pytest.approx(expected, rel=1e-12)
            assert dynamics.log_fixation_probability(a, b, beta) == pytest.approx(np.log(expected), rel=1e-12,
                                                                                  abs=1e-14)


@pytest.mark.parametrize("name", GAMES)
def test_neutral_fixation_is_one_over_z(name):
    _, game, _ = GAMES[name]()
    pc = PairwiseComparison(Z, game)
    for a, b in PAIRS:
        assert pc.calculate_fixation_probability(a, b, 0.0) == pytest.approx(1 / Z, rel=1e-14)


@pytest.mark.parametrize("beta", [50.0, 500.0, 5000.0])
def test_log_fixation_probability_under_strong_selection(beta):
    """F 2.3: log(rho) stays finite and exact where rho underflows to 0 or the sum overflows."""
    a_matrix, game, fit = two_player_game()
    pc = PairwiseComparison(Z, game)
    sd = StochDynamics(3, a_matrix, Z)
    for a, b in PAIRS:
        expected = log_fixation_probability(fit, Z, 3, a, b, beta)
        assert np.isfinite(expected)
        assert pc.calculate_log_fixation_probability(a, b, beta) == pytest.approx(expected, rel=1e-12, abs=1e-14)
        assert sd.log_fixation_probability(a, b, beta) == pytest.approx(expected, rel=1e-12, abs=1e-14)
        rho = pc.calculate_fixation_probability(a, b, beta)
        assert rho == pytest.approx(np.exp(expected), rel=1e-12, abs=1e-300)


def test_closed_form_agrees_with_absorbing_chain():
    """F 2.2 and F 3.3 give the same single-invader fixation probability (checks the page itself)."""
    _, _, fit = two_player_game()
    for beta in BETAS:
        transient, b_matrix, _, _ = absorbing_chain(fit, Z, 3, beta)
        rows = {tuple(n): i for i, n in enumerate(transient)}
        for a, b in PAIRS:
            single_invader = rows[tuple(edge_state(Z, 3, a, b, 1))]
            assert b_matrix[single_invader, a] == pytest.approx(fixation_probability(fit, Z, 3, a, b, beta),
                                                                rel=1e-10)


# ---------------------------------------------------------------------------
# F 4.1 - F 4.3: Monte Carlo estimators
# ---------------------------------------------------------------------------

BETA_MC = 1.3
WRONG_BETA = 0.6


def assert_power_margin(exact, wrong_exact, se, label):
    """The wrong value must be at least 2k standard errors from the exact one, so that an estimate
    within k SE of the exact value is outside k SE of the wrong one whatever the seed."""
    separation = np.max(np.abs(np.asarray(wrong_exact) - np.asarray(exact)) / np.asarray(se))
    assert separation >= 2 * DEFAULT_K, f"{label}: wrong value only {separation:.1f} SE away"


@pytest.fixture
def numerical_two_player():
    egt.Random.init_with_seed(3610063510)
    _, game, fit = two_player_game()
    return PairwiseComparisonNumerical(Z, game, 100000), fit


def test_estimate_fixation_probability_matches_definition_2_2(numerical_two_player):
    evolver, fit = numerical_two_player
    runs = 20000
    for a, b in [(0, 1), (2, 0)]:
        exact = fixation_probability(fit, Z, 3, a, b, BETA_MC)
        se = np.sqrt(exact * (1 - exact) / runs)
        wrong = fixation_probability(fit, Z, 3, a, b, WRONG_BETA)
        assert_power_margin(exact, wrong, se, f"rho({a} -> {b})")
        estimate = evolver.estimate_fixation_probability(a, b, runs, 10 ** 7, BETA_MC)
        assert_within_k_se(estimate, exact, se, label=f"rho({a} -> {b})")
        assert_detects(estimate, wrong, se)


def test_estimate_fixation_probability_discards_runs_that_are_not_absorbed(numerical_two_player):
    """F 4.1: with one update per run no run is absorbed from a single invader (Z > 2), so the estimate is 0."""
    evolver, _ = numerical_two_player
    assert evolver.estimate_fixation_probability(0, 1, 100, 1, BETA_MC) == 0.0


def test_estimate_absorption_probabilities_matches_definition_3_3(numerical_two_player):
    evolver, fit = numerical_two_player
    runs = 20000
    start = np.array([3, 3, 3])
    exact = _absorbing_row(fit, start, BETA_MC)[0]
    se = np.sqrt(exact * (1 - exact) / runs)
    wrong = _absorbing_row(fit, start, WRONG_BETA)[0]
    assert_power_margin(exact, wrong, se, "B(3, 3, 3)")
    estimate = evolver.estimate_absorption_probabilities(BETA_MC, start, runs)
    assert estimate.sum() == pytest.approx(1.0)
    assert_within_k_se(estimate, exact, se, label="B(3, 3, 3)")
    assert_detects(estimate, wrong, se)


def test_estimate_mean_absorption_time_matches_definition_3_3(numerical_two_player):
    evolver, fit = numerical_two_player
    runs = 5000
    start = np.array([3, 3, 3])
    _, mean_time, variance = _absorbing_row(fit, start, BETA_MC)
    se = np.sqrt(variance / runs)
    # The mean time changes slowly with beta here (beta = 0.6 is only 4 SE away with 5000 runs), so
    # the wrong value is the neutral process, beta = 0 (15.5 SE away).
    wrong = _absorbing_row(fit, start, 0.0)[1]
    assert_power_margin(mean_time, wrong, se, "t(3, 3, 3)")
    result = evolver.estimate_mean_absorption_time(BETA_MC, start, runs)
    assert_within_k_se(result["mean"], mean_time, se, label="t(3, 3, 3)")
    assert_detects(result["mean"], wrong, se)
    # The reported standard error estimates the exact one.
    assert result["stderr"] == pytest.approx(se, rel=0.15)


def _absorbing_row(fit, start, beta):
    transient, b_matrix, t, variance = absorbing_chain(fit, Z, 3, beta)
    row = next(i for i, n in enumerate(transient) if np.array_equal(n, start))
    return b_matrix[row], t[row], variance[row]

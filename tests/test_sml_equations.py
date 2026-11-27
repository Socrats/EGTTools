"""Check the small-mutation-limit implementations against ``docs/theory/small_mutation_limit.md``.

The references are the transcriptions in ``theory_reference.py``: the fixation matrix from F 2.2,
the SML transition matrix of S 2.2, and its stationary distribution solved in 60-digit decimal
arithmetic from log rho, so that no selection strength can underflow it.
"""
import itertools

import numpy as np
import pytest

import egttools as egt
from chains import reference_stationary
from egttools.analytical import PairwiseComparison, StochDynamics
from theory_reference import (fitness_n_player, fitness_two_player, full_transition_matrix, sml_fixation_matrix,
                              sml_log_fixation_matrix, sml_transition_matrix, stationary_distribution_decimal)

Z = 9
GROUP_SIZE = 3
BETAS = [0.0, 0.5, 1.3, 5.0]
STRONG_BETAS = [50.0, 500.0, 5000.0]


def two_player_game(k, seed=3):
    a = np.random.default_rng(seed).uniform(-1, 2, size=(k, k))
    return a, egt.games.Matrix2PlayerGameHolder(k, a), (lambda n: fitness_two_player(a, n))


def n_player_game(k, seed=3):
    table = np.random.default_rng(seed).uniform(-1, 2, size=(k, egt.calculate_nb_states(GROUP_SIZE, k)))
    return table, egt.games.MatrixNPlayerGameHolder(k, GROUP_SIZE, table), (
        lambda n: fitness_n_player(table, GROUP_SIZE, n))


def coordination_game():
    """Every strategy resists invasion, so every rho underflows under strong selection (S 2.3)."""
    a = np.diag([1.0, 1.001, 1.003])
    return a, egt.games.Matrix2PlayerGameHolder(3, a), (lambda n: fitness_two_player(a, n))


CASES = [("2p", 3), ("2p", 4), ("3p", 3)]


def make(name, k):
    return two_player_game(k) if name == "2p" else n_player_game(k)


def off_diagonal(k):
    return ~np.eye(k, dtype=bool)


# ---------------------------------------------------------------------------
# S 2.1 - S 2.2: fixation and transition matrices
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name,k", CASES)
@pytest.mark.parametrize("beta", BETAS)
def test_pairwise_comparison_sml_matches_definition_2_2(name, k, beta):
    _, game, fit = make(name, k)
    transitions, fixation = PairwiseComparison(Z, game).calculate_transition_and_fixation_matrix_sml(beta)
    rho = sml_fixation_matrix(fit, Z, k, beta)
    mask = off_diagonal(k)
    np.testing.assert_allclose(fixation[mask], rho[mask], rtol=1e-12)
    np.testing.assert_allclose(transitions, sml_transition_matrix(rho), rtol=1e-12, atol=1e-15)


@pytest.mark.parametrize("k", [3, 4])
@pytest.mark.parametrize("beta", BETAS)
def test_stoch_dynamics_sml_is_the_transposed_matrix(k, beta):
    a, _, fit = two_player_game(k)
    transitions, fixation = StochDynamics(k, a, Z).transition_and_fixation_matrix(beta)
    rho = sml_fixation_matrix(fit, Z, k, beta)
    mask = off_diagonal(k)
    np.testing.assert_allclose(fixation[mask], rho[mask], rtol=1e-12)
    np.testing.assert_allclose(transitions, sml_transition_matrix(rho).T, rtol=1e-12, atol=1e-15)


# ---------------------------------------------------------------------------
# S 2.3: log space
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("beta", BETAS + STRONG_BETAS)
def test_log_sml_matrix_matches_definition_2_3(beta):
    a, game, fit = two_player_game(3)
    log_rho = sml_log_fixation_matrix(fit, Z, 3, beta)
    mask = off_diagonal(3)
    c_log = -np.max(log_rho[mask])
    expected = np.eye(3)
    expected[mask] = np.exp(log_rho[mask] + c_log) / 2
    np.fill_diagonal(expected, 0.0)
    np.fill_diagonal(expected, 1.0 - expected.sum(axis=1))
    for transitions, log_fixation in (PairwiseComparison(Z, game).calculate_transition_and_log_fixation_matrix_sml(beta),
                                      StochDynamics(3, a, Z).transition_and_log_fixation_matrix(beta)):
        if np.allclose(transitions.sum(axis=0), 1.0) and not np.allclose(transitions.sum(axis=1), 1.0):
            transitions = transitions.T  # StochDynamics returns the transpose (API-06)
        np.testing.assert_allclose(log_fixation[mask], log_rho[mask], rtol=1e-12, atol=1e-14)
        np.testing.assert_allclose(transitions, expected, rtol=1e-12, atol=1e-15)


@pytest.mark.parametrize("beta", [1.3] + STRONG_BETAS)
def test_log_sml_matrix_keeps_the_stationary_distribution_when_rho_underflows(beta):
    """S 2.3 and S 3.1: in a coordination game every rho underflows under strong selection; the
    rescaled matrix still gives the stationary distribution of T."""
    a, game, fit = coordination_game()
    expected = stationary_distribution_decimal(sml_log_fixation_matrix(fit, Z, 3, beta))
    transitions, _ = PairwiseComparison(Z, game).calculate_transition_and_log_fixation_matrix_sml(beta)
    transposed, _ = StochDynamics(3, a, Z).transition_and_log_fixation_matrix(beta)
    np.testing.assert_allclose(reference_stationary(transitions), expected, atol=1e-13)
    np.testing.assert_allclose(reference_stationary(transposed.T), expected, atol=1e-13)


def test_plain_sml_matrix_loses_the_stationary_distribution_when_rho_underflows():
    """Why S 2.3 exists: at beta = 50 every rho is about e^-100, below the rounding unit of the
    diagonal, so T_ii rounds to exactly 1 and the stationary distribution solved from T - I is wrong."""
    _, game, fit = coordination_game()
    transitions, _ = PairwiseComparison(Z, game).calculate_transition_and_fixation_matrix_sml(50.0)
    np.testing.assert_array_equal(np.diag(transitions), 1.0)
    expected = stationary_distribution_decimal(sml_log_fixation_matrix(fit, Z, 3, 50.0))
    assert np.max(np.abs(reference_stationary(transitions) - expected)) > 0.1


# ---------------------------------------------------------------------------
# S 3.1: stationary distribution
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name,k", CASES)
@pytest.mark.parametrize("beta", BETAS)
def test_stationary_distribution_matches_definition_3_1(name, k, beta):
    _, game, fit = make(name, k)
    expected = stationary_distribution_decimal(sml_log_fixation_matrix(fit, Z, k, beta))
    transitions, _ = PairwiseComparison(Z, game).calculate_transition_and_fixation_matrix_sml(beta)
    np.testing.assert_allclose(egt.utils.calculate_stationary_distribution(transitions.T), expected, atol=1e-12)
    np.testing.assert_allclose(egt.utils.calculate_stationary_distribution_non_hermitian(transitions.T), expected,
                               atol=1e-12)


@pytest.mark.parametrize("k", [3, 4])
@pytest.mark.parametrize("beta", BETAS)
def test_stoch_dynamics_stationary_distribution_without_mutation_is_the_sml(k, beta):
    a, _, fit = two_player_game(k)
    expected = stationary_distribution_decimal(sml_log_fixation_matrix(fit, Z, k, beta))
    np.testing.assert_allclose(StochDynamics(k, a, Z).calculate_stationary_distribution(beta), expected, atol=1e-12)


def test_stationary_distribution_utils_under_strong_selection():
    """The eigendecomposition is less accurate for the near-identity matrices of strong selection
    (about 5e-10 measured at beta = 5000); the rescaled matrix keeps it usable."""
    _, game, fit = coordination_game()
    expected = stationary_distribution_decimal(sml_log_fixation_matrix(fit, Z, 3, 5000.0))
    transitions, _ = PairwiseComparison(Z, game).calculate_transition_and_log_fixation_matrix_sml(5000.0)
    np.testing.assert_allclose(egt.utils.calculate_stationary_distribution(transitions.T), expected, atol=1e-8)


# ---------------------------------------------------------------------------
# Theorem 1.1: the full chain converges to the SML as mu -> 0
# ---------------------------------------------------------------------------

def test_full_chain_converges_to_the_sml_in_proportion_to_mu():
    z, k, beta = 6, 3, 1.3
    _, _, fit = two_player_game(k)
    sml = stationary_distribution_decimal(sml_log_fixation_matrix(fit, z, k, beta))
    errors = []
    for mu in (1e-4, 1e-5, 1e-6):
        states, p = full_transition_matrix(fit, z, k, beta, mu)
        pi = reference_stationary(p)
        homogeneous = [next(i for i, n in enumerate(states) if n[l] == z) for l in range(k)]
        errors.append(np.max(np.abs(pi[homogeneous] - sml)))
    assert errors[-1] < 1e-4
    # Order mu: each tenfold decrease of mu divides the error by about 10.
    for larger, smaller in itertools.pairwise(errors):
        assert 8 < larger / smaller < 12

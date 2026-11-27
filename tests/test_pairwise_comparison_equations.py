"""Check every pairwise-comparison implementation against its specification.

The reference functions in ``theory_reference.py`` transcribe the numbered definitions in
``docs/theory/pairwise_comparison.md`` directly, with no shortcuts, and each
implementation is compared against the definition it claims to compute.
"""
import itertools

import numpy as np
import pytest

import egttools as egt
from egttools.analytical import PairwiseComparison, StochDynamics
from egttools.numerical import PairwiseComparisonNumerical
from theory_reference import all_states, fitness_n_player, fitness_two_player, gradient, transition


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

def two_player_case(k, seed):
    a = np.random.default_rng(seed).uniform(-1, 2, size=(k, k))
    return egt.games.Matrix2PlayerGameHolder(k, a), (lambda n: fitness_two_player(a, n)), a


def n_player_case(k, group_size, seed):
    table = np.random.default_rng(seed).uniform(-1, 2, size=(k, egt.calculate_nb_states(group_size, k)))
    game = egt.games.MatrixNPlayerGameHolder(k, group_size, table)
    return game, (lambda n: fitness_n_player(table, group_size, n))


GAMES = [("2p-k2", 2), ("2p-k3", 3), ("2p-k4", 4), ("3p-k3", 3)]
Z = 9
BETA = 1.3


def make_game(name, k):
    if name.startswith("2p"):
        game, fit, _ = two_player_case(k, seed=k)
    else:
        game, fit = n_player_case(k, group_size=3, seed=k)
    return game, fit


# ---------------------------------------------------------------------------
# Definition 2: fitness
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name,k", GAMES)
def test_fitness_matches_definition_2(name, k):
    game, fit = make_game(name, k)
    for n in all_states(Z, k):
        expected = fit(n)
        for l in range(k):
            if n[l] == 0:
                continue
            others = n - np.eye(k, dtype=np.uint64)[l]
            assert game.calculate_fitness(l, Z, others) == pytest.approx(expected[l], abs=1e-12)


# ---------------------------------------------------------------------------
# Definition 4.1: transition matrix
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mu", [0.0, 0.05])
@pytest.mark.parametrize("name,k", GAMES)
def test_transition_matrix_matches_definition_4_1(name, k, mu):
    game, fit = make_game(name, k)
    t = PairwiseComparison(Z, game).calculate_transition_matrix(BETA, mu)
    t = t.toarray() if hasattr(t, "toarray") else np.asarray(t)
    states = all_states(Z, k)
    for s, n in enumerate(states):
        f = fit(n)
        expected = np.zeros(len(states))
        for j, l in itertools.permutations(range(k), 2):
            p = transition(n, j, l, f, BETA, mu)
            if p > 0:
                expected[egt.calculate_state(Z, n - np.eye(k, dtype=int)[j] + np.eye(k, dtype=int)[l])] += p
        expected[s] = 1 - expected.sum()
        np.testing.assert_allclose(t[s], expected, atol=1e-12)


# ---------------------------------------------------------------------------
# Definition 5.1: gradient of selection
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name,k", GAMES)
def test_gradient_without_mutation_matches_definition_5_1(name, k):
    game, fit = make_game(name, k)
    evolver = PairwiseComparison(Z, game)
    for n in all_states(Z, k):
        np.testing.assert_allclose(evolver.calculate_gradient_of_selection(BETA, n),
                                   gradient(n, fit(n), BETA, 0.0), atol=1e-12)


@pytest.mark.parametrize("mu", [0.0, 0.05, 0.3])
@pytest.mark.parametrize("name,k", GAMES)
def test_gradient_with_mutation_matches_definition_5_1(name, k, mu):
    game, fit = make_game(name, k)
    evolver = PairwiseComparison(Z, game)
    for n in all_states(Z, k):
        np.testing.assert_allclose(evolver.calculate_gradient_of_selection_with_mutation(BETA, mu, n),
                                   gradient(n, fit(n), BETA, mu), atol=1e-12)


@pytest.mark.parametrize("mu", [0.0, 0.05])
@pytest.mark.parametrize("k", [2, 3, 4])
def test_stoch_dynamics_gradient_matches_definition_5_1(k, mu):
    _, fit, a = two_player_case(k, seed=k)
    dynamics = StochDynamics(k, a, Z, mu=mu)
    for n in all_states(Z, k):
        expected = gradient(n, fit(n), BETA, mu)
        np.testing.assert_allclose(dynamics.full_gradient_selection(n, BETA), expected, atol=1e-12)
        if mu == 0.0:
            np.testing.assert_allclose(dynamics.full_gradient_selection_without_mutation(n, BETA),
                                       expected, atol=1e-12)


@pytest.mark.parametrize("mu", [0.0, 0.05])
@pytest.mark.parametrize("name,k", GAMES)
def test_gradient_components_sum_to_zero(name, k, mu):
    game, _ = make_game(name, k)
    evolver = PairwiseComparison(Z, game)
    for n in all_states(Z, k):
        assert evolver.calculate_gradient_of_selection_with_mutation(BETA, mu, n).sum() == pytest.approx(0, abs=1e-12)


# ---------------------------------------------------------------------------
# Definition 3.2: the simulation implements the process of Definition 4.1
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mu", [0.01, 0.1])
def test_simulation_one_step_drift_matches_definition_5_1(mu):
    k = 3
    game, fit, _ = two_player_case(k, seed=11)
    evolver = PairwiseComparisonNumerical(Z, game, 1_000_000, seed=2024)
    trajectory = evolver.run_with_mutation(400_000, BETA, mu, np.array([3, 3, 3], dtype=np.uint64)).astype(np.int64)
    before, step = trajectory[:-1], np.diff(trajectory, axis=0)
    index = np.array([egt.calculate_state(Z, s) for s in before])
    checked = 0
    for s in np.unique(index):
        mask = index == s
        if mask.sum() < 3000:
            continue
        n = before[mask][0]
        expected = gradient(n, fit(n), BETA, mu)
        observed = step[mask].mean(axis=0)
        se = step[mask].std(axis=0, ddof=1) / np.sqrt(mask.sum())
        z_scores = np.abs(observed - expected) / np.maximum(se, 1e-12)
        assert np.all(z_scores < 5), (n, observed, expected, se)
        checked += 1
    assert checked >= 5


# ---------------------------------------------------------------------------
# Monte Carlo estimators converge to the stationary distribution of Definition 4.1
# ---------------------------------------------------------------------------

MC_MU = 0.1   # large enough that time spent in monomorphic states matters


@pytest.fixture(scope="module")
def mc_case():
    k = 3
    game, _, _ = two_player_case(k, seed=11)
    t = PairwiseComparison(Z, game).calculate_transition_matrix(BETA, MC_MU)
    t = t.toarray() if hasattr(t, "toarray") else np.asarray(t)
    pi = egt.utils.calculate_stationary_distribution(t.T)
    states = all_states(Z, k)
    monomorphic = (states == Z).any(axis=1).astype(float)
    return game, pi, states, monomorphic


def assert_within_se(per_run, expected, n_se=5):
    per_run = np.atleast_2d(per_run)
    mean = per_run.mean(axis=0)
    se = per_run.std(axis=0, ddof=1) / np.sqrt(per_run.shape[0])
    assert np.all(np.abs(mean - expected) < n_se * np.maximum(se, 1e-12)), (mean, expected, se)


def test_estimate_stationary_distribution_matches_exact(mc_case):
    game, pi, _, monomorphic = mc_case
    runs = [PairwiseComparisonNumerical(Z, game, 100_000, seed=s)
            .estimate_stationary_distribution(1, 60_000, 1_000, BETA, MC_MU) for s in range(30)]
    assert_within_se(np.array([[r @ monomorphic, r.max()] for r in runs]),
                     np.array([pi @ monomorphic, pi[np.argmax(np.mean(runs, axis=0))]]))


def test_estimate_stationary_distribution_sparse_matches_exact(mc_case):
    game, pi, _, monomorphic = mc_case
    runs = []
    for s in range(30):
        r = PairwiseComparisonNumerical(Z, game, 100_000, seed=s).estimate_stationary_distribution_sparse(
            1, 60_000, 1_000, BETA, MC_MU)
        runs.append(np.asarray(r.todense()).ravel())
    assert_within_se(np.array([[r @ monomorphic] for r in runs]), np.array([pi @ monomorphic]))


def test_estimate_strategy_distribution_matches_exact(mc_case):
    game, pi, states, _ = mc_case
    runs = np.array([PairwiseComparisonNumerical(Z, game, 100_000, seed=s)
                     .estimate_strategy_distribution(1, 60_000, 1_000, BETA, MC_MU) for s in range(30)])
    assert_within_se(runs, pi @ states / Z)


@pytest.mark.parametrize("precompute_limit", [20_000_000, 0], ids=["precomputed", "direct"])
def test_estimate_stationary_indicators_matches_exact(mc_case, precompute_limit):
    game, pi, states, monomorphic = mc_case
    indicators = [lambda n: float(max(n) == Z), lambda n: n[0] / Z]
    result = PairwiseComparisonNumerical(Z, game, 100_000, seed=3).estimate_stationary_indicators(
        indicators, 30, 60_000, 1_000, BETA, MC_MU, precompute_limit=precompute_limit, verbose=True)
    assert_within_se(result.per_run_values, pi @ np.column_stack([monomorphic, states[:, 0] / Z]))

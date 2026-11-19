"""Number of population states, C(Z + S - 1, S - 1), for every class that computes it.

The count must be exact whenever it fits in a signed 64-bit integer. When it does not fit,
constructing the class still works (methods such as fixation probabilities never need the
count), but ``nb_states`` and every method that needs the full state space raise
``OverflowError``. The reference is Python's exact ``math.comb``.
"""
from math import comb

import numpy as np
import pytest

egt = pytest.importorskip("egttools")

from egttools.analytical import PairwiseComparison  # noqa: E402
from egttools.numerical import (  # noqa: E402
    PairwiseComparisonNumerical,
    PairwiseComparisonTransitionOperator,
)

INT64_MAX = 2 ** 63 - 1

# (Z, S) pairs whose count fits in 32 bits but was computed wrongly (negative) when the
# arithmetic ran in int, plus small and boundary-free cases.
FITTING = [(10, 2), (100, 3), (50, 5), (135, 6), (136, 6), (52, 8), (31, 10), (1000, 4)]

# The count does not fit in int64: C(128, 28) ~ 1e28.
TOO_LARGE = (100, 29)


def matrix_game(nb_strategies, seed=0):
    payoffs = np.random.default_rng(seed).random((nb_strategies, nb_strategies))
    return egt.games.Matrix2PlayerGameHolder(nb_strategies, payoffs)


def expected_nb_states(population_size, nb_strategies):
    return comb(population_size + nb_strategies - 1, nb_strategies - 1)


@pytest.mark.parametrize("population_size, nb_strategies", FITTING)
def test_pairwise_comparison_nb_states_is_exact(population_size, nb_strategies):
    pc = PairwiseComparison(population_size, matrix_game(nb_strategies))
    assert pc.nb_states() == expected_nb_states(population_size, nb_strategies)


@pytest.mark.parametrize("population_size, nb_strategies", FITTING)
def test_pairwise_comparison_numerical_nb_states_is_exact(population_size, nb_strategies):
    pcn = PairwiseComparisonNumerical(population_size, matrix_game(nb_strategies), 100)
    assert pcn.nb_states == expected_nb_states(population_size, nb_strategies)


def test_pairwise_comparison_update_population_size_recomputes_nb_states():
    pc = PairwiseComparison(10, matrix_game(6))
    pc.update_population_size(136)
    assert pc.nb_states() == expected_nb_states(136, 6)


def test_calculate_nb_states_is_exact_for_any_size():
    # The public helper returns arbitrary-precision integers, so it never overflows.
    for population_size, nb_strategies in FITTING + [TOO_LARGE, (1000, 50)]:
        assert egt.calculate_nb_states(population_size, nb_strategies) == \
               expected_nb_states(population_size, nb_strategies)


def test_too_large_count_does_not_fit_in_int64():
    assert expected_nb_states(*TOO_LARGE) > INT64_MAX


class TestTooManyStates:
    """Z = 100 and S = 29: valid parameters whose state count exceeds int64."""

    def setup_method(self):
        self.population_size, self.nb_strategies = TOO_LARGE
        self.game = matrix_game(self.nb_strategies)

    def test_pairwise_comparison_constructs_and_computes_fixation(self):
        pc = PairwiseComparison(self.population_size, self.game)
        rho = pc.calculate_fixation_probability(1, 0, 1.0)
        assert 0.0 <= rho <= 1.0

    def test_pairwise_comparison_nb_states_raises(self):
        pc = PairwiseComparison(self.population_size, self.game)
        with pytest.raises(OverflowError):
            pc.nb_states()

    def test_pairwise_comparison_transition_matrix_raises(self):
        pc = PairwiseComparison(self.population_size, self.game)
        with pytest.raises(OverflowError):
            pc.calculate_transition_matrix(1.0, 0.01)

    def test_pairwise_comparison_numerical_constructs_and_estimates_fixation(self):
        pcn = PairwiseComparisonNumerical(self.population_size, self.game, 100)
        rho = pcn.estimate_fixation_probability(1, 0, 5, 100, 1.0)
        assert 0.0 <= rho <= 1.0

    def test_pairwise_comparison_numerical_nb_states_raises(self):
        pcn = PairwiseComparisonNumerical(self.population_size, self.game, 100)
        with pytest.raises(OverflowError):
            _ = pcn.nb_states

    @pytest.mark.filterwarnings("ignore:mu is very small:UserWarning")
    def test_pairwise_comparison_numerical_stationary_distribution_raises(self):
        pcn = PairwiseComparisonNumerical(self.population_size, self.game, 100)
        with pytest.raises(OverflowError):
            pcn.estimate_stationary_distribution(1, 10, 0, 1.0, 1e-2)

    def test_transition_operator_raises(self):
        with pytest.raises(OverflowError):
            PairwiseComparisonTransitionOperator(self.population_size, self.game, 1.0, 0.01)


def reference_fixation_probability(game, population_size, invader, resident, beta):
    """Cache-free fixation probability of `invader` in a population of `resident`.

    rho = 1 / (1 + sum_{k=1}^{Z-1} prod_{i=1}^{k} exp(beta (f_res(i) - f_inv(i)))), with fitness
    taken directly from the game at each edge state (i invaders, Z - i residents), the focal
    individual excluded. Computed in log space.
    """
    nb_strategies = game.nb_strategies()
    log_prod, log_terms = 0.0, []
    for i in range(1, population_size):
        state = np.zeros(nb_strategies, dtype=np.uint64)
        state[invader], state[resident] = i, population_size - i
        others_of_invader, others_of_resident = state.copy(), state.copy()
        others_of_invader[invader] -= 1
        others_of_resident[resident] -= 1
        f_inv = game.calculate_fitness(invader, population_size, others_of_invader)
        f_res = game.calculate_fitness(resident, population_size, others_of_resident)
        log_prod += beta * (f_res - f_inv)
        log_terms.append(log_prod)
    top = max(log_terms)
    log_phi = top + np.log(np.sum(np.exp(np.array(log_terms) - top)))
    return np.exp(-np.logaddexp(0.0, log_phi))  # 1 / (1 + phi) without overflow


@pytest.mark.parametrize("beta", [0.1, 1.0, 5.0])
@pytest.mark.parametrize("pre_calculate_edges", [False, True])
def test_fixation_with_many_strategies_matches_reference(beta, pre_calculate_edges):
    # 29 strategies: the global index of an edge state does not fit in 64 bits, so the fitness
    # cache must not depend on it.
    population_size, nb_strategies = TOO_LARGE
    payoffs = 10 * np.random.default_rng(3).random((nb_strategies, nb_strategies))
    game = egt.games.Matrix2PlayerGameHolder(nb_strategies, payoffs)
    pc = PairwiseComparison(population_size, game)
    if pre_calculate_edges:
        pc.pre_calculate_edge_fitnesses()
    for invader, resident in [(1, 0), (0, 1), (5, 17), (28, 3), (27, 28)]:
        expected = reference_fixation_probability(game, population_size, invader, resident, beta)
        assert pc.calculate_fixation_probability(invader, resident, beta) == pytest.approx(expected, rel=1e-10)

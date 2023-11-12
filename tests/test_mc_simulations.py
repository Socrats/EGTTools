import pytest

import os
from sys import platform

import numpy as np

egt = pytest.importorskip("egttools")

Random = egt.Random
PairwiseComparisonNumerical = egt.numerical.PairwiseComparisonNumerical
NormalFormGame = egt.games.NormalFormGame


@pytest.fixture
def setup_hawk_dove_parameters() -> np.ndarray:
    # Payoff matrix
    v, d, t = 2, 3, 1
    payoffs = np.array([
        [(v - d) / 2, v],
        [0, (v / 2) - t],
    ])

    # Necessary to avoid issues with Anaconda on MacOSX
    if platform == "darwin":
        os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

    Random.init(3610063510)

    assert Random._seed == 3610063510

    return payoffs


def test_normal_form_game_runs(setup_hawk_dove_parameters) -> None:
    """
    This test checks that a Normal form game runs as expected
    """

    payoffs = setup_hawk_dove_parameters

    game = NormalFormGame(1, payoffs)

    assert game.nb_strategies() == 2
    assert game.nb_rounds == 1
    assert game.type() == "NormalFormGame"
    np.testing.assert_array_equal(game.payoffs(), payoffs)
    np.testing.assert_array_equal(game.expected_payoffs(), payoffs)


def test_pairwise_moran_run(setup_hawk_dove_parameters) -> None:
    """
    This test checks that the run method of PairwiseComparisonNumerical executes.
    """
    payoffs = setup_hawk_dove_parameters

    game = NormalFormGame(1, payoffs)

    pop_size = 100
    cache_size = 10000
    nb_generations = int(1e6)
    beta = 1.0
    mu = 1e-3
    initial_state = [50, 50]

    evolver = PairwiseComparisonNumerical(pop_size, game, cache_size)
    result = evolver.run(nb_generations, beta=beta, mu=mu, init_state=initial_state)

    assert result.shape == (nb_generations + 1, game.nb_strategies())


def test_pairwise_moran_stationary_distribution(setup_hawk_dove_parameters) -> None:
    """
    This test checks that the stationary_distribution method of PairwiseComparisonNumerical executes.
    """
    payoffs = setup_hawk_dove_parameters

    game = NormalFormGame(1, payoffs)

    pop_size = 100
    cache_size = 10000
    nb_generations = int(1e3)
    transitory = int(1e3)
    beta = 10
    mu = 1e-3
    runs = 10

    nb_states = pop_size + 1

    evolver = PairwiseComparisonNumerical(pop_size, game, cache_size)
    dist = evolver.estimate_stationary_distribution(runs, nb_generations, transitory, beta, mu)
    assert dist.shape == (nb_states,)


def test_pairwise_moran_stationary_distribution_sparse(setup_hawk_dove_parameters) -> None:
    """
    This test checks that the stationary_distribution method of PairwiseComparisonNumerical executes.
    """
    payoffs = setup_hawk_dove_parameters

    game = NormalFormGame(1, payoffs)

    pop_size = 100
    cache_size = 10000
    nb_generations = int(1e3)
    transitory = int(1e3)
    beta = 10
    mu = 1e-3
    runs = 10

    nb_states = pop_size + 1

    evolver = PairwiseComparisonNumerical(pop_size, game, cache_size)
    dist = evolver.estimate_stationary_distribution_sparse(runs, nb_generations, transitory, beta, mu)
    assert dist.shape == (1, nb_states)

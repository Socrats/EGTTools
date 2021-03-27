import numpy as np
import pytest

from egttools.numerical.games import NormalFormGame
from egttools.numerical import PairwiseMoran
from egttools.numerical import Random

@pytest.fixture
def setup_hawk_dove_parameters() -> np.ndarray:
    # Payoff matrix
    v, d, t = 2, 3, 1
    payoffs = np.array([
        [(v - d) / 2, v],
        [0, (v / 2) - t],
    ])

    return payoffs


def test_normal_form_game_runs(setup_hawk_dove_parameters) -> None:
    """
    This test checks that a Normal form game runs as expected
    """

    payoffs = setup_hawk_dove_parameters

    game = NormalFormGame(1, payoffs)

    assert game.nb_strategies == 2
    assert game.nb_rounds == 1
    assert game.type() == "NormalFormGame"
    assert (game.payoffs() == payoffs).all()
    assert (game.expected_payoffs() == payoffs).all()


def test_pairwise_moran_run(setup_hawk_dove_parameters) -> None:
    """
    This test checks that the run method of PairwiseMoran executes.
    """
    payoffs = setup_hawk_dove_parameters

    Random.init()
    Random.seed(3610063510)

    assert Random.seed_ == 3610063510

    game = NormalFormGame(1, payoffs)

    pop_size = 100
    cache_size = 1000000
    nb_generations = int(1e6)
    beta = 1
    mu = 1e-3
    initial_state = [50, 50]

    evolver = PairwiseMoran(pop_size, game, cache_size)
    result = evolver.run(nb_generations, beta, mu, initial_state)

    assert result.shape == (nb_generations + 1, game.nb_strategies)

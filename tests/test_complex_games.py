import numpy as np

from scipy.sparse import lil_matrix

from egttools.utils import calculate_stationary_distribution
from egttools.analytical.complex_games import PairwiseComparison2D
from egttools.games import NormalFormGame
from pytest import fixture


@fixture
def load_game_transitions1():
    game_transitions = lil_matrix((2, 2), dtype=np.float64)
    game_transitions[0, 0] = 0.7
    game_transitions[0, 1] = 0.3
    game_transitions[1, 0] = 0.2
    game_transitions[1, 1] = 0.8

    return game_transitions.tocsr()


@fixture
def load_game_transitions2():
    game_transitions = lil_matrix((2, 2), dtype=np.float64)
    game_transitions[0, 0] = 1
    game_transitions[0, 1] = 0
    game_transitions[1, 0] = 1
    game_transitions[1, 1] = 0

    return game_transitions.tocsr()


@fixture
def load_games():
    payoff_matrix1 = np.array([
        [1, 3],
        [0, 2]
    ])
    payoff_matrix2 = np.array([
        [3, 0],
        [0, 3]
    ])

    game1 = NormalFormGame(1, payoff_matrix1)
    game2 = NormalFormGame(1, payoff_matrix2)
    games = [game1, game2]

    return games


@fixture
def load_game_and_evolver1(load_games, load_game_transitions1):
    games = load_games

    game_transitions = load_game_transitions1

    evolver = PairwiseComparison2D(population_size=10, nb_strategies=2, games=games, game_transitions=game_transitions,
                                   lda=0.5)

    return games, evolver


@fixture
def load_game_and_evolver2(load_games, load_game_transitions2):
    games = load_games

    game_transitions = load_game_transitions2

    evolver = PairwiseComparison2D(population_size=100, nb_strategies=2, games=games, game_transitions=game_transitions,
                                   lda=0.5)

    return games, evolver


def test_probabilities_sum_to_one(load_game_and_evolver1):
    games, evolver = load_game_and_evolver1

    transition_matrix = evolver.calculate_transition_matrix(beta=1, mu=1e-3)

    assert np.allclose(transition_matrix.sum(axis=1), 1.)


def test_pd_case(load_game_and_evolver2):
    games, evolver = load_game_and_evolver2

    transition_matrix = evolver.calculate_transition_matrix(beta=10, mu=0)

    sd = calculate_stationary_distribution(transition_matrix.transpose())

    assert np.isclose(sd.sum(), 1.)

    state_index = evolver.calculate_state_index(0, 0)

    assert np.isclose(sd[state_index], 1.)


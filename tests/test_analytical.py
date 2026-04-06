import numpy as np
import pytest
import egttools as egt


class DummyPairwiseReplicatorGame(egt.games.AbstractReplicatorGame):
    def __init__(self, payoff_matrix: np.ndarray):
        super().__init__()
        self._payoff_matrix = np.asarray(payoff_matrix, dtype=np.float64)

    def calculate_payoffs(self):
        return self._payoff_matrix

    def calculate_fitness(self, frequencies):
        frequencies = np.asarray(frequencies, dtype=np.float64)
        return self._payoff_matrix @ frequencies

    def nb_strategies(self):
        return self._payoff_matrix.shape[0]

    def group_size(self):
        return 2

    def __str__(self):
        return "DummyPairwiseReplicatorGame"

    def type(self):
        return "DummyPairwiseReplicatorGame"

    def payoffs(self):
        return self._payoff_matrix


class DummyNPlayerReplicatorGame(egt.games.AbstractReplicatorGame):
    """
    Simple 2-strategy, 3-player replicator game whose fitness matches the
    matrix-based `replicator_equation_n_player` convention used in EGTtools.

    The payoff matrix is assumed to have shape (2, 4), with columns corresponding
    to the full group compositions:
        [3,0], [2,1], [1,2], [0,3]
    """

    def __init__(self, payoff_matrix: np.ndarray):
        super().__init__()
        payoff_matrix = np.asarray(payoff_matrix, dtype=np.float64)
        if payoff_matrix.shape != (2, 4):
            raise ValueError("DummyNPlayerReplicatorGame requires a payoff matrix of shape (2, 4).")
        self._payoff_matrix = payoff_matrix

    def calculate_payoffs(self):
        return self._payoff_matrix

    def calculate_fitness(self, frequencies):
        frequencies = np.asarray(frequencies, dtype=np.float64)
        x0, x1 = frequencies

        # Probabilities of co-player compositions for 2 co-players:
        p0 = x0 ** 2
        p1 = 2.0 * x0 * x1
        p2 = x1 ** 2

        fitness = np.empty(2, dtype=np.float64)

        # Focal strategy 0:
        # full groups [3,0], [2,1], [1,2]
        fitness[0] = (
                p0 * self._payoff_matrix[0, 0]
                + p1 * self._payoff_matrix[0, 1]
                + p2 * self._payoff_matrix[0, 2]
        )

        # Focal strategy 1:
        # full groups [2,1], [1,2], [0,3]
        fitness[1] = (
                p0 * self._payoff_matrix[1, 1]
                + p1 * self._payoff_matrix[1, 2]
                + p2 * self._payoff_matrix[1, 3]
        )

        return fitness

    def nb_strategies(self):
        return 2

    def group_size(self):
        return 3

    def __str__(self):
        return "DummyNPlayerReplicatorGame"

    def type(self):
        return "DummyNPlayerReplicatorGame"

    def payoffs(self):
        return self._payoff_matrix


def test_analytical_fitness_calculation_stoch_dynamics():
    group_size = 6
    min_nb_cooperators = 3
    c = 0.1
    b = 1
    population_size = 50
    risk = 1.
    game = egt.games.OneShotCRD(b, c, risk, group_size, min_nb_cooperators)
    payoffs = egt.utils.transform_payoffs_to_pairwise(game.nb_strategies(), game)
    evolver = egt.analytical.StochDynamics(game.nb_strategies(), payoffs, population_size, group_size)
    result = evolver.fitness_group(population_size - 1, 0, 1)

    assert result > 0


def test_analytical_pairwise_comparison_generic_game():
    group_size = 6
    min_nb_cooperators = 3
    c = 0.1
    b = 1
    population_size = 50
    risk = 1.
    game = egt.games.OneShotCRD(b, c, risk, group_size, min_nb_cooperators)
    evolver = egt.analytical.PairwiseComparison(population_size, game)
    result = evolver.calculate_fixation_probability(0, 1, 1)

    assert np.isclose(0.003682100470889826, result)


def test_analytical_pairwise_comparison_2_player_game():
    # Payoff matrix
    v = 2
    d = 3
    t = 1
    payoff_matrix = np.array([
        [(v - d) / 2, v],
        [0, (v / 2) - t],
    ])
    game = egt.games.Matrix2PlayerGameHolder(2, payoff_matrix=payoff_matrix)
    evolver = egt.analytical.PairwiseComparison(100, game)
    result = evolver.calculate_fixation_probability(0, 1, 1)

    assert np.isclose(0.8641155742462664, result)


def test_analytical_pairwise_comparison_n_player_game():
    # Payoff matrix
    v = 2
    d = 3
    t = 1
    payoff_matrix = np.array([
        [(v - d) / 2, v, v / 2, d],
        [0, (v / 2) - t, t, v],
    ])
    game = egt.games.MatrixNPlayerGameHolder(nb_strategies=2, group_size=3, payoff_matrix=payoff_matrix)
    evolver = egt.analytical.PairwiseComparison(100, game)
    result = evolver.calculate_fixation_probability(1, 0, 1)

    assert np.isclose(0.020104159364630978, result)


def test_if_stoch_dynamics_matches_with_pairwise_comparison():
    # Payoff matrix
    v = 2
    d = 3
    t = 1
    payoff_matrix = np.array([
        [(v - d) / 2, v],
        [0, (v / 2) - t],
    ])
    game = egt.games.Matrix2PlayerGameHolder(2, payoff_matrix=payoff_matrix)
    evolver1 = egt.analytical.PairwiseComparison(100, game)
    evolver2 = egt.analytical.StochDynamics(nb_strategies=2, payoffs=payoff_matrix, pop_size=100, group_size=2, mu=0)

    assert np.isclose(evolver1.calculate_fixation_probability(0, 1, 1), evolver2.fixation_probability(0, 1, 1))
    assert np.isclose(evolver1.calculate_gradient_of_selection(1, np.array([30
                                                                               , 70]))[0],
                      evolver2.gradient_selection(30, 0, 1, 1))
    transition1, fixation1 = evolver1.calculate_transition_and_fixation_matrix_sml(1)
    transition2, fixation2 = evolver2.transition_and_fixation_matrix(1)

    assert np.allclose(transition1.transpose(), transition2)
    assert np.allclose(fixation1, fixation2)

    full_transitions1 = evolver1.calculate_transition_matrix(1, 0.01)
    evolver2.mu = 0.01
    full_transitions2 = evolver2.calculate_full_transition_matrix(1)

    assert np.allclose(full_transitions1.toarray().transpose(), full_transitions2.toarray())

    # Now the same but for N-player games
    payoff_matrix = np.array([
        [(v - d) / 2, v, v / 2, d],
        [0, (v / 2) - t, t, v],
    ])
    game = egt.games.MatrixNPlayerGameHolder(nb_strategies=2, group_size=3, payoff_matrix=payoff_matrix)
    pairwise_payoffs = egt.utils.transform_payoffs_to_pairwise(2, game)
    evolver1 = egt.analytical.PairwiseComparison(100, game)
    evolver2 = egt.analytical.StochDynamics(nb_strategies=2, payoffs=pairwise_payoffs, pop_size=100, group_size=3, mu=0)

    assert np.isclose(evolver1.calculate_fixation_probability(1, 0, 1), evolver2.fixation_probability(1, 0, 1))
    assert np.isclose(evolver1.calculate_gradient_of_selection(1, np.array([30, 70]))[0],
                      evolver2.gradient_selection(30, 0, 1, 1))

    transition1, fixation1 = evolver1.calculate_transition_and_fixation_matrix_sml(1)
    transition2, fixation2 = evolver2.transition_and_fixation_matrix(1)

    assert np.allclose(transition1.transpose(), transition2)
    assert np.allclose(fixation1, fixation2)

    full_transitions1 = evolver1.calculate_transition_matrix(1, 0.01)
    evolver2 = egt.analytical.StochDynamics(nb_strategies=2, payoffs=payoff_matrix, pop_size=100, group_size=3, mu=0.01)
    full_transitions2 = evolver2.calculate_full_transition_matrix(1)

    assert np.allclose(full_transitions1.toarray().transpose(), full_transitions2.toarray())


def test_abstract_replicator_game_python_subclass_pairwise():
    payoff_matrix = np.array([
        [-0.5, 2.0],
        [0.0, 0.0],
    ], dtype=np.float64)
    game = DummyPairwiseReplicatorGame(payoff_matrix)

    x = np.array([0.4, 0.6], dtype=np.float64)
    fitness = game.calculate_fitness(x)

    np.testing.assert_allclose(fitness, payoff_matrix @ x)
    np.testing.assert_allclose(game.calculate_payoffs(), payoff_matrix)
    np.testing.assert_allclose(game.payoffs(), payoff_matrix)
    assert game.nb_strategies() == 2
    assert game.group_size() == 2
    assert game.type() == "DummyPairwiseReplicatorGame"
    assert str(game) == "DummyPairwiseReplicatorGame"


def test_replicator_dynamics_2_player():
    v = 2
    d = 3
    t = 1
    payoff_matrix = np.array([
        [(v - d) / 2, v],
        [0, (v / 2) - t],
    ], dtype=np.float64)

    x = np.array([0.4, 0.6], dtype=np.float64)
    result = egt.analytical.replicator_equation(x, payoff_matrix)

    fitness = payoff_matrix @ x
    expected = x * (fitness - np.dot(x, fitness))

    np.testing.assert_allclose(result, expected)


def test_replicator_dynamics_2_player_with_abstract_replicator_game_matches_matrix():
    v = 2
    d = 3
    t = 1
    payoff_matrix = np.array([
        [(v - d) / 2, v],
        [0, (v / 2) - t],
    ], dtype=np.float64)

    game = DummyPairwiseReplicatorGame(payoff_matrix)
    x = np.array([0.4, 0.6], dtype=np.float64)

    result_matrix = egt.analytical.replicator_equation(x, payoff_matrix)
    result_game = egt.analytical.replicator_equation(x, game)

    np.testing.assert_allclose(result_game, result_matrix)


def test_replicator_dynamics_2_player_game_overload_rejects_non_pairwise_game():
    payoff_matrix = np.array([
        [-0.5, 2.0, 1.0, 3.0],
        [0.0, 0.0, 1.0, 2.0],
    ], dtype=np.float64)

    game = DummyNPlayerReplicatorGame(payoff_matrix)
    x = np.array([0.4, 0.6], dtype=np.float64)

    with pytest.raises((RuntimeError, ValueError)):
        egt.analytical.replicator_equation(x, game)


def test_abstract_replicator_game_python_subclass_n_player():
    payoff_matrix = np.array([
        [-0.5, 2.0, 1.0, 3.0],
        [0.0, 0.0, 1.0, 2.0],
    ], dtype=np.float64)
    game = DummyNPlayerReplicatorGame(payoff_matrix)

    x = np.array([0.4, 0.6], dtype=np.float64)
    fitness = game.calculate_fitness(x)

    p0 = x[0] ** 2
    p1 = 2.0 * x[0] * x[1]
    p2 = x[1] ** 2
    expected_fitness = np.array([
        p0 * payoff_matrix[0, 0] + p1 * payoff_matrix[0, 1] + p2 * payoff_matrix[0, 2],
        p0 * payoff_matrix[1, 1] + p1 * payoff_matrix[1, 2] + p2 * payoff_matrix[1, 3],
        ], dtype=np.float64)

    np.testing.assert_allclose(fitness, expected_fitness)
    np.testing.assert_allclose(game.calculate_payoffs(), payoff_matrix)
    np.testing.assert_allclose(game.payoffs(), payoff_matrix)
    assert game.nb_strategies() == 2
    assert game.group_size() == 3
    assert game.type() == "DummyNPlayerReplicatorGame"
    assert str(game) == "DummyNPlayerReplicatorGame"


def test_replicator_dynamics_n_player():
    v = 2
    d = 3
    t = 1
    payoff_matrix = np.array([
        [(v - d) / 2, v, v / 2, d],
        [0, (v / 2) - t, t, v],
    ], dtype=np.float64)

    x = np.array([0.4, 0.6], dtype=np.float64)
    result = egt.analytical.replicator_equation_n_player(x, payoff_matrix, group_size=3)

    assert result.shape == (2,)
    assert np.isfinite(result).all()


def test_replicator_dynamics_n_player_with_abstract_replicator_game_matches_matrix():
    payoff_matrix = np.array([
        [-0.5, 2.0, 1.0, 3.0],
        [0.0, 0.0, 1.0, 2.0],
    ], dtype=np.float64)

    game = DummyNPlayerReplicatorGame(payoff_matrix)
    x = np.array([0.4, 0.6], dtype=np.float64)

    result_matrix = egt.analytical.replicator_equation_n_player(x, payoff_matrix, group_size=3)
    result_game = egt.analytical.replicator_equation_n_player(x, game)

    np.testing.assert_allclose(result_game, result_matrix)
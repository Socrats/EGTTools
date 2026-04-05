import egttools as egt
import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pytest

matplotlib.use("Agg")


def test_plot_gradients_2d_array():
    gradients = np.random.rand(10, 2)
    # all ok returns None
    assert isinstance(egt.plotting.plot_gradients(gradients), plt.Axes)


def test_plot_gradient_none_input():
    # should error on input not being a np.array
    # accessing a particular atribute
    with pytest.raises(AttributeError):
        egt.plotting.plot_gradients(None)


def test_draw_invasion_diagram():
    # preliminary from the examples
    T, R, P, S, beta, Z = 2, 1, 0, 1, .01, 100
    A = np.array([[P, T], [S, R]])

    strategies = [
        egt.behaviors.NormalForm.TwoActions.Cooperator(),
        egt.behaviors.NormalForm.TwoActions.Random(),
        egt.behaviors.NormalForm.TwoActions.GRIM(),
    ]

    strategy_labels = [
        strategy.type().replace("NFGStrategies::", "") for strategy in strategies
    ]

    game = egt.games.NormalFormGame(1, A, strategies)
    evolver = egt.analytical.PairwiseComparison(Z, game)
    (
        transition_matrix,
        fixation_probabilities,
    ) = evolver.calculate_transition_and_fixation_matrix_sml(beta)

    stationary_distribution = egt.utils.calculate_stationary_distribution(
        transition_matrix.transpose()
    )

    assert isinstance(
        egt.plotting.draw_invasion_diagram(
            strategies=strategy_labels,
            drift=1,
            fixation_probabilities=fixation_probabilities,
            stationary_distribution=stationary_distribution,
        ),
        nx.DiGraph,
    )


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
    def __init__(self, payoff_matrix: np.ndarray):
        super().__init__()
        payoff_matrix = np.asarray(payoff_matrix, dtype=np.float64)
        if payoff_matrix.shape != (3, 10):
            raise ValueError("DummyNPlayerReplicatorGame requires a payoff matrix of shape (3, 10).")
        self._payoff_matrix = payoff_matrix

    def calculate_payoffs(self):
        return self._payoff_matrix

    def calculate_fitness(self, frequencies):
        x = np.asarray(frequencies, dtype=np.float64)
        x0, x1, x2 = x

        p200 = x0 * x0
        p110 = 2.0 * x0 * x1
        p101 = 2.0 * x0 * x2
        p020 = x1 * x1
        p011 = 2.0 * x1 * x2
        p002 = x2 * x2

        fitness = np.empty(3, dtype=np.float64)

        fitness[0] = (
                p200 * self._payoff_matrix[0, 0]
                + p110 * self._payoff_matrix[0, 1]
                + p101 * self._payoff_matrix[0, 2]
                + p020 * self._payoff_matrix[0, 3]
                + p011 * self._payoff_matrix[0, 4]
                + p002 * self._payoff_matrix[0, 5]
        )

        fitness[1] = (
                p200 * self._payoff_matrix[1, 1]
                + p110 * self._payoff_matrix[1, 3]
                + p101 * self._payoff_matrix[1, 4]
                + p020 * self._payoff_matrix[1, 6]
                + p011 * self._payoff_matrix[1, 7]
                + p002 * self._payoff_matrix[1, 8]
        )

        fitness[2] = (
                p200 * self._payoff_matrix[2, 2]
                + p110 * self._payoff_matrix[2, 4]
                + p101 * self._payoff_matrix[2, 5]
                + p020 * self._payoff_matrix[2, 7]
                + p011 * self._payoff_matrix[2, 8]
                + p002 * self._payoff_matrix[2, 9]
        )

        return fitness

    def nb_strategies(self):
        return 3

    def group_size(self):
        return 3

    def __str__(self):
        return "DummyNPlayerReplicatorGame"

    def type(self):
        return "DummyNPlayerReplicatorGame"

    def payoffs(self):
        return self._payoff_matrix


def test_plot_replicator_dynamics_in_simplex_with_pairwise_payoff_matrix():
    payoff_matrix = np.array([
        [3.0, 0.0, 1.0],
        [2.0, 1.0, 0.0],
        [0.0, 2.0, 2.5],
    ], dtype=np.float64)

    simplex, grad_fn, roots, roots_xy, stability = egt.plotting.plot_replicator_dynamics_in_simplex(
        payoff_matrix=payoff_matrix,
        group_size=2,
        nb_points_simplex=10,
        nb_of_initial_points_for_root_search=5,
    )

    assert isinstance(simplex, egt.plotting.Simplex2D)
    assert callable(grad_fn)
    assert isinstance(roots, list)
    assert isinstance(roots_xy, list)
    assert isinstance(stability, list)

    x = np.array([0.2, 0.5, 0.3], dtype=np.float64)
    expected = egt.analytical.replicator_equation(x, payoff_matrix)
    np.testing.assert_allclose(grad_fn(x, 0.0), expected)


def test_plot_replicator_dynamics_in_simplex_with_abstract_game_keeps_old_logic():
    payoff_matrix = np.array([
        [3.0, 0.0, 1.0],
        [2.0, 1.0, 0.0],
        [0.0, 2.0, 2.5],
    ], dtype=np.float64)
    game = egt.games.Matrix2PlayerGameHolder(3, payoff_matrix)

    simplex, grad_fn, roots, roots_xy, stability = egt.plotting.plot_replicator_dynamics_in_simplex(
        game=game,
        nb_points_simplex=10,
        nb_of_initial_points_for_root_search=5,
    )

    assert isinstance(simplex, egt.plotting.Simplex2D)
    assert isinstance(roots, list)
    assert isinstance(roots_xy, list)
    assert isinstance(stability, list)

    x = np.array([0.2, 0.5, 0.3], dtype=np.float64)
    expected = egt.analytical.replicator_equation(x, payoff_matrix)
    np.testing.assert_allclose(grad_fn(x, 0.0), expected)


def test_plot_replicator_dynamics_in_simplex_with_abstract_replicator_game_pairwise():
    payoff_matrix = np.array([
        [3.0, 0.0, 1.0],
        [2.0, 1.0, 0.0],
        [0.0, 2.0, 2.5],
    ], dtype=np.float64)
    game = DummyPairwiseReplicatorGame(payoff_matrix)

    simplex, grad_fn, roots, roots_xy, stability = egt.plotting.plot_replicator_dynamics_in_simplex(
        game=game,
        nb_points_simplex=10,
        nb_of_initial_points_for_root_search=5,
    )

    assert isinstance(simplex, egt.plotting.Simplex2D)
    assert isinstance(roots, list)
    assert isinstance(roots_xy, list)
    assert isinstance(stability, list)

    x = np.array([0.2, 0.5, 0.3], dtype=np.float64)
    expected = egt.analytical.replicator_equation(x, game)
    np.testing.assert_allclose(grad_fn(x, 0.0), expected)


def test_plot_replicator_dynamics_in_simplex_with_n_player_payoff_matrix():
    payoff_matrix = np.array([
        [1.0, 1.2, 0.8, 0.4, 1.1, 0.9, 0.5, 0.7, 0.6, 0.3],
        [0.5, 0.8, 1.3, 1.6, 0.9, 1.1, 1.4, 0.7, 1.0, 1.2],
        [0.4, 0.6, 0.7, 1.5, 0.8, 1.0, 1.2, 0.9, 1.1, 1.3],
    ], dtype=np.float64)

    simplex, grad_fn, roots, roots_xy, stability = egt.plotting.plot_replicator_dynamics_in_simplex(
        payoff_matrix=payoff_matrix,
        group_size=3,
        nb_points_simplex=10,
        nb_of_initial_points_for_root_search=5,
    )

    assert isinstance(simplex, egt.plotting.Simplex2D)
    assert isinstance(roots, list)
    assert isinstance(roots_xy, list)
    assert isinstance(stability, list)

    x = np.array([0.2, 0.5, 0.3], dtype=np.float64)
    expected = egt.analytical.replicator_equation_n_player(x, payoff_matrix, 3)
    np.testing.assert_allclose(grad_fn(x, 0.0), expected)


def test_plot_replicator_dynamics_in_simplex_with_abstract_replicator_game_n_player():
    payoff_matrix = np.array([
        [1.0, 1.2, 0.8, 0.4, 1.1, 0.9, 0.5, 0.7, 0.6, 0.3],
        [0.5, 0.8, 1.3, 1.6, 0.9, 1.1, 1.4, 0.7, 1.0, 1.2],
        [0.4, 0.6, 0.7, 1.5, 0.8, 1.0, 1.2, 0.9, 1.1, 1.3],
    ], dtype=np.float64)
    game = DummyNPlayerReplicatorGame(payoff_matrix)

    simplex, grad_fn, roots, roots_xy, stability = egt.plotting.plot_replicator_dynamics_in_simplex(
        game=game,
        nb_points_simplex=10,
        nb_of_initial_points_for_root_search=5,
    )

    assert isinstance(simplex, egt.plotting.Simplex2D)
    assert isinstance(roots, list)
    assert isinstance(roots_xy, list)
    assert isinstance(stability, list)

    x = np.array([0.2, 0.5, 0.3], dtype=np.float64)
    expected = egt.analytical.replicator_equation_n_player(x, game)
    np.testing.assert_allclose(grad_fn(x, 0.0), expected)


def test_plot_replicator_dynamics_in_simplex_requires_input():
    with pytest.raises(ValueError):
        egt.plotting.plot_replicator_dynamics_in_simplex()


def test_plot_pairwise_comparison_rule_dynamics_in_simplex_with_payoff_matrix():
    payoff_matrix = np.array([
        [3.0, 0.0, 1.0],
        [2.0, 1.0, 0.0],
        [0.0, 2.0, 2.5],
    ], dtype=np.float64)

    simplex, grad_fn, roots, roots_xy, stability, game, evolver = (
        egt.plotting.plot_pairwise_comparison_rule_dynamics_in_simplex(
            population_size=8,
            beta=1.0,
            payoff_matrix=payoff_matrix,
            group_size=2,
        )
    )

    assert isinstance(simplex, egt.plotting.Simplex2D)
    assert callable(grad_fn)
    assert isinstance(roots, list)
    assert isinstance(roots_xy, list)
    assert isinstance(stability, list)
    assert isinstance(game, egt.games.AbstractGame)
    assert isinstance(evolver, egt.analytical.PairwiseComparison)


def test_plot_pairwise_comparison_rule_dynamics_in_simplex_with_game():
    payoff_matrix = np.array([
        [3.0, 0.0, 1.0],
        [2.0, 1.0, 0.0],
        [0.0, 2.0, 2.5],
    ], dtype=np.float64)
    game = egt.games.Matrix2PlayerGameHolder(3, payoff_matrix)

    simplex, grad_fn, roots, roots_xy, stability, returned_game, evolver = (
        egt.plotting.plot_pairwise_comparison_rule_dynamics_in_simplex(
            population_size=8,
            beta=1.0,
            game=game,
        )
    )

    assert isinstance(simplex, egt.plotting.Simplex2D)
    assert callable(grad_fn)
    assert isinstance(roots, list)
    assert isinstance(roots_xy, list)
    assert isinstance(stability, list)
    assert returned_game is game
    assert isinstance(evolver, egt.analytical.PairwiseComparison)


def test_plot_pairwise_comparison_rule_dynamics_in_simplex_without_roots_with_payoff_matrix():
    payoff_matrix = np.array([
        [3.0, 0.0, 1.0],
        [2.0, 1.0, 0.0],
        [0.0, 2.0, 2.5],
    ], dtype=np.float64)

    simplex, grad_fn, game, evolver = (
        egt.plotting.plot_pairwise_comparison_rule_dynamics_in_simplex_without_roots(
            population_size=8,
            beta=1.0,
            payoff_matrix=payoff_matrix,
            group_size=2,
        )
    )

    assert isinstance(simplex, egt.plotting.Simplex2D)
    assert callable(grad_fn)
    assert isinstance(game, egt.games.AbstractGame)
    assert isinstance(evolver, egt.analytical.PairwiseComparison)


def test_plot_pairwise_comparison_rule_dynamics_in_simplex_without_roots_with_game():
    payoff_matrix = np.array([
        [3.0, 0.0, 1.0],
        [2.0, 1.0, 0.0],
        [0.0, 2.0, 2.5],
    ], dtype=np.float64)
    game = egt.games.Matrix2PlayerGameHolder(3, payoff_matrix)

    simplex, grad_fn, returned_game, evolver = (
        egt.plotting.plot_pairwise_comparison_rule_dynamics_in_simplex_without_roots(
            population_size=8,
            beta=1.0,
            game=game,
        )
    )

    assert isinstance(simplex, egt.plotting.Simplex2D)
    assert callable(grad_fn)
    assert returned_game is game
    assert isinstance(evolver, egt.analytical.PairwiseComparison)

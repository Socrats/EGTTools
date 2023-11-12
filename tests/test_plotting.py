import pytest
import numpy as np, networkx as nx, egttools as egt, matplotlib.pyplot as plt


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

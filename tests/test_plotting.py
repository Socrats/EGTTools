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


# ---------------------------------------------------------------------------
# Tests for draw_triangle with offset and draw_stationary_distribution_discrete
# ---------------------------------------------------------------------------

def _make_discrete_simplex(size: int = 10):
    """Helper: return a discrete Simplex2D with an axis attached."""
    simplex = egt.plotting.Simplex2D(discrete=True, size=size)
    simplex.add_axis(figsize=(6, 5))
    return simplex


def _uniform_sd(size: int = 10):
    """Uniform stationary distribution over the discrete 3-strategy simplex."""
    nb_states = egt.calculate_nb_states(size, 3)
    sd = np.ones(nb_states) / nb_states
    return sd


# --- draw_triangle tests ---

def test_draw_triangle_default():
    """Default call (offset=0) draws the triangle without raising."""
    simplex = _make_discrete_simplex()
    result = simplex.draw_triangle()
    assert result is simplex
    plt.close("all")


def test_draw_triangle_with_offset_returns_self():
    """draw_triangle with offset > 0 returns self for method chaining."""
    simplex = _make_discrete_simplex()
    result = simplex.draw_triangle(offset=0.03, corner_gap=0.05)
    assert result is simplex
    plt.close("all")


def test_draw_triangle_offset_produces_three_lines():
    """draw_triangle with offset draws exactly 3 line artists."""
    simplex = _make_discrete_simplex()
    n_before = len(simplex.ax.lines)
    simplex.draw_triangle(offset=0.03, corner_gap=0.05)
    n_after = len(simplex.ax.lines)
    assert n_after - n_before == 3
    plt.close("all")


def test_draw_triangle_no_offset_uses_fewer_lines_than_offset():
    """draw_triangle without offset uses triplot (not 3 individual segments)."""
    simplex_no_offset = _make_discrete_simplex()
    n_before = len(simplex_no_offset.ax.lines)
    simplex_no_offset.draw_triangle(offset=0.0)
    n_no_offset = len(simplex_no_offset.ax.lines) - n_before

    simplex_offset = _make_discrete_simplex()
    n_before2 = len(simplex_offset.ax.lines)
    simplex_offset.draw_triangle(offset=0.03)
    n_with_offset = len(simplex_offset.ax.lines) - n_before2

    # offset mode adds exactly 3 separate line segments; triplot adds fewer
    assert n_with_offset == 3
    assert n_no_offset != 3
    plt.close("all")


def test_draw_triangle_offset_segments_are_outside_corners():
    """Each offset edge segment should lie further from the centroid than the original edge."""
    simplex = _make_discrete_simplex()
    offset = 0.04
    simplex.draw_triangle(offset=offset, corner_gap=0.0)
    centroid = simplex.corners.mean(axis=0)
    # For each drawn segment the midpoint should be further from the centroid than
    # the corresponding original edge midpoint.
    orig_edges = [(0, 2), (2, 1), (1, 0)]
    for line, (i, j) in zip(simplex.ax.lines, orig_edges):
        xdata, ydata = line.get_xdata(), line.get_ydata()
        mid_drawn = np.array([(xdata[0] + xdata[1]) / 2, (ydata[0] + ydata[1]) / 2])
        mid_orig = (simplex.corners[i] + simplex.corners[j]) / 2
        d_drawn = np.linalg.norm(mid_drawn - centroid)
        d_orig = np.linalg.norm(mid_orig - centroid)
        assert d_drawn > d_orig - 1e-9
    plt.close("all")


# --- draw_stationary_distribution_discrete tests ---

def test_draw_stationary_distribution_discrete_returns_self():
    simplex = _make_discrete_simplex()
    sd = _uniform_sd()
    result = simplex.draw_stationary_distribution_discrete(sd)
    assert result is simplex
    plt.close("all")


def test_draw_stationary_distribution_discrete_adds_scatter():
    """The method should add exactly one PathCollection (scatter) to the axes."""
    simplex = _make_discrete_simplex()
    sd = _uniform_sd()
    n_before = len(simplex.ax.collections)
    simplex.draw_stationary_distribution_discrete(sd, colorbar=False)
    n_after = len(simplex.ax.collections)
    assert n_after - n_before == 1
    plt.close("all")


def test_draw_stationary_distribution_discrete_correct_number_of_points():
    """Scatter should contain one point per discrete state."""
    size = 10
    simplex = _make_discrete_simplex(size)
    sd = _uniform_sd(size)
    simplex.draw_stationary_distribution_discrete(sd, colorbar=False)
    collection = simplex.ax.collections[-1]
    assert len(collection.get_offsets()) == len(sd)
    plt.close("all")


def test_draw_stationary_distribution_discrete_with_hexagons():
    simplex = _make_discrete_simplex()
    sd = _uniform_sd()
    result = simplex.draw_stationary_distribution_discrete(sd, marker='h', colorbar=False)
    assert result is simplex
    plt.close("all")


def test_draw_stationary_distribution_discrete_raises_on_continuous_simplex():
    """Should raise if called on a non-discrete simplex."""
    simplex = egt.plotting.Simplex2D(discrete=False)
    simplex.add_axis()
    sd = np.array([0.5, 0.5])
    with pytest.raises(Exception, match="discrete"):
        simplex.draw_stationary_distribution_discrete(sd)
    plt.close("all")


def test_draw_stationary_distribution_discrete_uniform_sizes():
    """All markers should be the same size; probability is encoded via colour only."""
    size = 5
    simplex = _make_discrete_simplex(size)
    nb_states = egt.calculate_nb_states(size, 3)
    sd = np.zeros(nb_states)
    sd[0] = 1.0  # all mass on first state
    simplex.draw_stationary_distribution_discrete(sd, colorbar=False)
    collection = simplex.ax.collections[-1]
    sizes = collection.get_sizes()
    # Every marker must be exactly the same size
    assert np.all(sizes == pytest.approx(sizes[0], rel=1e-6))
    plt.close("all")


def test_draw_stationary_distribution_discrete_custom_marker_size():
    """Passing marker_size should override the auto-computed size."""
    simplex = _make_discrete_simplex()
    sd = _uniform_sd()
    custom_size = 42.0
    simplex.draw_stationary_distribution_discrete(sd, marker_size=custom_size, colorbar=False)
    collection = simplex.ax.collections[-1]
    sizes = collection.get_sizes()
    assert np.all(sizes == pytest.approx(custom_size, rel=1e-6))
    plt.close("all")


def test_draw_triangle_offset_and_discrete_distribution_compose():
    """Offset triangle and discrete distribution should compose without errors."""
    size = 8
    simplex = _make_discrete_simplex(size)
    sd = _uniform_sd(size)
    (simplex
     .draw_triangle(offset=0.03, corner_gap=0.05)
     .draw_stationary_distribution_discrete(sd, colorbar=False))
    assert len(simplex.ax.collections) >= 1
    assert len(simplex.ax.lines) == 3
    plt.close("all")


# --- Visual test (saves to /tmp for manual inspection) ---

def test_visual_discrete_stationary_distribution(tmp_path):
    """Visual test: renders offset triangle + discrete stationary distribution.

    The resulting figure is saved to tmp_path/visual_discrete_sd.png so you
    can inspect it manually.  The test passes as long as no exception is raised.
    """
    import os

    size = 12
    simplex = egt.plotting.Simplex2D(discrete=True, size=size)
    fig, ax = plt.subplots(figsize=(6, 5))
    simplex.add_axis(ax=ax)

    # Build a non-uniform stationary distribution (peaked near first corner)
    nb_states = egt.calculate_nb_states(size, 3)
    rng = np.random.default_rng(42)
    sd_raw = rng.exponential(scale=1.0, size=nb_states)
    sd = sd_raw / sd_raw.sum()

    (simplex
     .draw_triangle(offset=0.04, corner_gap=0.04)
     .draw_stationary_distribution_discrete(sd, colorbar=True,
                                            label='stationary dist.',
                                            shrink=0.5)
     .draw_axes(visible=False)
     .add_vertex_labels(['A', 'B', 'C']))

    ax.set_title("Discrete stationary distribution on Simplex2D\n"
                 "(offset triangle + hexagon markers, no axes)")
    fig.tight_layout()

    out = tmp_path / "visual_discrete_sd.png"
    fig.savefig(str(out), dpi=120)
    plt.close("all")

    assert out.exists(), f"Figure was not saved to {out}"
    print(f"\nVisual output saved to: {out}")

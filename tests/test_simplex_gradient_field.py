"""Tests for the continuous gradient field used by the pairwise comparison simplex plots.

The gradient of selection is defined only at integer population states
(docs/theory/pairwise_comparison.md, Definition 5.1); the plots extend it by linear
interpolation on the lattice triangles and find its zeros exactly.
"""
import warnings

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

import egttools as egt
from egttools.analytical import PairwiseComparison
from egttools.plotting.helpers import (
    calculate_stability,
    find_roots_of_lattice_gradient,
    lattice_gradient_interpolator,
)
from egttools.plotting.simplified import plot_pairwise_comparison_rule_dynamics_in_simplex

Z = 30
BETA = 1.0
# Anti-coordination: every strategy does better when rare, so the centre attracts and
# every vertex repels.
ANTI_COORDINATION = np.array([[0., 1., 1.], [1., 0., 1.], [1., 1., 0.]])


def exact_gradient(payoffs, mu=None):
    evolver = PairwiseComparison(Z, egt.games.Matrix2PlayerGameHolder(3, payoffs))
    if mu is None:
        return lambda n: evolver.calculate_gradient_of_selection(BETA, n)
    return lambda n: evolver.calculate_gradient_of_selection_with_mutation(BETA, mu, n)


def test_interpolator_is_exact_at_population_states():
    gradient = exact_gradient(np.random.default_rng(1).uniform(-1, 1, (3, 3)))
    interpolate, lattice = lattice_gradient_interpolator(gradient, Z)
    for i in range(Z + 1):
        for j in range(Z + 1 - i):
            n = np.array([i, j, Z - i - j])
            np.testing.assert_allclose(interpolate(n.astype(float)), gradient(n), atol=1e-14)
            np.testing.assert_allclose(lattice[i, j], gradient(n), atol=1e-14)


def test_interpolator_is_linear_inside_a_triangle_and_continuous():
    gradient = exact_gradient(np.random.default_rng(2).uniform(-1, 1, (3, 3)))
    interpolate, _ = lattice_gradient_interpolator(gradient, Z)
    a, b, c = (np.array(p, dtype=float) for p in ([10, 5, 15], [11, 5, 14], [10, 6, 14]))
    centroid = (a + b + c) / 3
    np.testing.assert_allclose(interpolate(centroid), (interpolate(a) + interpolate(b) + interpolate(c)) / 3,
                               atol=1e-14)
    # continuity across the shared edge of two triangles
    edge_point = (b + c) / 2
    np.testing.assert_allclose(interpolate(edge_point - 1e-9 * np.array([1, 1, -2])),
                               interpolate(edge_point + 1e-9 * np.array([1, 1, -2])), atol=1e-8)


def test_interpolator_never_casts_fractional_states():
    gradient_calls = []

    def recording_gradient(n):
        gradient_calls.append(np.asarray(n))
        return exact_gradient(ANTI_COORDINATION)(n)

    interpolate, _ = lattice_gradient_interpolator(recording_gradient, Z)
    interpolate(np.array([10.3, 9.9, 9.8]))
    assert all(np.issubdtype(n.dtype, np.integer) and n.sum() == Z for n in gradient_calls)


@pytest.mark.parametrize("mu", [None, 0.05])
def test_roots_are_zeros_of_the_interpolated_field(mu):
    gradient = exact_gradient(np.random.default_rng(3).uniform(-1, 1, (3, 3)), mu)
    interpolate, lattice = lattice_gradient_interpolator(gradient, Z)
    roots = find_roots_of_lattice_gradient(lattice)
    assert roots
    for root in roots:
        assert root.sum() == pytest.approx(1.)
        np.testing.assert_allclose(interpolate(root * Z), 0., atol=1e-9)


def test_anti_coordination_roots_and_stability():
    interpolate, lattice = lattice_gradient_interpolator(exact_gradient(ANTI_COORDINATION), Z)
    roots = find_roots_of_lattice_gradient(lattice)
    as_tuples = {tuple(np.round(r, 6)) for r in roots}
    assert (round(1 / 3, 6),) * 3 in as_tuples
    for vertex in np.eye(3):
        assert tuple(vertex) in as_tuples

    stability = dict(zip((tuple(np.round(r, 6)) for r in roots),
                         calculate_stability(roots, lambda x: interpolate(np.asarray(x) * Z), return_mode="int")))
    assert stability[(round(1 / 3, 6),) * 3] == 1          # stable
    for vertex in np.eye(3):
        assert stability[tuple(vertex)] == -1              # unstable


@pytest.mark.parametrize("mu", [None, 0.05])
def test_plot_has_no_casting_warnings_and_finds_the_stable_centre(mu):
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        _, gradient_fn, roots, roots_xy, stability, _, _ = plot_pairwise_comparison_rule_dynamics_in_simplex(
            population_size=Z, beta=BETA, mu=mu, payoff_matrix=ANTI_COORDINATION)
    centre = [k for k, r in enumerate(roots) if np.allclose(r, 1 / 3, atol=1e-6)]
    assert len(centre) == 1
    assert stability[centre[0]] == 1
    assert len(roots_xy) == len(roots) == len(stability)
    # the returned function works in counts, like Simplex2D.draw_trajectory_from_roots expects
    np.testing.assert_allclose(gradient_fn(np.array([10., 10., 10.]), 0), 0., atol=1e-12)


def test_stability_classification_matches_replicator_dynamics_of_coordination_game():
    from egttools.analytical import replicator_equation
    payoffs = np.eye(3)  # pure coordination
    f = lambda x: replicator_equation(np.asarray(x, dtype=np.float64), payoffs)
    vertices = list(np.eye(3))
    edge_midpoints = [np.array([.5, .5, 0.]), np.array([.5, 0., .5]), np.array([0., .5, .5])]
    centre = [np.full(3, 1 / 3)]
    stability = calculate_stability(vertices + edge_midpoints + centre, f, return_mode="int")
    assert stability == [1, 1, 1, 0, 0, 0, -1]

"""Colormap support in the gradient, 2D simplex and 3D simplex plots."""
import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection, PathCollection

import egttools as egt
from egttools.analytical import replicator_equation
from egttools.plotting import Simplex2D

RPS = np.array([[0., -1., 1.], [1., 0., -1.], [-1., 1., 0.]])


def replicator(u, t):
    return replicator_equation(np.asarray(u, dtype=np.float64), RPS)


def line_collections(ax):
    return [c for c in ax.collections if isinstance(c, LineCollection)]


# --- 1D gradient plots --------------------------------------------------------------------------

def test_plot_gradients_with_cmap_colours_by_value_centred_on_zero():
    x = np.linspace(0, 1, 51)
    gradient = x * (1 - x) * (x - 0.3)
    ax = egt.plotting.plot_gradients(gradient, cmap="RdBu")
    (collection,) = line_collections(ax)
    np.testing.assert_allclose(collection.get_array(), (gradient[:-1] + gradient[1:]) / 2)
    norm = collection.norm
    assert norm.vcenter == 0. and norm.halfrange == pytest.approx(np.abs(gradient).max())
    plt.close("all")


def test_plot_gradients_with_cmap_and_several_curves_share_the_norm():
    x = np.linspace(0, 1, 31)
    gradients = np.column_stack([x * (1 - x), -2 * x * (1 - x)])
    ax = egt.plotting.plot_gradients(gradients, cmap="viridis")
    collections = line_collections(ax)
    assert len(collections) == 2
    assert collections[0].norm.halfrange == collections[1].norm.halfrange == pytest.approx(np.abs(gradients).max())
    plt.close("all")


def test_plot_gradients_without_cmap_draws_a_line():
    ax = egt.plotting.plot_gradients(np.linspace(-1, 1, 11))
    assert not line_collections(ax) and len(ax.lines) >= 1
    plt.close("all")


def test_plot_gradient_with_cmap():
    x = np.linspace(0, 1, 21)
    ax = egt.plotting.plot_gradient(x, x * (1 - x) * (x - .5), np.array([0., .5, 1.]),
                                    np.array([True, False, True]), [], cmap="coolwarm")
    assert len(line_collections(ax)) == 1
    plt.close("all")


# --- 2D simplex trajectories --------------------------------------------------------------------

def simplex():
    s = Simplex2D()
    s.add_axis()
    return s


def test_trajectory_from_points_with_cmap_is_coloured_along_time_with_arrow():
    s = simplex()
    s.draw_trajectory_from_points(replicator, [np.array([.6, .3, .1])], trajectory_length=5, step=.1,
                                  cmap="viridis", draw_arrow=True)
    (collection,) = line_collections(s.ax)
    colors = collection.get_colors()
    assert len(colors) == 49 and not np.allclose(colors[0], colors[-1])   # start and end differ
    assert len(s.ax.texts) == 1                                         # the arrow annotation
    plt.close("all")


def test_trajectory_from_roots_with_cmap():
    s = simplex()
    s.draw_trajectory_from_roots(replicator, [np.full(3, 1 / 3)], [False], trajectory_length=3, step=.1,
                                 cmap="plasma")
    assert len(line_collections(s.ax)) >= 1
    plt.close("all")


def test_trajectory_from_vector_and_random_trajectories_with_cmap():
    s = simplex()
    s.draw_trajectory_from_vector(np.array([[1., 0., 0.], [.5, .5, 0.], [.3, .3, .4]]), cmap="magma")
    s.draw_trajectories(replicator, 2, trajectory_length=1, step=.1, cmap="cividis")
    assert len(line_collections(s.ax)) == 3
    plt.close("all")


def test_scatter_shadow_with_cmap():
    s = simplex()
    s.draw_scatter_shadow(replicator, 1, trajectory_length=1, step=.1, cmap="viridis")
    (points,) = [c for c in s.ax.collections if isinstance(c, PathCollection)]
    np.testing.assert_allclose(points.get_array(), np.linspace(0, 1, 10))
    plt.close("all")


def test_trajectories_without_cmap_are_unchanged():
    s = simplex()
    s.draw_trajectory_from_points(replicator, [np.array([.6, .3, .1])], trajectory_length=2, step=.1)
    assert not line_collections(s.ax) and len(s.ax.lines) == 1
    plt.close("all")


# --- 3D simplex (plotly) ------------------------------------------------------------------------

def test_plotly_colorscale_conversion():
    pytest.importorskip("plotly")
    from egttools.plotting.simplex3d import _plotly_colorscale
    assert _plotly_colorscale("Viridis") == "Viridis"                 # plotly name: unchanged
    for colormap in ("coolwarm", plt.get_cmap("coolwarm")):           # matplotlib name or object
        scale = _plotly_colorscale(colormap)
        assert len(scale) == 11 and scale[0][0] == 0. and scale[-1][0] == 1.
        assert scale[0][1].startswith("rgb(")
    explicit = [[0., "rgb(0,0,0)"], [1., "rgb(255,255,255)"]]
    assert _plotly_colorscale(explicit) is explicit
    with pytest.raises(ValueError):
        _plotly_colorscale("not-a-colormap")


def test_simplex3d_accepts_a_matplotlib_colormap():
    pytest.importorskip("plotly")
    from egttools.plotting import Simplex3D
    payoffs = np.eye(4)
    s = Simplex3D().draw_slice(fixed_strategy=0, value=.2, colorscale="coolwarm", n_grid=4, n_seeds=5,
                               gradient_fn=lambda x: replicator_equation(np.asarray(x, dtype=np.float64), payoffs))
    assert len(s._traces) > 0

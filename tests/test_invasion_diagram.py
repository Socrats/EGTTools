"""Tests for egttools.plotting.invasion_graph and draw_invasion_diagram."""
import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

import egttools as egt
from egttools.analytical import PairwiseComparison

Z = 50


def sml(payoffs, beta=1.0):
    evolver = PairwiseComparison(Z, egt.games.Matrix2PlayerGameHolder(len(payoffs), np.asarray(payoffs, float)))
    transitions, fixation = evolver.calculate_transition_and_fixation_matrix_sml(beta)
    return np.asarray(fixation), egt.utils.calculate_stationary_distribution(np.asarray(transitions).T)


def edges(graph):
    return {(u, v): d["kind"] for u, v, d in graph.edges(data=True)}


def ratio_matrix(r01, r10):
    """Fixation matrix for two strategies with given rho/rho_N in each direction."""
    return np.array([[0., r01], [r10, 0.]]) / Z


def test_pair_classification():
    tol = 0.005
    # one-way selection: 1 invades 0
    assert edges(egt.plotting.invasion_graph(ratio_matrix(3., .2), Z)) == {(0, 1): "selection"}
    # mutual invasion: two directed edges
    assert edges(egt.plotting.invasion_graph(ratio_matrix(3., 2.), Z)) == {(0, 1): "selection", (1, 0): "selection"}
    # exact and near-neutral pairs: one drift edge
    assert edges(egt.plotting.invasion_graph(ratio_matrix(1., 1.), Z)) == {(0, 1): "drift"}
    assert edges(egt.plotting.invasion_graph(ratio_matrix(1 + tol / 2, 1 - tol / 2), Z)) == {(0, 1): "drift"}
    # neither invades (bistable pair): no edge
    assert edges(egt.plotting.invasion_graph(ratio_matrix(.5, .4), Z)) == {}
    # just above the tolerance: selection
    assert edges(egt.plotting.invasion_graph(ratio_matrix(1 + 2 * tol, 1 - 2 * tol), Z)) == {(0, 1): "selection"}


def test_near_neutral_strategies_are_drift():
    # X and Y differ by 1e-5 in one payoff: rho/rho_N is 1.0002 and 0.9999
    fixation, sd = sml([[1., 1., 2.], [1. + 1e-5, 1., 2.], [0., 0., 1.]])
    graph = egt.plotting.invasion_graph(fixation, Z, sd, ["X", "Y", "W"])
    assert edges(graph) == {(0, 1): "drift", (2, 0): "selection", (2, 1): "selection"}


def test_graph_attributes_and_filtering():
    fixation, sd = sml([[0., 3., 1.], [1., 2., 1.], [-1., -1., 0.]])
    graph = egt.plotting.invasion_graph(fixation, Z, sd, ["H", "D", "L"])
    assert graph.graph["population_size"] == Z
    assert graph.nodes[0]["label"] == "H"
    assert graph.nodes[0]["frequency"] == pytest.approx(sd[0])
    for u, v, data in graph.edges(data=True):
        assert data["ratio"] == pytest.approx(fixation[u, v] * Z)
    filtered = egt.plotting.invasion_graph(fixation, Z, sd, ["H", "D", "L"], min_frequency=1e-3)
    assert 2 not in filtered.nodes                      # the dominated strategy is filtered out
    assert all(2 not in e for e in filtered.edges)


def test_duplicate_labels_do_not_merge_strategies():
    graph = egt.plotting.invasion_graph(ratio_matrix(1., 1.), Z, labels=["A", "A"])
    assert graph.number_of_nodes() == 2


@pytest.mark.parametrize("bad", [dict(fixation=np.zeros((2, 3))), dict(population_size=1),
                                 dict(labels=["only one"]), dict(sd=np.ones(3))])
def test_invalid_input(bad):
    with pytest.raises(ValueError):
        egt.plotting.invasion_graph(bad.get("fixation", ratio_matrix(1., 1.)), bad.get("population_size", Z),
                                    bad.get("sd"), bad.get("labels"))


def test_drawing_mutual_invasion_and_drift():
    fixation, sd = sml([[0., 3., 1., 0.], [1., 2., 1., 1.], [-1., -1., 0., -1.], [0., 3., 1., 0.]])
    graph = egt.plotting.invasion_graph(fixation, Z, sd, ["Hawk", "Dove", "Loser", "HawkCopy"])
    _, ax = plt.subplots()
    returned = egt.plotting.draw_invasion_diagram(graph, ax=ax)
    assert returned is ax
    patches = [p for p in ax.patches if isinstance(p, FancyArrowPatch)]
    kinds = list(edges(graph).values())
    assert len(patches) == len(kinds)
    curved = [p for p in patches if getattr(p.get_connectionstyle(), "rad", 0) != 0]
    mutual = [e for e in graph.edges if graph.has_edge(e[1], e[0]) and graph.edges[e]["kind"] == "selection"]
    assert len(curved) == len(mutual) == 4               # Hawk<->Dove and HawkCopy<->Dove
    dashed = [p for p in patches if p.get_linestyle() in ("--", "dashed")]
    assert len(dashed) == kinds.count("drift") == 1      # Hawk -- HawkCopy
    ratio_texts = [t for t in ax.texts if "rho" in t.get_text()]
    assert len(ratio_texts) == kinds.count("selection")
    plt.close("all")


def test_drawing_with_filter_and_colours_indexed_by_strategy():
    fixation, sd = sml([[0., 3., 1.], [1., 2., 1.], [-1., -1., 0.]])
    graph = egt.plotting.invasion_graph(fixation, Z, sd, ["H", "D", "L"], min_frequency=1e-3)
    _, ax = plt.subplots()
    egt.plotting.draw_invasion_diagram(graph, ax=ax, colors=["red", "blue", "gray"],
                                       scale_nodes_by_frequency=True, scale_edges_by_ratio=True)
    facecolors = ax.collections[-1].get_facecolors()
    np.testing.assert_allclose(facecolors[:, :3], [matplotlib.colors.to_rgb(c) for c in ("red", "blue")])
    _, ax = plt.subplots()
    egt.plotting.draw_invasion_diagram(graph, ax=ax, colors={"H": "red", "D": "blue"}, layout={0: (0, 0), 1: (1, 0)})
    plt.close("all")


def test_min_ratio_and_labels_can_be_hidden():
    fixation, sd = sml([[0., 3., 1.], [1., 2., 1.], [-1., -1., 0.]])
    graph = egt.plotting.invasion_graph(fixation, Z, sd)
    _, ax = plt.subplots()
    egt.plotting.draw_invasion_diagram(graph, ax=ax, min_ratio=1e6, show_frequencies=False)
    assert not [p for p in ax.patches if isinstance(p, FancyArrowPatch)]
    assert {t.get_text() for t in ax.texts} == {"0", "1", "2"}
    plt.close("all")


@pytest.mark.parametrize("name_position", ["outside", "inside"])
@pytest.mark.parametrize("frequency_position", ["outside", "inside"])
def test_label_positions(name_position, frequency_position):
    fixation, sd = sml([[0., 3., 1.], [1., 2., 1.], [-1., -1., 0.]])
    graph = egt.plotting.invasion_graph(fixation, Z, sd, ["H", "D", "L"])
    _, ax = plt.subplots()
    egt.plotting.draw_invasion_diagram(graph, ax=ax, name_position=name_position,
                                       frequency_position=frequency_position, show_ratios=False,
                                       frequency_format="percent", label_offsets={"L": (0., -40.)})
    texts = {t.get_text(): t for t in ax.texts}
    assert {"H", "D", "L"} <= set(texts)
    percentages = {f"{f:.0%}" if f >= 0.005 or f == 0 else "<1%" for f in sd}
    assert percentages <= set(texts)
    if name_position == "outside":
        assert texts["L"].xyann == (0., -40.)            # custom offset used for the outside name
    else:
        assert abs(texts["H"].xyann[0]) < 1e-12          # name centred horizontally on the node
    plt.close("all")


def test_frequency_format():
    from egttools.plotting.indicators import _format_percentage
    assert _format_percentage(0.4567) == "46%"
    assert _format_percentage(0.001) == "<1%"
    assert _format_percentage(0.0) == "0%"
    graph = egt.plotting.invasion_graph(ratio_matrix(3., .2), Z, np.array([0.25, 0.75]))
    _, ax = plt.subplots()
    egt.plotting.draw_invasion_diagram(graph, ax=ax, frequency_format="{:.3f}")
    assert {"0.250", "0.750"} <= {t.get_text() for t in ax.texts}
    _, ax = plt.subplots()
    egt.plotting.draw_invasion_diagram(graph, ax=ax, frequency_format=lambda f: f"p={f:.1f}")
    assert {"p=0.2", "p=0.8"} <= {t.get_text() for t in ax.texts}
    plt.close("all")


@pytest.mark.parametrize("bad", [dict(name_position="above"), dict(frequency_position="left")])
def test_invalid_label_position(bad):
    graph = egt.plotting.invasion_graph(ratio_matrix(3., .2), Z)
    with pytest.raises(ValueError):
        egt.plotting.draw_invasion_diagram(graph, **bad)
    plt.close("all")


def test_contrasting_text_color():
    from egttools.plotting.indicators import _contrasting_text_color
    assert _contrasting_text_color("navy") == "white"
    assert _contrasting_text_color("yellow") != "white"


def node_sizes(ax):
    return np.asarray(ax.collections[-1].get_sizes())


@pytest.fixture
def three_strategy_graph():
    fixation, sd = sml([[0., 3., 1.], [1., 2., 1.], [-1., -1., 0.]])
    return egt.plotting.invasion_graph(fixation, Z, sd, ["H", "D", "L"]), sd


def test_default_style_is_simple(three_strategy_graph):
    graph, sd = three_strategy_graph
    _, ax = plt.subplots()
    egt.plotting.draw_invasion_diagram(graph, ax=ax)
    assert np.all(node_sizes(ax) == 2000.)                          # equal nodes
    assert {f"{f:.2f}" for f in sd} <= {t.get_text() for t in ax.texts}  # plain frequencies
    plt.close("all")


def test_proportional_style_scales_nodes_with_a_minimum(three_strategy_graph):
    graph, sd = three_strategy_graph
    _, ax = plt.subplots()
    egt.plotting.draw_invasion_diagram(graph, ax=ax, style="proportional")
    sizes = node_sizes(ax)
    assert sizes.max() == pytest.approx(4500.)
    assert sizes.min() >= 600.
    assert np.all(np.diff(sizes[np.argsort(sd)]) >= 0)            # larger frequency, larger node
    assert any(t.get_text().endswith("%") for t in ax.texts)
    plt.close("all")


def test_explicit_arguments_override_the_style(three_strategy_graph):
    graph, _ = three_strategy_graph
    _, ax = plt.subplots()
    egt.plotting.draw_invasion_diagram(graph, ax=ax, style="proportional", scale_nodes_by_frequency=False,
                                       node_size=1000.)
    assert np.all(node_sizes(ax) == 1000.)
    _, ax = plt.subplots()
    egt.plotting.draw_invasion_diagram(graph, ax=ax, style={"node_size": 1234., "frequency_format": "percent"})
    assert np.all(node_sizes(ax) == 1234.)
    plt.close("all")


@pytest.mark.parametrize("style", sorted(egt.plotting.INVASION_DIAGRAM_STYLES))
def test_every_named_style_draws(three_strategy_graph, style):
    graph, _ = three_strategy_graph
    _, ax = plt.subplots()
    egt.plotting.draw_invasion_diagram(graph, ax=ax, style=style)
    plt.close("all")


def test_unknown_style():
    graph = egt.plotting.invasion_graph(ratio_matrix(3., .2), Z)
    with pytest.raises(ValueError, match="available"):
        egt.plotting.draw_invasion_diagram(graph, style="fancy")
    plt.close("all")


def test_scaling_is_skipped_without_stationary_distribution():
    graph = egt.plotting.invasion_graph(ratio_matrix(3., .2), Z)
    _, ax = plt.subplots()
    egt.plotting.draw_invasion_diagram(graph, ax=ax, style="proportional")
    assert np.all(node_sizes(ax) == 4500.)
    plt.close("all")


@pytest.mark.parametrize("palette", ["Set2", "viridis", plt.get_cmap("plasma")])
def test_palette_names_and_colormaps(three_strategy_graph, palette):
    graph, _ = three_strategy_graph
    _, ax = plt.subplots()
    egt.plotting.draw_invasion_diagram(graph, ax=ax, colors=palette)
    facecolors = ax.collections[-1].get_facecolors()
    assert len({tuple(np.round(c, 4)) for c in facecolors}) == 3     # three distinct colours
    plt.close("all")


def test_palette_colours_follow_strategies_when_filtered():
    fixation, sd = sml([[0., 3., 1.], [1., 2., 1.], [-1., -1., 0.]])
    full = egt.plotting.invasion_graph(fixation, Z, sd)
    filtered = egt.plotting.invasion_graph(fixation, Z, sd, min_frequency=1e-3)
    _, (ax1, ax2) = plt.subplots(1, 2)
    egt.plotting.draw_invasion_diagram(full, ax=ax1, colors="Set2")
    egt.plotting.draw_invasion_diagram(filtered, ax=ax2, colors="Set2")
    np.testing.assert_allclose(ax1.collections[-1].get_facecolors()[:2], ax2.collections[-1].get_facecolors())
    plt.close("all")

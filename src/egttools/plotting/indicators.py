# Copyright (c) 2019-2021  Elias Fernandez
#
# This file is part of EGTtools.
#
# EGTtools is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# EGTtools is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with EGTtools.  If not, see <http://www.gnu.org/licenses/>

"""Helper function to visualize evolutionary dynamics"""

import networkx as nx
import numpy
import numpy as np
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib import patheffects

from matplotlib.ticker import AutoMinorLocator
from typing import Dict, List, Optional, Sequence, Tuple, Union


def plot_gradients(gradients: numpy.ndarray, fig_title: Optional[str] = None,
                   xlabel: Optional[str] = 'frequency of strategy 0',
                   ylabel: Optional[str] = 'gradient of selection (G)',
                   figsize: Optional[Tuple[int, int]] = (7, 5),
                   color: Optional[Optional[Union[str, Tuple[int, int, int], List[str], List[
                       Tuple[int, int, int]], plt.cm.colors.Colormap]]] = 'b',
                   linelabel: Optional[str] = None,
                   linewidth_gradient: Optional[Union[int, List[int]]] = 3,
                   marker: Optional[str] = None,
                   marker_plot_freq: Optional[int] = 1,
                   marker_linewidth: Optional[int] = 3,
                   marker_size: Optional[int] = 20,
                   marker_facecolor: Optional[Optional[Union[str, Tuple[int, int, int], List[str], List[
                       Tuple[int, int, int]], plt.cm.colors.Colormap]]] = 'b',
                   marker_edgecolor: Optional[Optional[Union[str, Tuple[int, int, int], List[str], List[
                       Tuple[int, int, int]], plt.cm.colors.Colormap]]] = 'b',
                   roots: Optional[List[np.ndarray]] = None,
                   stability: Optional[List[int]] = None,
                   linewidth_edges: Optional[Union[int, List[int]]] = 3,
                   edgecolors: Optional[str] = 'black',
                   nodesize: Optional[int] = 100,
                   nodeborder_width: Optional[int] = 3,
                   arrowstyle: Optional[str] = '-|>',
                   nb_minor_ticks: Optional[int] = 2,
                   major_ticks_length: Optional[int] = 7,
                   minor_ticks_length: Optional[int] = 4,
                   ticks_width: Optional[int] = 2,
                   ticks_labels_size: Optional[int] = 15,
                   ticks_direction: Optional[str] = 'in',
                   ticks_labels_pad: Optional[int] = 10,
                   ticks_labels_fontweight: Optional[Union[str, int]] = 'bold',
                   axis_labels_fontweight: Optional[Union[str, int]] = 'bold',
                   axis_labels_fontsize: Optional[int] = 15,
                   ticks_left: Optional[bool] = True,
                   ticks_right: Optional[bool] = True,
                   ticks_top: Optional[bool] = True,
                   ticks_bottom: Optional[bool] = True,
                   spine_left_linewidth: Optional[int] = 2,
                   spine_right_linewidth: Optional[int] = 2,
                   spine_top_linewidth: Optional[int] = 2,
                   spine_bottom_linewidth: Optional[int] = 2,
                   ax: Optional[plt.axis] = None) -> plt.axis:
    """
    This function plots the gradient of selection for 1-simplexes (only two strategies).

    There is the possibility of plotting the stationary points (roots) of the system and their stability,
    but it is recommended that you only do this when analysing the replicator equation.

    Parameters
    ----------
    gradients: a numpy array with the gradients to plot.
    fig_title: a title for the figure.
    xlabel: the label of the x axis.
    ylabel: the label of the y axis.
    figsize: the dimensions of the figure.
    color: the color to use to plot the line.
    linelabel: label assigned to the plotted line.
    linewidth_gradient: width of the gradient curve.
    marker: use a marker to plot the points (by default no marker is shown).
    marker_plot_freq: how often to plot a marker (so that there aren't many overlapping).
    marker_linewidth: linewidth of the edge of the marker.
    marker_size: size of the marker.
    marker_facecolor: marker fill color.
    marker_edgecolor: marker edge color.
    roots: a list of numpy arrays containing the coordinates of the stationary points of the dynamical system.
    stability: a list of integers indicating the stability of the roots (-1 - unstable, 0 - saddle, 1 - stable).
    linewidth_edges: width of the arrows indicating the direction of the gradients.
    edgecolors: color of the arrows indicating the direction of selection.
    nodesize: size of the circles representing the roots.
    nodeborder_width: width of the border of the circles.
    arrowstyle: style of the arrows that represent the direction of selection.
    nb_minor_ticks: number of minor ticks to display.
    major_ticks_length: length of major ticks.
    minor_ticks_length: length of minor ticks.
    ticks_width: width of the ticks.
    ticks_labels_size: size of the tick labels.
    ticks_direction: direction of the ticks ("in" or "out").
    ticks_labels_pad: pad for the labels of the ticks.
    ticks_labels_fontweight: font weight of the tick labels (e.g., "bold').
    axis_labels_fontweight: font weight of the axis labels (e.g., "bold').
    axis_labels_fontsize: font size of the axis labels.
    ticks_left: indicates whether to display ticks on the left spine.
    ticks_right: indicates whether to display ticks on the right spine.
    ticks_top: indicates whether to display ticks on the top spine.
    ticks_bottom: indicates whether to display ticks on the bottom spine.
    spine_left_linewidth: line width of the left spine.
    spine_right_linewidth: line width of the right spine.
    spine_top_linewidth: line width of the top spine.
    spine_bottom_linewidth: line width of the bottom spine.
    ax: a matplotlib.pyplot axis object in which this figure will be plot. If None, then a new axis and figure will
        be created.

    Returns
    -------
    matplotlib.pyplot.axis

    The axis in which the figure has been plot.
    """
    x_values = np.linspace(0, 1, num=gradients.shape[0], dtype=np.float64)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    if fig_title is not None:
        ax.set_title(fig_title)

    ax.plot(x_values, gradients, color=color, linewidth=linewidth_gradient, label=linelabel, zorder=1)
    if marker is not None:
        [ ax.scatter(x_values[::marker_plot_freq], gradient[::marker_plot_freq], 
marker=marker, s=marker_size, facecolors=marker_facecolor,
			   edgecolors=marker_edgecolor, linewidths=marker_linewidth, zorder=1.5) for gradient in gradients.T]
    # if stability is given, draw the stable points and arrows
    if stability is not None:
        if roots is None:
            raise Exception("To plot the stability you must also input the roots of the dynamical system.")

        # sort roots by first element, this will make the plotting of gradient arrows in
        # the egttools.plotting.indicators.plot_gradients work
        indexes = sorted(range(len(roots)), key=lambda k: roots[k][0])
        sorted_roots = [roots[i] for i in indexes]
        sorted_stability = [stability[i] for i in indexes]

        G = nx.DiGraph()
        G.add_nodes_from(np.arange(len(sorted_roots)))

        pos = {0: (sorted_roots[0][0], 0)}
        # Create edges
        for i in range(1, len(sorted_roots)):
            pos[i] = (sorted_roots[i][0], 0)

            if (sorted_stability[i] == -1) and (sorted_stability[i - 1] == 1):
                G.add_edge(i, i - 1)
            elif (sorted_stability[i] == -1) and (sorted_stability[i - 1] == 0):
                G.add_edge(i, i - 1)
            elif (sorted_stability[i] == -1) and (sorted_stability[i - 1] == -1):
                G.add_edge(i, i - 1)
                G.add_edge(i - 1, -1)
            elif (sorted_stability[i] == 0) and (sorted_stability[i - 1] == 1):
                G.add_edge(i, i - 1)
            elif (sorted_stability[i] == 0) and (sorted_stability[i - 1] == -1):
                G.add_edge(i - 1, i)
            elif (sorted_stability[i] == 1) and (sorted_stability[i - 1] == -1):
                G.add_edge(i - 1, i)
            elif (sorted_stability[i] == 1) and (sorted_stability[i - 1] == 0):
                G.add_edge(i - 1, i)

        def get_color(x):
            if x == -1:
                return 'white'
            elif x == 0:
                return 'gray'
            else:
                return 'black'

        node_colors = list(map(get_color, sorted_stability))

        nx.draw(G, pos=pos, edgecolors=edgecolors, linewidths=nodeborder_width, node_size=nodesize,
                node_color=node_colors, width=linewidth_edges,
                arrowstyle=arrowstyle,
                ax=ax)
        ax.set_xlim(-0.1, 1.1)
        ax.spines['left'].set_position(('axes', 0.083))
        ax.spines['right'].set_position(('axes', 0.917))
        ax.spines.top.set_bounds((0, 1))
        ax.spines.bottom.set_bounds((0, 1))
        ax.spines.left.set_zorder(0)
        ax.spines.right.set_zorder(0)
        ax.spines.top.set_zorder(0)
        ax.spines.bottom.set_zorder(0)
    else:
        ax.plot([0, 1], [0, 0], color='k', linewidth=linewidth_edges, zorder=1.5)
        ax.set_xlim(0, 1)

    ax.axis('on')  # turns on axis
    # ax.set_xticks(np.linspace(0, 1, 6))
    ax.xaxis.set_minor_locator(AutoMinorLocator(nb_minor_ticks))
    ax.yaxis.set_minor_locator(AutoMinorLocator(nb_minor_ticks))
    ax.tick_params(which='both', left=ticks_left, bottom=ticks_bottom, top=ticks_top, right=ticks_right, labelleft=True,
                   labelbottom=True,
                   width=ticks_width,
                   labelsize=ticks_labels_size, direction=ticks_direction, pad=ticks_labels_pad)
    ax.tick_params(which='major', length=major_ticks_length)
    ax.tick_params(which='minor', length=minor_ticks_length)
    ax.spines['left'].set_linewidth(spine_left_linewidth)
    ax.spines['right'].set_linewidth(spine_right_linewidth)
    ax.spines['top'].set_linewidth(spine_top_linewidth)
    ax.spines['bottom'].set_linewidth(spine_bottom_linewidth)

    for label in ax.get_xticklabels():
        label.set_fontweight(ticks_labels_fontweight)  # If change to 551, label will be bold-like
    for label in ax.get_yticklabels():
        label.set_fontweight(ticks_labels_fontweight)  # If change to 551, label will be bold-like

    ax.set_xlabel(xlabel, fontsize=axis_labels_fontsize, fontweight=axis_labels_fontweight)
    ax.set_ylabel(ylabel, fontsize=axis_labels_fontsize, fontweight=axis_labels_fontweight)

    return ax


def plot_gradient(x, gradients, saddle_points, saddle_type, gradient_direction, fig_title='', xlabel='', figsize=(5, 4),
                  **kwargs):
    """
    Creates a figure plotting the gradient of selection together with the saddle points,
    and the gradient arrows.

    :param x: vector containing the possible states in x axis. It must have the same length as gradient
    :param gradients: vector containing the gradient for each possible state
    :param saddle_points: vector containing all saddle points
    :param saddle_type: vector of booleans indicating whether the saddle point is stable
    :param gradient_direction: vector of points indicating the direction of the gradient
                               between unstable and stable saddle points
    :param fig_title: a string containing the title of the figure
    :param xlabel: label for x axis
    :param figsize: a tuple indicating the size of the figure
    :param kwargs: you may pass an axis object
    :returns a figure object
    """
    if 'ax' not in kwargs:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        ax = kwargs['ax']
    ax.plot(x, gradients, zorder=1)
    ax.plot([0, 1], [0, 0], 'k', zorder=1)
    # First we plot unstable points
    if saddle_points[~saddle_type].any():
        points1 = ax.scatter(x=saddle_points[~saddle_type],
                             y=np.zeros((len(saddle_points[~saddle_type], ))),
                             marker="o", color='k', s=80,
                             facecolors='white',
                             zorder=3)
        points1.set_clip_on(False)  # Plot points over axis
    # Then we plot stable points
    if saddle_points[saddle_type].any():
        points2 = ax.scatter(x=saddle_points[saddle_type],
                             y=np.zeros((len(saddle_points[saddle_type], ))),
                             marker="o",
                             color='k',
                             s=80,
                             zorder=3)
        points2.set_clip_on(False)
    # Plot arrows following the gradient
    for point in gradient_direction:
        ax.annotate("",
                    xy=(point[1], 0.0), xycoords='data',
                    xytext=(point[0], 0.0), textcoords='data',
                    arrowprops=dict(arrowstyle="->",
                                    connectionstyle="arc3"),
                    zorder=2
                    )
    # Set the axis style
    ax.set_xlim(0, 1)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('$G(x)$')
    ax.set_title(fig_title)
    return ax


def invasion_graph(fixation_probabilities: np.ndarray,
                   population_size: int,
                   stationary_distribution: Optional[np.ndarray] = None,
                   labels: Optional[Sequence[str]] = None,
                   drift_tolerance: float = 0.005,
                   min_frequency: float = 0.) -> nx.DiGraph:
    """
    Builds the invasion graph of the small-mutation limit.

    Each pair of strategies is classified once, by comparing its fixation probabilities with
    the neutral fixation probability :math:`\\rho_N = 1/Z`. With
    :math:`r_{ij} = \\rho_{ij} / \\rho_N`, where :math:`\\rho_{ij}` is the probability that a
    single mutant of strategy :math:`j` takes over a population of strategy :math:`i`:

    - **selection**: an edge :math:`i \\to j` for every direction with
      :math:`r_{ij} > 1 + \\text{drift\\_tolerance}` (both edges if each strategy invades the other);
    - **drift**: one undirected edge if neither direction exceeds :math:`1 + \\text{drift\\_tolerance}`
      and at least one is within ``drift_tolerance`` of 1 (neutral or nearly neutral pair);
    - no edge otherwise (neither strategy invades the other).

    Parameters
    ----------
    fixation_probabilities : numpy.ndarray
        Square matrix whose entry ``[i, j]`` is the probability that a single mutant of strategy
        ``j`` fixates in a population of strategy ``i`` (as returned by
        ``calculate_transition_and_fixation_matrix_sml``).
    population_size : int
        Population size :math:`Z`; the neutral fixation probability is :math:`1/Z`.
    stationary_distribution : numpy.ndarray, optional
        Stationary distribution over the monomorphic states, stored as the ``frequency`` of each node.
    labels : Sequence[str], optional
        Name of each strategy; defaults to the strategy indices.
    drift_tolerance : float, default=0.005
        Tolerance on :math:`r_{ij}` within which a pair counts as neutral. The default treats every
        ratio that would be displayed as ``1.00`` as drift.
    min_frequency : float, default=0
        Strategies whose stationary frequency is below this value are left out.

    Returns
    -------
    networkx.DiGraph
        Nodes are strategy indices with attributes ``label`` and ``frequency``. Edges have
        attributes ``ratio`` (:math:`r_{ij}`) and ``kind`` (``"selection"`` or ``"drift"``).
        ``graph.graph`` stores ``population_size`` and ``drift_tolerance``.
    """
    fixation_probabilities = np.asarray(fixation_probabilities, dtype=np.float64)
    if fixation_probabilities.ndim != 2 or fixation_probabilities.shape[0] != fixation_probabilities.shape[1]:
        raise ValueError("fixation_probabilities must be a square matrix")
    nb_strategies = fixation_probabilities.shape[0]
    if population_size < 2:
        raise ValueError("population_size must be >= 2")
    if labels is None:
        labels = [str(i) for i in range(nb_strategies)]
    if len(labels) != nb_strategies:
        raise ValueError("labels must have one entry per strategy")
    if stationary_distribution is None:
        frequencies = np.full(nb_strategies, np.nan)
    else:
        frequencies = np.asarray(stationary_distribution, dtype=np.float64)
        if frequencies.shape != (nb_strategies,):
            raise ValueError("stationary_distribution must have one entry per strategy")

    ratios = fixation_probabilities * population_size
    kept = [i for i in range(nb_strategies) if np.isnan(frequencies[i]) or frequencies[i] >= min_frequency]

    graph = nx.DiGraph(population_size=population_size, drift_tolerance=drift_tolerance)
    for i in kept:
        graph.add_node(i, label=labels[i], frequency=float(frequencies[i]))
    for position, i in enumerate(kept):
        for j in kept[position + 1:]:
            r_ij, r_ji = ratios[i, j], ratios[j, i]
            if r_ij > 1 + drift_tolerance:
                graph.add_edge(i, j, ratio=float(r_ij), kind="selection")
            if r_ji > 1 + drift_tolerance:
                graph.add_edge(j, i, ratio=float(r_ji), kind="selection")
            if (r_ij <= 1 + drift_tolerance and r_ji <= 1 + drift_tolerance
                    and (abs(r_ij - 1) <= drift_tolerance or abs(r_ji - 1) <= drift_tolerance)):
                graph.add_edge(i, j, ratio=float(max(r_ij, r_ji)), kind="drift")
    return graph


def _contrasting_text_color(color) -> str:
    """Black or white, whichever is more readable on ``color`` (WCAG relative luminance)."""
    channels = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in mcolors.to_rgb(color)]
    luminance = 0.2126 * channels[0] + 0.7152 * channels[1] + 0.0722 * channels[2]
    return "#222222" if luminance > 0.179 else "white"


def _node_colors(graph: nx.DiGraph, colors) -> List:
    nodes = list(graph.nodes)
    if colors is None:
        from seaborn import color_palette
        palette = color_palette("colorblind", max(nodes, default=0) + 1)
        return [palette[i] for i in nodes]
    if isinstance(colors, dict):
        return [colors[i] if i in colors else colors[graph.nodes[i]["label"]] for i in nodes]
    return [colors[i] for i in nodes]  # indexed by strategy, so filtered graphs keep their colours


def draw_invasion_diagram(graph: nx.DiGraph,
                          ax: Optional[plt.Axes] = None,
                          layout: Union[str, Dict[int, Tuple[float, float]]] = "circular",
                          colors: Optional[Union[Sequence, Dict]] = None,
                          node_size: float = 2000.,
                          scale_nodes_by_frequency: bool = False,
                          edge_width: float = 1.5,
                          scale_edges_by_ratio: bool = False,
                          min_ratio: float = 1.,
                          curvature: float = 0.2,
                          show_ratios: bool = True,
                          show_frequencies: bool = True,
                          ratio_format: str = r"{:.2f}$\rho_N$",
                          frequency_format: str = "{:.2f}",
                          ratio_label_position: float = 0.35,
                          label_position: str = "outside",
                          label_offsets: Optional[Dict] = None,
                          label_padding: float = 4.,
                          node_kwargs: Optional[Dict] = None,
                          edge_kwargs: Optional[Dict] = None,
                          drift_kwargs: Optional[Dict] = None,
                          label_kwargs: Optional[Dict] = None,
                          ratio_label_kwargs: Optional[Dict] = None,
                          frequency_label_kwargs: Optional[Dict] = None) -> plt.Axes:
    """
    Draws an invasion diagram from a graph built with :func:`invasion_graph`.

    Selection edges are arrows from the resident to the strategy that invades it, labelled with
    :math:`\\rho/\\rho_N`; pairs that invade each other are drawn as two curved arrows with
    separate labels. Neutral pairs are drawn as dashed lines without arrow heads. The stationary
    frequency of each strategy is written next to its node, away from the centre of the diagram.

    Parameters
    ----------
    graph : networkx.DiGraph
        Graph returned by :func:`invasion_graph`.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on; a new 6x6 inch figure is created if omitted.
    layout : str or dict, default="circular"
        ``"circular"``, ``"shell"``, ``"spring"`` (seeded, so reproducible) or a dict mapping
        each node (strategy index) to an ``(x, y)`` position.
    colors : sequence or dict, optional
        Node colours, either a sequence indexed by strategy index (covering all strategies, so
        colours stay the same when strategies are filtered out) or a dict keyed by strategy index
        or label. Defaults to seaborn's colorblind palette.
    node_size : float, default=2000
        Node area in points^2 (the largest node, if ``scale_nodes_by_frequency``).
    scale_nodes_by_frequency : bool, default=False
        Make node areas proportional to the stationary frequency (with a minimum of 15%).
    edge_width : float, default=1.5
        Line width of the edges.
    scale_edges_by_ratio : bool, default=False
        Scale selection edges by :math:`1 + \\log_{10}(\\rho/\\rho_N)`.
    min_ratio : float, default=1
        Hide selection edges with :math:`\\rho/\\rho_N` below this value (declutters large diagrams).
    curvature : float, default=0.2
        Curvature of the two arrows of a pair that invade each other.
    show_ratios, show_frequencies : bool, default=True
        Whether to write :math:`\\rho/\\rho_N` on selection edges and the stationary frequency next
        to each node.
    ratio_format, frequency_format : str
        Format strings for those labels.
    ratio_label_position : float, default=0.35
        Where to write :math:`\\rho/\\rho_N` along each edge, as a fraction of the way from the
        resident to the invader. Values other than 0.5 keep the labels of edges that cross at
        their midpoints (e.g. diameters of a circular layout) apart.
    label_position : {"outside", "inside"}, default="outside"
        ``"outside"`` writes each strategy's name, with its stationary frequency on the line below,
        on the side of the node facing away from the centre of the diagram. ``"inside"`` writes the
        name inside the node (in black or white, whichever contrasts with the node colour) and the
        frequency outside.
    label_offsets : dict, optional
        Per-strategy placement of the outside labels, keyed by strategy index or label, as an
        ``(dx, dy)`` offset in points from the node centre. Overrides the automatic placement.
    label_padding : float, default=4
        Gap, in points, between the edge of a node and its outside labels.
    node_kwargs, edge_kwargs, drift_kwargs : dict, optional
        Extra arguments for ``networkx.draw_networkx_nodes`` and ``networkx.draw_networkx_edges``
        (selection and drift edges).
    label_kwargs, ratio_label_kwargs, frequency_label_kwargs : dict, optional
        Extra text properties (e.g. ``fontsize``, ``color``) for strategy names, edge ratios and
        frequencies. All text has a thin white outline so it stays readable over edges; pass
        ``path_effects=[]`` to remove it.

    Returns
    -------
    matplotlib.axes.Axes
        The axes containing the diagram.

    Examples
    --------
    >>> import egttools as egt
    >>> import numpy as np
    >>> Z = 100
    >>> game = egt.games.Matrix2PlayerGameHolder(3, np.array([[1., 0, 2], [2, 1, 0], [0, 2, 1]]))
    >>> evolver = egt.analytical.PairwiseComparison(Z, game)
    >>> transitions, fixation_probabilities = evolver.calculate_transition_and_fixation_matrix_sml(1.)
    >>> sd = egt.utils.calculate_stationary_distribution(transitions.T)
    >>> graph = egt.plotting.invasion_graph(fixation_probabilities, Z, sd, labels=["R", "P", "S"])
    >>> ax = egt.plotting.draw_invasion_diagram(graph)
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))
    nodes = list(graph.nodes)

    if isinstance(layout, dict):
        pos = {node: np.asarray(layout[node], dtype=np.float64) for node in nodes}
    elif layout == "circular":
        pos = nx.circular_layout(graph)
    elif layout == "shell":
        pos = nx.shell_layout(graph)
    elif layout == "spring":
        pos = nx.spring_layout(graph, seed=0)
    else:
        raise ValueError("layout must be 'circular', 'shell', 'spring' or a dict of positions")

    frequencies = np.array([graph.nodes[i]["frequency"] for i in nodes], dtype=np.float64)
    if scale_nodes_by_frequency and nodes and np.nanmax(frequencies) > 0:
        sizes = [node_size * max(f / np.nanmax(frequencies), 0.15) if np.isfinite(f) else node_size
                 for f in frequencies]
    else:
        sizes = [node_size] * len(nodes)

    # Edges first so that nodes are drawn on top of them.
    common = dict(nodelist=nodes, node_size=sizes, ax=ax)
    for u, v, data in graph.edges(data=True):
        if data["kind"] == "drift":
            nx.draw_networkx_edges(graph, pos, edgelist=[(u, v)], width=edge_width, style="dashed",
                                   alpha=0.6, arrows=True, arrowstyle="-", **common, **(drift_kwargs or {}))
            continue
        if data["ratio"] < min_ratio:
            continue
        reciprocal = graph.has_edge(v, u) and graph.edges[v, u]["kind"] == "selection"
        rad = curvature if reciprocal else 0.
        width = edge_width * (1 + np.log10(data["ratio"])) if scale_edges_by_ratio else edge_width
        nx.draw_networkx_edges(graph, pos, edgelist=[(u, v)], width=width, arrows=True, arrowstyle="-|>",
                               arrowsize=15, connectionstyle=f"arc3,rad={rad}", **common, **(edge_kwargs or {}))
        if show_ratios:
            # Point at parameter s on matplotlib's arc3 curve: a quadratic Bezier curve whose control
            # point is mid + rad * (dy, -dx) (valid because the axes have equal aspect).
            p0, p2 = np.asarray(pos[u], dtype=np.float64), np.asarray(pos[v], dtype=np.float64)
            control = (p0 + p2) / 2 + rad * np.array([p2[1] - p0[1], p0[0] - p2[0]])
            s = ratio_label_position
            x, y = (1 - s) ** 2 * p0 + 2 * (1 - s) * s * control + s ** 2 * p2
            ax.text(x, y, ratio_format.format(data["ratio"]), ha="center", va="center", zorder=3,
                    **{"fontsize": 9, "color": "#222222",
                       "path_effects": [patheffects.withStroke(linewidth=3, foreground="white")],
                       **(ratio_label_kwargs or {})})

    nx.draw_networkx_nodes(graph, pos, nodelist=nodes, node_size=sizes, node_color=_node_colors(graph, colors),
                           ax=ax, **{"edgecolors": "white", "linewidths": 1.5, **(node_kwargs or {})})

    node_colors = _node_colors(graph, colors)
    halo = [patheffects.withStroke(linewidth=3, foreground="white")]
    name_style = {"fontsize": 11, "fontweight": "bold", "color": "#222222", "path_effects": halo,
                  **(label_kwargs or {})}
    frequency_style = {"fontsize": 9, "color": "#555555", "path_effects": halo, **(frequency_label_kwargs or {})}
    line_height = 1.25 * name_style.get("fontsize", 11)

    centre = np.mean([pos[i] for i in nodes], axis=0) if nodes else np.zeros(2)
    for i, size, node_color in zip(nodes, sizes, node_colors):
        x, y = pos[i]
        name = graph.nodes[i]["label"]
        frequency = graph.nodes[i]["frequency"]
        frequency_text = (frequency_format.format(frequency)
                          if show_frequencies and np.isfinite(frequency) else None)

        # Direction away from the centre of the diagram; labels go on that side of the node.
        direction = np.asarray(pos[i], dtype=np.float64) - centre
        norm = np.linalg.norm(direction)
        direction = direction / norm if norm > 1e-12 else np.array([0., -1.])
        # Node sizes are areas in points^2, so the radius is sqrt(size) / 2 points.
        offset = (np.sqrt(size) / 2 + label_padding) * direction
        if label_offsets is not None:
            custom = label_offsets.get(i, label_offsets.get(name))
            if custom is not None:
                offset = np.asarray(custom, dtype=np.float64)
        ha = "left" if offset[0] > 0.3 * np.linalg.norm(offset) else (
            "right" if offset[0] < -0.3 * np.linalg.norm(offset) else "center")
        above = offset[1] > 0.3 * np.linalg.norm(offset)
        below = offset[1] < -0.3 * np.linalg.norm(offset)

        def place(text, dy, va, style):
            ax.annotate(text, xy=(x, y), xytext=(offset[0], offset[1] + dy), textcoords="offset points",
                        ha=ha, va=va, zorder=5, **style)

        if label_position == "inside":
            ax.annotate(name, xy=(x, y), ha="center", va="center", zorder=5,
                        **{**name_style, "color": _contrasting_text_color(node_color), "path_effects": [],
                           **(label_kwargs or {})})
            if frequency_text is not None:
                place(frequency_text, 0., "bottom" if above else "top" if below else "center", frequency_style)
        elif label_position == "outside":
            # Name above frequency, as one block on the outer side of the node.
            if frequency_text is None:
                place(name, 0., "bottom" if above else "top" if below else "center", name_style)
            elif above:
                place(frequency_text, 0., "bottom", frequency_style)
                place(name, line_height, "bottom", name_style)
            elif below:
                place(name, 0., "top", name_style)
                place(frequency_text, -line_height, "top", frequency_style)
            else:
                place(name, 0., "bottom", name_style)
                place(frequency_text, 0., "top", frequency_style)
        else:
            raise ValueError("label_position must be 'outside' or 'inside'")

    ax.set_aspect("equal")
    ax.margins(0.2)
    ax.set_axis_off()
    return ax

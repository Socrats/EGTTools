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
from matplotlib.collections import LineCollection

from matplotlib.ticker import AutoMinorLocator
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union


def _plot_gradient_curves(ax: plt.Axes, x: np.ndarray, gradients: np.ndarray, color, cmap, linewidth, label,
                          zorder) -> None:
    """Plots one or several gradient curves, in ``color`` or coloured by their value with ``cmap``.

    With ``cmap``, the colour norm is centred on 0 and shared by all curves, so a diverging colormap
    separates positive from negative selection.
    """
    if cmap is None:
        ax.plot(x, gradients, color=color, linewidth=linewidth, label=label, zorder=zorder)
        return
    curves = np.asarray(gradients, dtype=np.float64).reshape(len(x), -1)
    halfrange = float(np.nanmax(np.abs(curves))) if np.isfinite(curves).any() else 1.
    norm = mcolors.CenteredNorm(vcenter=0., halfrange=halfrange if halfrange > 0 else 1.)
    for k in range(curves.shape[1]):
        y = curves[:, k]
        segments = np.stack([np.column_stack([x[:-1], y[:-1]]), np.column_stack([x[1:], y[1:]])], axis=1)
        collection = LineCollection(segments, cmap=cmap, norm=norm, linewidths=linewidth, zorder=zorder,
                                    label=label if k == 0 else None)
        collection.set_array((y[:-1] + y[1:]) / 2)
        ax.add_collection(collection)
    ax.autoscale_view()


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
                   ax: Optional[plt.axis] = None,
                   cmap: Optional[Union[str, mcolors.Colormap]] = None) -> plt.axis:
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
    cmap: if given, colour the gradient curve(s) by the value of the gradient with this colormap (instead of
        ``color``). The colour norm is centred on 0, so a diverging colormap (e.g. ``"RdBu"``) shows where
        selection favours and where it opposes the strategy.

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

    _plot_gradient_curves(ax, x_values, gradients, color, cmap, linewidth_gradient, linelabel, zorder=1)
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
                  cmap=None, **kwargs):
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
    :param cmap: if given, colour the gradient by its value with this colormap (norm centred on 0)
    :param kwargs: you may pass an axis object
    :returns a figure object
    """
    if 'ax' not in kwargs:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        ax = kwargs['ax']
    _plot_gradient_curves(ax, np.asarray(x, dtype=np.float64), np.asarray(gradients, dtype=np.float64), None, cmap,
                          None, None, zorder=1)
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


#: Named presets for :func:`draw_invasion_diagram` (``style=``). Each maps parameter names to values;
#: explicit arguments to ``draw_invasion_diagram`` take precedence.
INVASION_DIAGRAM_STYLES: Dict[str, Dict] = {
    "simple": dict(node_size=2000., min_node_size=600., scale_nodes_by_frequency=False,
                   scale_edges_by_ratio=False, name_position="outside", frequency_position="inside",
                   frequency_format="{:.2f}"),
    "proportional": dict(node_size=4500., min_node_size=600., scale_nodes_by_frequency=True,
                         scale_edges_by_ratio=True, name_position="outside", frequency_position="inside",
                         frequency_format="percent"),
    "classic": dict(node_size=2000., min_node_size=600., scale_nodes_by_frequency=False,
                    scale_edges_by_ratio=False, name_position="outside", frequency_position="outside",
                    frequency_format="{:.2f}"),
    "compact": dict(node_size=2200., min_node_size=600., scale_nodes_by_frequency=False,
                    scale_edges_by_ratio=False, name_position="inside", frequency_position="inside",
                    frequency_format="percent", label_kwargs={"fontsize": 8},
                    frequency_label_kwargs={"fontsize": 8}),
}


def _invasion_style(style: Union[str, Dict]) -> Dict:
    """Resolves ``style`` to a complete preset (a dict fills in from ``"simple"``)."""
    if isinstance(style, dict):
        return {**INVASION_DIAGRAM_STYLES["simple"], **style}
    if style not in INVASION_DIAGRAM_STYLES:
        raise ValueError(f"unknown style {style!r}; available: {', '.join(INVASION_DIAGRAM_STYLES)}")
    return INVASION_DIAGRAM_STYLES[style]


def _format_percentage(frequency: float) -> str:
    """Whole percentage, with "<1%" for positive frequencies that would round to 0%."""
    if 0. < frequency < 0.005:
        return "<1%"
    return f"{frequency:.0%}"


def _contrasting_text_color(color) -> str:
    """Black or white, whichever is more readable on ``color`` (WCAG relative luminance)."""
    channels = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in mcolors.to_rgb(color)]
    luminance = 0.2126 * channels[0] + 0.7152 * channels[1] + 0.0722 * channels[2]
    return "#222222" if luminance > 0.179 else "white"


def _node_colors(graph: nx.DiGraph, colors) -> List:
    nodes = list(graph.nodes)
    nb_colors = max(nodes, default=0) + 1  # one colour per strategy index, so filtering keeps colours
    if colors is None:
        colors = "colorblind"
    if isinstance(colors, str):
        from seaborn import color_palette
        palette = color_palette(colors, nb_colors)  # seaborn palettes and matplotlib colormap names
        return [palette[i] for i in nodes]
    if isinstance(colors, mcolors.Colormap):
        return [colors(i / max(nb_colors - 1, 1)) for i in nodes]
    if isinstance(colors, dict):
        return [colors[i] if i in colors else colors[graph.nodes[i]["label"]] for i in nodes]
    return [colors[i] for i in nodes]  # indexed by strategy, so filtered graphs keep their colours


def draw_invasion_diagram(graph: nx.DiGraph,
                          ax: Optional[plt.Axes] = None,
                          style: Union[str, Dict] = "simple",
                          layout: Union[str, Dict[int, Tuple[float, float]]] = "circular",
                          colors: Optional[Union[str, mcolors.Colormap, Sequence, Dict]] = None,
                          node_size: Optional[float] = None,
                          scale_nodes_by_frequency: Optional[bool] = None,
                          min_node_size: Optional[float] = None,
                          edge_width: float = 1.5,
                          scale_edges_by_ratio: Optional[bool] = None,
                          min_ratio: float = 1.,
                          curvature: float = 0.2,
                          show_ratios: bool = True,
                          show_frequencies: bool = True,
                          ratio_format: str = r"{:.2f}$\rho_N$",
                          frequency_format: Union[str, Callable[[float], str], None] = None,
                          ratio_label_position: float = 0.35,
                          name_position: Optional[str] = None,
                          frequency_position: Optional[str] = None,
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
    separate labels. Neutral pairs are drawn as dashed lines without arrow heads. The look is
    chosen with ``style`` (see :data:`INVASION_DIAGRAM_STYLES`) and every element can be adjusted
    with the parameters below.

    Parameters
    ----------
    graph : networkx.DiGraph
        Graph returned by :func:`invasion_graph`.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on; a new 6x6 inch figure is created if omitted.
    style : str or dict, default="simple"
        Named preset from :data:`INVASION_DIAGRAM_STYLES`, or a dict with the same keys. It sets
        the parameters below that are left as ``None`` (node sizes and scaling, edge scaling,
        label positions, frequency format and label font sizes); explicit arguments always win.

        - ``"simple"``: equal nodes, stationary frequency inside each node (``0.47``), name outside.
        - ``"proportional"``: node area grows with the stationary frequency and arrow width with
          :math:`\\log_{10}(\\rho/\\rho_N)`; frequency inside as a percentage, name outside.
        - ``"classic"``: equal nodes, name and frequency outside the node.
        - ``"compact"``: equal nodes with name and percentage inside, smaller fonts; suited to
          diagrams with many strategies.
    layout : str or dict, default="circular"
        ``"circular"``, ``"shell"``, ``"spring"`` (seeded, so reproducible) or a dict mapping
        each node (strategy index) to an ``(x, y)`` position.
    colors : str, matplotlib.colors.Colormap, sequence or dict, optional
        Node colours: a palette name (any seaborn palette or matplotlib colormap name, e.g.
        ``"Set2"`` or ``"viridis"``), a matplotlib colormap, a sequence of colours indexed by
        strategy index, or a dict keyed by strategy index or label. Palettes and sequences are
        indexed by strategy, so colours stay the same when strategies are filtered out.
        Defaults to seaborn's ``"colorblind"`` palette.
    node_size : float, optional
        Node area in points^2 (the largest node, if ``scale_nodes_by_frequency``). Set by ``style``.
    min_node_size : float, optional
        Smallest node area in points^2 when ``scale_nodes_by_frequency`` is set; chosen by the
        styles so that a percentage still fits inside the node.
    scale_nodes_by_frequency : bool, optional
        Make node areas grow linearly with the stationary frequency, from ``min_node_size`` to
        ``node_size``. Ignored if the graph has no stationary distribution. Set by ``style``.
    edge_width : float, default=1.5
        Line width of the edges.
    scale_edges_by_ratio : bool, optional
        Set by ``style``. Scale the width of selection edges by :math:`1 + \\log_{10}(\\rho/\\rho_N)`.
    min_ratio : float, default=1
        Hide selection edges with :math:`\\rho/\\rho_N` below this value (declutters large diagrams).
    curvature : float, default=0.2
        Curvature of the two arrows of a pair that invade each other.
    show_ratios, show_frequencies : bool, default=True
        Whether to write :math:`\\rho/\\rho_N` on selection edges and the stationary frequency next
        to each node.
    ratio_format : str
        Format string for :math:`\\rho/\\rho_N`.
    frequency_format : str or callable, optional
        Format string or function for the stationary frequencies, or ``"percent"`` for whole
        percentages (``"<1%"`` for frequencies that are positive but round to 0%). Set by ``style``.
    ratio_label_position : float, default=0.35
        Where to write :math:`\\rho/\\rho_N` along each edge, as a fraction of the way from the
        resident to the invader. Values other than 0.5 keep the labels of edges that cross at
        their midpoints (e.g. diameters of a circular layout) apart.
    name_position, frequency_position : {"outside", "inside"}, optional
        Where to write each strategy's name and stationary frequency (set by ``style``). Outside labels go on the side of the node facing away from the
        centre of the diagram, the name above the frequency when both are outside. Text inside a
        node is black or white, whichever contrasts with the node colour; when both are inside,
        the name is written above the frequency.
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
    preset = _invasion_style(style)
    node_size = preset["node_size"] if node_size is None else node_size
    min_node_size = preset["min_node_size"] if min_node_size is None else min_node_size
    scale_nodes_by_frequency = (preset["scale_nodes_by_frequency"] if scale_nodes_by_frequency is None
                                else scale_nodes_by_frequency)
    scale_edges_by_ratio = preset["scale_edges_by_ratio"] if scale_edges_by_ratio is None else scale_edges_by_ratio
    name_position = preset["name_position"] if name_position is None else name_position
    frequency_position = preset["frequency_position"] if frequency_position is None else frequency_position
    frequency_format = preset["frequency_format"] if frequency_format is None else frequency_format
    label_kwargs = {**preset.get("label_kwargs", {}), **(label_kwargs or {})}
    frequency_label_kwargs = {**preset.get("frequency_label_kwargs", {}), **(frequency_label_kwargs or {})}

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
    if (scale_nodes_by_frequency and nodes and np.any(np.isfinite(frequencies))
            and np.nanmax(frequencies) > 0):
        # Area grows linearly with the stationary frequency, from min_node_size to node_size.
        largest = np.nanmax(frequencies)
        sizes = [min_node_size + (node_size - min_node_size) * (f / largest) if np.isfinite(f) else node_size
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

    for position, option in ((name_position, "name_position"), (frequency_position, "frequency_position")):
        if position not in ("outside", "inside"):
            raise ValueError(f"{option} must be 'outside' or 'inside'")
    if frequency_format == "percent":
        format_frequency = _format_percentage
    elif callable(frequency_format):
        format_frequency = frequency_format
    else:
        format_frequency = frequency_format.format

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
        frequency_text = format_frequency(frequency) if show_frequencies and np.isfinite(frequency) else None

        # Direction away from the centre of the diagram; outside labels go on that side of the node.
        direction = np.asarray(pos[i], dtype=np.float64) - centre
        norm = np.linalg.norm(direction)
        direction = direction / norm if norm > 1e-12 else np.array([0., -1.])
        # Node sizes are areas in points^2, so the radius is sqrt(size) / 2 points.
        offset = (np.sqrt(size) / 2 + label_padding) * direction
        if label_offsets is not None:
            custom = label_offsets.get(i, label_offsets.get(name))
            if custom is not None:
                offset = np.asarray(custom, dtype=np.float64)
        length = np.linalg.norm(offset)
        ha = "left" if offset[0] > 0.3 * length else "right" if offset[0] < -0.3 * length else "center"
        above, below = offset[1] > 0.3 * length, offset[1] < -0.3 * length

        def outside(text, dy, va, style):
            ax.annotate(text, xy=(x, y), xytext=(offset[0], offset[1] + dy), textcoords="offset points",
                        ha=ha, va=va, zorder=5, **style)

        def inside(text, dy, style, user_kwargs):
            # Contrasting colour and no outline, unless the user set them explicitly.
            user_kwargs = user_kwargs or {}
            ax.annotate(text, xy=(x, y), xytext=(0., dy), textcoords="offset points", ha="center", va="center",
                        zorder=5, **{**style,
                                     "color": user_kwargs.get("color", _contrasting_text_color(node_color)),
                                     "path_effects": user_kwargs.get("path_effects", [])})

        labels_here = ((name, name_style, label_kwargs, name_position),
                       (frequency_text, frequency_style, frequency_label_kwargs, frequency_position))
        outside_texts = [(text, style) for text, style, _, where in labels_here
                         if text is not None and where == "outside"]
        inside_texts = [(text, style, user) for text, style, user, where in labels_here
                        if text is not None and where == "inside"]

        # Inside: one line centred, or two lines (name above frequency) around the centre.
        spacing = 0.8 * line_height
        for k, (text, style, user) in enumerate(inside_texts):
            inside(text, (len(inside_texts) - 1) / 2 * spacing - k * spacing, style, user)

        # Outside: one line, or a two-line block (name above frequency) on the outer side.
        if len(outside_texts) == 1:
            text, style = outside_texts[0]
            outside(text, 0., "bottom" if above else "top" if below else "center", style)
        elif len(outside_texts) == 2:
            (first, first_style), (second, second_style) = outside_texts
            if above:
                outside(second, 0., "bottom", second_style)
                outside(first, line_height, "bottom", first_style)
            elif below:
                outside(first, 0., "top", first_style)
                outside(second, -line_height, "top", second_style)
            else:
                outside(first, 0., "bottom", first_style)
                outside(second, 0., "top", second_style)

    ax.set_aspect("equal")
    ax.margins(0.2)
    ax.set_axis_off()
    return ax

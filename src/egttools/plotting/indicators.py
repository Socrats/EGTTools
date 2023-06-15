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
import matplotlib.pyplot as plt

from matplotlib.ticker import AutoMinorLocator
from typing import List, Optional, Tuple, Union


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


def draw_invasion_diagram(strategies: List[str], drift: float, fixation_probabilities: np.ndarray,
                          stationary_distribution: np.ndarray,
                          atol: float = 1e-4,
                          max_displayed_label_letters: Optional[int] = 4,
                          min_strategy_frequency: Optional[float] = -1,
                          node_size: Optional[int] = 4000,
                          font_size_node_labels: Optional[int] = 18,
                          font_size_edge_labels: Optional[int] = 14,
                          font_size_sd_labels: Optional[int] = 12,
                          display_node_labels: Optional[bool] = True,
                          display_edge_labels: Optional[bool] = True,
                          display_sd_labels: Optional[bool] = True,
                          node_labels_top_separation: Optional[float] = 0.15,
                          node_labels_bottom_separation: Optional[float] = - 0.2,
                          edge_width: Optional[int] = 2,
                          node_linewidth: Optional[float] = 0,
                          node_edgecolors: Optional[str] = None,
                          figsize: Optional[Tuple[int, int]] = (10, 10),
                          dpi: Optional[int] = 150,
                          colors: Optional[List[str]] = None,
                          ax: Optional[plt.axis] = None) -> nx.Graph:
    """
    Draws the markov chain for a given stationary distribution of monomorphic states.

    Parameters
    ----------
    strategies : List[str]
        Strategies and array of string labels for each strategy present in the population.
    drift : float
        drift = 1/pop_size
    fixation_probabilities : numpy.ndarray[float, 2]
        A matrix specifying the fixation probabilities.
    stationary_distribution : numpy.ndarray[float, 1]
        An array containing the stationary distribution (probability of each state in the system).
    atol : float
        The tolerance for considering a value equal to 1 (to detect wheter there is random drift). Default is 1e-4.
    max_displayed_label_letters : int
        Maximum number of letters of the strategy labels contained in the `strategies` List to
        be displayed.
    min_strategy_frequency: Optional[float]
        Minimum frequency of a strategy (its probability given by the stationary distribution)
        to be shown in the Graph.
    font_size_node_labels : Optional[int]
        Font size of the labels displayed inside each node.
    font_size_edge_labels : Optional[int]
        Font size of the labels displayed in each edge (which contain the fixation probabilities).
    font_size_sd_labels : Optional[int]
        Font size of the labels displayed beside each node containing the value of the stationary distribution.
    display_node_labels : Optional[bool]
        Indicates wether the node labels should be displayed.
    display_edge_labels : Optional[bool]
        Indicates wether the edge labels should be displayed.
    display_sd_labels : Optional[bool]
        Indicates whether the stationary distribution labels should be displayed.
    node_labels_top_separation : Optional[float]
        Gives the separation of node label for nodes with positive y (y > 0)
    node_labels_bottom_separation : Optional[float]
        Gives the separation of node label for nodes with negative y (y <= 0)
    edge_width : Optional[int]
        Width of the edge line.
    node_linewidth: Optional[float]
        Line width of node border
    node_edgecolors: Optional[str]
        Colors of node borders
    figsize : Optional[Tuple[int, int]]
        Size of the default figure (Only used if ax is not specified).
    dpi : Optional[int]
        Pixel density of the default plot
    node_size : Optional[int]
        Size of the nodes of the Graph to be plotted
    colors : Optional[List[str]]
        A list with the colors used to plot the nodes of the graph.
    ax : Optional[plt.axis]
        Axis on which to draw the graph.

    Returns
    -------
    networkx.Graph
        The graph depicting the Markov chain which represents the invasion dynamics.

    Notes
    -----
    If there are too many strategies, this function may not only take a lot of time to generate the Graph, but
    it will also not be easy to visualize. Also, you should only use this function when ploting the invasion
    diagram assuming the small mutation limit of the replication dynamics (SML).

    See Also
    --------
    plot_gradient

    Examples
    -------
    >>> import egttools as egt
    >>> import matplotlib.pyplot as plt
    >>> strategies = [egt.behaviors.NormalForm.TwoActions.Cooperator(), egt.behaviors.NormalForm.TwoActions.Defector(),
    ...               egt.behaviors.NormalForm.TwoActions.TFT(), egt.behaviors.NormalForm.TwoActions.Pavlov(),
    ...               egt.behaviors.NormalForm.TwoActions.Random(), egt.behaviors.NormalForm.TwoActions.GRIM()]
    >>> strategy_labels = [strategy.type().replace("NFGStrategies::", '') for strategy in strategies]
    >>> T=4; R=2; P=1; S=0; Z= 100; beta=1
    >>> A = np.array([
    ...     [P, T],
    ...     [S, R]
    ... ])
    >>> game = egt.games.NormalFormGame(100, A, strategies)
    >>> evolver = egt.analytical.StochDynamics(len(strategies), game.expected_payoffs(), Z)
    >>> sd = evolver.calculate_stationary_distribution(beta)
    >>> transitions, fixation_probabilities = evolver.transition_and_fixation_matrix(beta)
    >>> fig, ax = plt.subplots(figsize=(5, 5), dpi=150)
    >>> G = egt.plotting.draw_invasion_diagram(strategy_labels, 1/Z, fixation_probabilities, sd,
    ...     node_size=2000, min_strategy_frequency=0.00001, ax=ax)
    >>> plt.axis('off')
    >>> plt.show() # display
    """
    fixation_probabilities_normalized = fixation_probabilities / drift

    used_strategies = [strategy for i, strategy in enumerate(strategies) if
                       (stationary_distribution[i] > min_strategy_frequency)]

    used_strategies_idx = np.where(stationary_distribution > min_strategy_frequency)[0]

    q = len(used_strategies)

    G = nx.DiGraph()
    G.add_nodes_from(used_strategies)
    if colors is None:
        from seaborn import color_palette
        ncolors = color_palette("colorblind", q)
    else:
        ncolors = colors

    if ax is None:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    for j in range(q):
        for i in range(q):
            if fixation_probabilities_normalized[used_strategies_idx[i], used_strategies_idx[j]] >= 1 - atol:
                G.add_edge(used_strategies[i], used_strategies[j],
                           weight=fixation_probabilities_normalized[used_strategies_idx[i], used_strategies_idx[j]])

    eselect = [(u, v) for (u, v, d) in G.edges(data=True) if d['weight'] > 1 + atol]
    eselect_labels = dict(((u, v), r"{0:.2f}$\rho_N$".format(d['weight']))
                          for (u, v, d) in G.edges(data=True) if d['weight'] > 1 + atol)
    edrift = [(u, v) for (u, v, d) in G.edges(data=True) if np.isclose(d['weight'], 1, atol=atol)]

    pos = nx.circular_layout(G)  # positions for all nodes

    # nodes
    nx.draw_networkx_nodes(G,
                           pos,
                           node_size=node_size,
                           node_color=ncolors,
                           linewidths=node_linewidth,
                           edgecolors=ncolors if node_edgecolors is None else node_edgecolors,
                           ax=ax)

    # edges
    nx.draw_networkx_edges(G,
                           pos,
                           node_size=node_size,
                           edgelist=eselect,
                           width=edge_width,
                           arrows=True,
                           arrowstyle='-|>', ax=ax)
    nx.draw_networkx_edges(G,
                           pos,
                           node_size=node_size,
                           edgelist=edrift,
                           width=edge_width,
                           alpha=0.5,
                           style='dashed',
                           arrows=False, ax=ax)

    # node labels
    if display_node_labels:
        nx.draw_networkx_labels(G, pos, {strat: strat[:max_displayed_label_letters] for strat in used_strategies},
                                font_size=font_size_node_labels, font_weight='bold', font_color='black', ax=ax)

    # edge labels
    if display_edge_labels:
        nx.draw_networkx_edge_labels(G, pos, edge_labels=eselect_labels, font_size=font_size_edge_labels, ax=ax)

    for i, (key, value) in enumerate(pos.items()):
        x, y = value
        if y > 0:
            value = node_labels_top_separation
        else:
            value = node_labels_bottom_separation
        if display_sd_labels:
            ax.text(x, y + value, s="{0:.2f}".format(stationary_distribution[used_strategies_idx[i]]),
                    horizontalalignment='center', fontsize=font_size_sd_labels)

    return G


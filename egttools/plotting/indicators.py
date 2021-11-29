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
import numpy as np
import matplotlib.pyplot as plt

from typing import List, Optional, Tuple


def plot_gradient(x, gradients, saddle_points, saddle_type, gradient_direction, fig_title='', xlabel='', figsize=(5, 4),
                  **kwargs):
    """
    Creates a figure plotting the gradient of selection toghether with the saddle points,
    and the gradient arrows.
    :param x: vector containing the possible states in x axis. It must have the same length as gradient
    :param gradients: vector containing the gradient for each possible state
    :param saddle_points: vector containing all saddle points
    :param saddle_type: vector of booleans indicating whether or not the saddle point is stable
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


def draw_stationary_distribution(strategies: List[str], drift: float, fixation_probabilities: np.ndarray,
                                 stationary_distribution: np.ndarray,
                                 max_displayed_label_letters: Optional[int] = 4,
                                 min_strategy_frequency: Optional[float] = -1,
                                 node_size: Optional[int] = 4000,
                                 font_size_node_labels: Optional[int] = 18,
                                 font_size_edge_labels: Optional[int] = 14,
                                 font_size_sd_labels: Optional[int] = 12,
                                 edge_width: Optional[int] = 2,
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
    edge_width : Optional[int]
        Width of the edge line.
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
    >>> import matplotlib.pyplot as plt
    >>> import egttools as egt
    >>> strategies = [egt.behaviors.Cooperator(), egt.behaviors.Defector(), egt.behaviors.TFT(),
    ...               egt.behaviors.Pavlov(), egt.behaviors.Random(), egt.behaviors.GRIM()]
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
    >>> G = egt.plotting.draw_stationary_distribution(strategy_labels, 1/Z, fixation_probabilities, sd,
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
            if fixation_probabilities_normalized[used_strategies_idx[i], used_strategies_idx[j]] >= 1:
                G.add_edge(used_strategies[i], used_strategies[j],
                           weight=fixation_probabilities_normalized[used_strategies_idx[i], used_strategies_idx[j]])

    eselect = [(u, v) for (u, v, d) in G.edges(data=True) if d['weight'] > 1.0]
    eselect_labels = dict(((u, v), r"{0:.2f}$\rho_N$".format(d['weight']))
                          for (u, v, d) in G.edges(data=True) if d['weight'] > 1.0)
    edrift = [(u, v) for (u, v, d) in G.edges(data=True) if d['weight'] == 1.0]

    pos = nx.circular_layout(G)  # positions for all nodes

    # nodes
    nx.draw_networkx_nodes(G,
                           pos,
                           node_size=node_size,
                           node_color=ncolors,
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
    nx.draw_networkx_labels(G, pos, {strat: strat[:max_displayed_label_letters] for strat in used_strategies},
                            font_size=font_size_node_labels, font_weight='bold', font_color='black', ax=ax)

    # edge labels
    nx.draw_networkx_edge_labels(G, pos, edge_labels=eselect_labels, font_size=font_size_edge_labels, ax=ax)

    for i, (key, value) in enumerate(pos.items()):
        x, y = value
        if y > 0:
            value = 0.15
        else:
            value = - 0.2
        ax.text(x, y + value, s="{0:.2f}".format(stationary_distribution[used_strategies_idx[i]]),
                horizontalalignment='center', fontsize=font_size_sd_labels)

    return G

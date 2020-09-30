import numpy as np
import matplotlib.pyplot as plt


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


def draw_stationary_distribution(strats, drift, f, stationary, colors=None, ax=None):
    """
    Draws the markov chain for a given stationary distribution of monomorfic states
    :param strats and array of string labels for each strategy present in the population
    :param drift double 1/Z
    :param f a matrix specifying the fixation probabilities
    :param stationary numpy.array with the strationary distribution
    :param colors a list with the colors used to plot the nodes of the graph
    :param ax matplotlib.pyplot.axis to draw on the specified axis.
    """
    import networkx as nx
    q = len(strats)

    G = nx.DiGraph()
    G.add_nodes_from(strats)
    if colors is None:
        from seaborn import color_palette
        ncolors = color_palette("colorblind", q)
    else:
        ncolors = colors

    if ax is None:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 10), dpi=150)

    for j in range(q):
        for i in range(q):
            if f[i, j] >= drift:
                G.add_edge(strats[i], strats[j], weight=f[i, j])

    eselect = [(u, v) for (u, v, d) in G.edges(data=True) if d['weight'] > 1.0]
    eselect_labels = dict(((u, v), float("{0:.2f}".format(d['weight'])))
                          for (u, v, d) in G.edges(data=True) if d['weight'] > 1.0)
    edrift = [(u, v) for (u, v, d) in G.edges(data=True) if d['weight'] == 1.0]

    pos = nx.circular_layout(G)  # positions for all nodes

    # nodes
    nx.draw_networkx_nodes(G,
                           pos,
                           node_size=4000,
                           node_color=ncolors,
                           with_labels=True, ax=ax)

    # edges
    nx.draw_networkx_edges(G,
                           pos,
                           node_size=4000,
                           edgelist=eselect,
                           edge_labels=eselect_labels,
                           width=2,
                           with_labels=True,
                           arrows=True,
                           arrowstyle='-|>', ax=ax)
    nx.draw_networkx_edges(G,
                           pos,
                           node_size=4000,
                           edgelist=edrift,
                           width=2,
                           alpha=0.5,
                           style='dashed',
                           arrows=False, ax=ax)

    # node labels
    nx.draw_networkx_labels(G, pos, {strat: strat[:4] for strat in strats},
                            font_size=18, font_weight='bold', font_color='black', ax=ax)

    # edge labels
    nx.draw_networkx_edge_labels(G, pos, edge_labels=eselect_labels, font_size=14, ax=ax)

    for i, (key, value) in enumerate(pos.items()):
        x, y = value
        if y > 0:
            value = 0.15
        else:
            value = - 0.2
        ax.text(x, y + value, s="{0:.2f}".format(stationary[i]), horizontalalignment='center', fontsize=12)

    return G

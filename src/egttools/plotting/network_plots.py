"""
Network visualisation utilities for EGTtools.

Provides six plot types commonly used when analysing evolutionary games on
structured populations:

1. :func:`plot_network_state`         — snapshot of the network, nodes coloured by strategy
2. :func:`plot_strategy_evolution`    — time-series of strategy frequencies
3. :func:`animate_network_evolution`  — GIF/MP4 animation of the evolving network
4. :func:`plot_edge_homophily`        — homophily (fraction same-strategy edges) over time
5. :func:`plot_parameter_sweep`       — cooperation level vs two parameters (heatmap)
6. :func:`plot_strategy_by_degree`    — dominant strategy per degree bin (scale-free nets)
"""

from __future__ import annotations

import warnings
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np

__all__ = [
    "plot_network_state",
    "plot_strategy_evolution",
    "animate_network_evolution",
    "plot_edge_homophily",
    "plot_parameter_sweep",
    "plot_strategy_by_degree",
]

# Default colour cycle — maximally distinguishable for up to 8 strategies.
_DEFAULT_COLORS = [
    "#4C72B0",  # blue
    "#DD8452",  # orange
    "#55A868",  # green
    "#C44E52",  # red
    "#8172B3",  # purple
    "#937860",  # brown
    "#DA8BC3",  # pink
    "#8C8C8C",  # grey
]


def _strategy_colors(nb_strategies: int, strategy_colors: Optional[List] = None) -> list:
    if strategy_colors is not None:
        return list(strategy_colors)
    colors = _DEFAULT_COLORS * (nb_strategies // len(_DEFAULT_COLORS) + 1)
    return colors[:nb_strategies]


# ---------------------------------------------------------------------------
# 1. Network state snapshot
# ---------------------------------------------------------------------------

def plot_network_state(
    G,
    population: Sequence[int],
    strategy_names: Optional[Sequence[str]] = None,
    strategy_colors: Optional[Sequence] = None,
    ax: Optional[plt.Axes] = None,
    layout: Optional[Dict[int, Tuple[float, float]]] = None,
    node_size: int = 80,
    edge_color: str = "#aaaaaa",
    edge_alpha: float = 0.5,
    legend: bool = True,
) -> plt.Axes:
    """
    Draw a network snapshot with nodes coloured by their current strategy.

    Parameters
    ----------
    G : networkx.Graph (or any object with an ``.adjacency()`` method)
        The network topology.
    population : sequence of int
        Strategy index of each node (same order as ``list(G.nodes())``).
    strategy_names : sequence of str, optional
        Labels for each strategy.  Defaults to ``["Strategy 0", ...]``.
    strategy_colors : sequence, optional
        Colours for each strategy.
    ax : matplotlib.axes.Axes, optional
        Existing axes to draw on.
    layout : dict, optional
        ``{node: (x, y)}`` positions.  Computed via NetworkX spring layout if omitted.
    node_size : int
    edge_color : str
    edge_alpha : float
    legend : bool
        Whether to draw a strategy legend.

    Returns
    -------
    matplotlib.axes.Axes
    """
    import networkx as nx

    population = list(population)
    nb_strategies = max(population) + 1
    colors = _strategy_colors(nb_strategies, strategy_colors)
    names = list(strategy_names) if strategy_names else [f"Strategy {s}" for s in range(nb_strategies)]

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))

    if layout is None:
        layout = nx.spring_layout(G, seed=42)

    nodes = list(G.nodes())
    node_colors = [colors[population[i]] for i in range(len(nodes))]

    nx.draw_networkx_edges(G, layout, ax=ax, edge_color=edge_color, alpha=edge_alpha, width=0.8)
    nx.draw_networkx_nodes(G, layout, ax=ax,
                           node_color=node_colors, node_size=node_size, linewidths=0.5,
                           edgecolors="white")

    if legend:
        present = sorted(set(population))
        handles = [
            plt.Line2D([0], [0], marker="o", color="w",
                       markerfacecolor=colors[s], markersize=9, label=names[s])
            for s in present
        ]
        ax.legend(handles=handles, loc="best", fontsize=8, framealpha=0.7)

    ax.set_axis_off()
    return ax


# ---------------------------------------------------------------------------
# 2. Strategy frequency evolution
# ---------------------------------------------------------------------------

def plot_strategy_evolution(
    trajectory: np.ndarray,
    strategy_names: Optional[Sequence[str]] = None,
    strategy_colors: Optional[Sequence] = None,
    ax: Optional[plt.Axes] = None,
    xlabel: str = "Generation",
    ylabel: str = "Strategy frequency",
    title: str = "",
) -> plt.Axes:
    """
    Plot time-series of strategy frequencies.

    Parameters
    ----------
    trajectory : numpy.ndarray, shape (T, nb_strategies)
        Output from ``NetworkMCEstimator.run()``; rows are generations, columns are
        strategy counts.  Counts are normalised internally to frequencies in [0, 1].
    strategy_names : sequence of str, optional
    strategy_colors : sequence, optional
    ax : matplotlib.axes.Axes, optional

    Returns
    -------
    matplotlib.axes.Axes
    """
    trajectory = np.asarray(trajectory, dtype=float)
    T, nb_strategies = trajectory.shape
    row_sums = trajectory.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    freqs = trajectory / row_sums

    colors = _strategy_colors(nb_strategies, strategy_colors)
    names = list(strategy_names) if strategy_names else [f"Strategy {s}" for s in range(nb_strategies)]

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 3))

    t = np.arange(T)
    for s in range(nb_strategies):
        ax.plot(t, freqs[:, s], color=colors[s], label=names[s], linewidth=1.2)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.set_xlim(0, T - 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="best", fontsize=8, framealpha=0.7)
    return ax


# ---------------------------------------------------------------------------
# 3. Network evolution animation
# ---------------------------------------------------------------------------

def animate_network_evolution(
    G,
    snapshot_matrix: np.ndarray,
    strategy_names: Optional[Sequence[str]] = None,
    strategy_colors: Optional[Sequence] = None,
    layout: Optional[Dict] = None,
    interval_ms: int = 200,
    node_size: int = 80,
    save_path: Optional[str] = None,
    fps: int = 5,
) -> "matplotlib.animation.FuncAnimation":
    """
    Animate the evolution of a network, colouring nodes by strategy.

    Parameters
    ----------
    G : networkx.Graph
    snapshot_matrix : numpy.ndarray, shape (nb_snapshots, N)
        Each row is a per-node strategy vector at one snapshot.
        Produced by ``NetworkMCEstimator.run_snapshots(return_full=True)`` or by
        accumulating the per-snapshot callback into a 2-D array.
    strategy_names : sequence of str, optional
    strategy_colors : sequence, optional
    layout : dict, optional
        ``{node: (x, y)}``.  Computed once via spring layout if omitted.
    interval_ms : int
        Milliseconds between frames.
    node_size : int
    save_path : str, optional
        If given, save the animation to this path.  Extension determines format:
        ``.gif`` uses Pillow; ``.mp4`` uses ffmpeg.
    fps : int
        Frames per second (only used when saving).

    Returns
    -------
    matplotlib.animation.FuncAnimation
    """
    import networkx as nx
    from matplotlib.animation import FuncAnimation

    snapshots = np.asarray(snapshot_matrix, dtype=int)
    nb_snapshots, N = snapshots.shape
    nb_strategies = int(snapshots.max()) + 1
    colors = _strategy_colors(nb_strategies, strategy_colors)
    names = list(strategy_names) if strategy_names else [f"Strategy {s}" for s in range(nb_strategies)]

    if layout is None:
        layout = nx.spring_layout(G, seed=42)

    pos_array = np.array([layout[n] for n in G.nodes()])

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_axis_off()

    nx.draw_networkx_edges(G, layout, ax=ax, edge_color="#aaaaaa", alpha=0.4, width=0.8)
    scatter = ax.scatter(pos_array[:, 0], pos_array[:, 1],
                         c=[colors[snapshots[0, i]] for i in range(N)],
                         s=node_size, zorder=5, linewidths=0.5, edgecolors="white")
    title_obj = ax.set_title("t = 0", fontsize=10)

    present = sorted(set(snapshots.flatten()))
    legend_handles = [
        plt.Line2D([0], [0], marker="o", color="w",
                   markerfacecolor=colors[s], markersize=9, label=names[s])
        for s in present
    ]
    ax.legend(handles=legend_handles, loc="best", fontsize=8, framealpha=0.7)

    def update(frame: int):
        node_colors = [colors[snapshots[frame, i]] for i in range(N)]
        scatter.set_facecolor(node_colors)
        title_obj.set_text(f"t = {frame}")
        return scatter, title_obj

    anim = FuncAnimation(fig, update, frames=nb_snapshots,
                         interval=interval_ms, blit=True)

    if save_path is not None:
        if save_path.endswith(".gif"):
            try:
                anim.save(save_path, writer="pillow", fps=fps)
            except Exception as e:
                warnings.warn(f"Could not save GIF (Pillow required): {e}")
        elif save_path.endswith(".mp4"):
            try:
                anim.save(save_path, writer="ffmpeg", fps=fps)
            except Exception as e:
                warnings.warn(f"Could not save MP4 (ffmpeg required): {e}")
        else:
            warnings.warn(f"Unknown extension for save_path '{save_path}'. Use .gif or .mp4.")

    return anim


# ---------------------------------------------------------------------------
# 4. Edge homophily evolution
# ---------------------------------------------------------------------------

def plot_edge_homophily(
    homophily: Sequence[float],
    ax: Optional[plt.Axes] = None,
    color: str = "#4C72B0",
    label: str = "Edge homophily",
    xlabel: str = "Generation",
    ylabel: str = "Fraction same-strategy edges",
    title: str = "",
) -> plt.Axes:
    """
    Plot the evolution of edge homophily over time.

    Useful for co-evolutionary networks where the topology adapts alongside
    strategy dynamics.

    Parameters
    ----------
    homophily : sequence of float
        Per-generation fraction of edges connecting same-strategy nodes.
        Typically computed from the topology callback of
        ``NetworkCoEvolutionary.run_snapshots()``.
    ax : matplotlib.axes.Axes, optional
    color : str
    label : str
    xlabel : str
    ylabel : str
    title : str

    Returns
    -------
    matplotlib.axes.Axes
    """
    h = np.asarray(homophily, dtype=float)
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 3))

    ax.plot(np.arange(len(h)), h, color=color, label=label, linewidth=1.2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 1)
    ax.set_xlim(0, max(len(h) - 1, 1))
    if title:
        ax.set_title(title)
    ax.legend(loc="best", fontsize=8, framealpha=0.7)
    return ax


# ---------------------------------------------------------------------------
# 5. Parameter sweep heatmap
# ---------------------------------------------------------------------------

def plot_parameter_sweep(
    cooperation_matrix: np.ndarray,
    x_values: Sequence[float],
    y_values: Sequence[float],
    xlabel: str = "Parameter X",
    ylabel: str = "Parameter Y",
    title: str = "Cooperation level",
    cmap: str = "RdBu_r",
    vmin: float = 0.0,
    vmax: float = 1.0,
    ax: Optional[plt.Axes] = None,
    colorbar: bool = True,
) -> plt.Axes:
    """
    Plot a 2-D parameter sweep as a heatmap.

    Parameters
    ----------
    cooperation_matrix : numpy.ndarray, shape (len(y_values), len(x_values))
        Mean cooperation level (or any scalar metric) for each parameter combination.
        Rows correspond to ``y_values``, columns to ``x_values``.
    x_values : sequence of float
        Values swept along the x-axis.
    y_values : sequence of float
        Values swept along the y-axis.
    xlabel : str
    ylabel : str
    title : str
    cmap : str
        Matplotlib colormap name.
    vmin, vmax : float
        Colour scale limits.
    ax : matplotlib.axes.Axes, optional
    colorbar : bool

    Returns
    -------
    matplotlib.axes.Axes

    Example
    -------
    >>> import numpy as np
    >>> betas = np.linspace(0, 5, 20)
    >>> rewire_probs = np.linspace(0, 1, 20)
    >>> coop = np.zeros((len(rewire_probs), len(betas)))
    >>> # ... fill coop via nested loops calling estimate_strategy_distribution ...
    >>> plot_parameter_sweep(coop, betas, rewire_probs,
    ...                      xlabel="beta", ylabel="rewiring probability")
    """
    mat = np.asarray(cooperation_matrix, dtype=float)
    x = list(x_values)
    y = list(y_values)

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))

    im = ax.imshow(mat, origin="lower", aspect="auto",
                   extent=[x[0], x[-1], y[0], y[-1]],
                   cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    if colorbar:
        plt.colorbar(im, ax=ax, label=title)

    return ax


# ---------------------------------------------------------------------------
# 6. Degree-stratified strategy distribution
# ---------------------------------------------------------------------------

def plot_strategy_by_degree(
    degrees: Sequence[int],
    population: Sequence[int],
    nb_strategies: int,
    strategy_names: Optional[Sequence[str]] = None,
    strategy_colors: Optional[Sequence] = None,
    ax: Optional[plt.Axes] = None,
    bins: Union[int, str] = "auto",
    xlabel: str = "Node degree",
    ylabel: str = "Strategy frequency",
    title: str = "Strategy distribution by degree",
) -> plt.Axes:
    """
    Plot the frequency of each strategy stratified by node degree.

    Useful for heterogeneous networks (e.g. Barabási–Albert scale-free graphs) to
    reveal whether hub nodes and peripheral nodes adopt different strategies.

    Parameters
    ----------
    degrees : sequence of int
        Degree of each node (length N).
    population : sequence of int
        Current strategy of each node (length N).
    nb_strategies : int
    strategy_names : sequence of str, optional
    strategy_colors : sequence, optional
    ax : matplotlib.axes.Axes, optional
    bins : int or str
        Number of degree bins, or ``'auto'`` to use unique degree values.
    xlabel : str
    ylabel : str
    title : str

    Returns
    -------
    matplotlib.axes.Axes
    """
    degrees = np.asarray(degrees, dtype=int)
    population = np.asarray(population, dtype=int)
    colors = _strategy_colors(nb_strategies, strategy_colors)
    names = list(strategy_names) if strategy_names else [f"Strategy {s}" for s in range(nb_strategies)]

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))

    if bins == "auto":
        unique_degrees = np.sort(np.unique(degrees))
        bin_centers = unique_degrees
        bin_freqs = np.zeros((len(unique_degrees), nb_strategies))
        for i, d in enumerate(unique_degrees):
            mask = degrees == d
            if mask.sum() == 0:
                continue
            for s in range(nb_strategies):
                bin_freqs[i, s] = (population[mask] == s).sum() / mask.sum()
    else:
        counts, bin_edges = np.histogram(degrees, bins=int(bins))
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        bin_freqs = np.zeros((len(bin_centers), nb_strategies))
        for i in range(len(bin_centers)):
            mask = (degrees >= bin_edges[i]) & (degrees < bin_edges[i + 1])
            if i == len(bin_centers) - 1:
                mask = (degrees >= bin_edges[i]) & (degrees <= bin_edges[i + 1])
            if mask.sum() == 0:
                continue
            for s in range(nb_strategies):
                bin_freqs[i, s] = (population[mask] == s).sum() / mask.sum()

    bottom = np.zeros(len(bin_centers))
    for s in range(nb_strategies):
        ax.bar(bin_centers, bin_freqs[:, s], bottom=bottom,
               color=colors[s], label=names[s], alpha=0.85,
               width=(bin_centers[1] - bin_centers[0]) * 0.8 if len(bin_centers) > 1 else 0.5)
        bottom += bin_freqs[:, s]

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim(0, 1)
    ax.legend(loc="best", fontsize=8, framealpha=0.7)
    return ax

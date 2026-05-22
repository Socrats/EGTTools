"""
Reproduce Figures 1, 2, and 3 from:

    Pinheiro, F.L., Pacheco, J.M., Santos, F.C. (2012)
    "From Local to Global Dilemmas in Social Networks"
    PLOS ONE 7(2): e32114

Paper parameters:
  N = 1000 nodes, <k> = 4, L = 10^5 elementary time-steps per simulation,
  Ω = 2×10^7 total simulations, 1000 independently generated networks,
  starting from every possible initial fraction j/N (j = 1, ..., N-1).

Figure 1 — Time-independent G^A(x) on homogeneous random networks (k=4)
            vs well-mixed, β=1, B=1.005 and B=1.015.
            Panel b: stationary distributions for both B values (smoothed KDE
            with transparent fill, matching paper's visual style).

Figure 2 — Time-dependent G^A(x, t) at t=5, 15, 25 on homogeneous network,
            (B=1.01, β=10), plus root evolution with example trajectories.

Figure 3 — G^A(x) on Barabási–Albert scale-free networks for B=1.15, 1.25,
            1.35, β=0.1, plus unstable-root evolution and trajectories.

Usage::

    python scripts/reproduce_pinheiro2012.py          # quick preview
    python scripts/reproduce_pinheiro2012.py \\
        --N 200 --nb_networks 5 --runs_per_j 3 \\
        --nb_generations 100 --output_dir results
"""

import argparse
import os
from typing import List, Tuple

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

try:
    import networkx as nx
except ImportError:
    raise SystemExit("networkx required: pip install networkx")

try:
    from scipy.ndimage import gaussian_filter1d
    from scipy.stats import gaussian_kde
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    import egttools
    from egttools.numerical.structure import NetworkMCEstimatorPC
except ImportError:
    raise SystemExit("egttools is not installed in the current environment.")

# Paper-like global style
rcParams.update({
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
})


# ---------------------------------------------------------------------------
# PD game — accumulated payoffs as in Pinheiro et al. (2012)
# ---------------------------------------------------------------------------

class PinheiroGame(egttools.games.AbstractSpatialGame):
    """
    Prisoner's Dilemma, Pinheiro 2012: T=B, R=1, S=1-B, P=0.
    Strategies: 0=Cooperate, 1=Defect.

    Fitness is the ACCUMULATED payoff (sum over neighbours) to match the
    Fermi imitation rule in the paper.
    state = [n_C, n_D] — neighbourhood composition.
    """

    def __init__(self, B: float):
        super().__init__()
        self.B = float(B)

    def calculate_fitness(self, strategy_index: int, state) -> float:
        state = np.asarray(state, dtype=int)
        n_c = int(state[0])
        n_d = int(state[1]) if len(state) > 1 else int(state.sum()) - n_c
        if strategy_index == 0:      # Cooperate: R*n_C + S*n_D
            return float(n_c) + (1.0 - self.B) * float(n_d)
        else:                        # Defect: T*n_C + P*n_D = B*n_C
            return self.B * float(n_c)

    def nb_strategies(self) -> int:
        return 2

    def toString(self) -> str:
        return f"PinheiroPD(B={self.B})"

    def type(self) -> str:
        return "PinheiroGame"


# ---------------------------------------------------------------------------
# Topology builders
# ---------------------------------------------------------------------------

def homogeneous_random_graph(N: int, k: int, seed: int) -> nx.Graph:
    return nx.random_regular_graph(k, N, seed=seed)


def ba_graph(N: int, m: int = 2, seed: int = 0) -> nx.Graph:
    return nx.barabasi_albert_graph(N, m, seed=seed)


def graph_to_topology(G: nx.Graph) -> dict:
    return {n: list(nbrs) for n, nbrs in G.adjacency()}


# ---------------------------------------------------------------------------
# Well-mixed AGoS (analytical, accumulated payoffs)
# ---------------------------------------------------------------------------

def well_mixed_agos(N: int, B: float, beta: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Exact G^WM(j) with accumulated payoffs in a well-mixed population.

    f_D - f_C = B*N - N + 1  (constant in j).
    G^WM(j) = j(N-j)/[N(N-1)] * [fermi(β,f_D,f_C) - fermi(β,f_C,f_D)]
    """
    j = np.arange(1, N)
    delta = B * N - N + 1
    f_DC = 1.0 / (1.0 + np.exp(beta * delta))
    f_CD = 1.0 / (1.0 + np.exp(-beta * delta))
    G = j * (N - j) / (N * (N - 1)) * (f_DC - f_CD)
    return j / N, G


# ---------------------------------------------------------------------------
# Smoothing helpers
# ---------------------------------------------------------------------------

def smooth_agos(j_vals: np.ndarray, G: np.ndarray,
                n_out: int = 300) -> Tuple[np.ndarray, np.ndarray]:
    """
    Smooth a noisy G^A(x) curve with a Gaussian kernel.
    gaussian_filter1d cannot introduce new extrema, unlike SG+spline.
    sigma is scaled to ~3% of data length so wider arrays are smoothed more.
    """
    mask = ~np.isnan(G) & (j_vals > 0) & (j_vals < 1)
    x, y = j_vals[mask], G[mask]
    if len(x) < 5:
        return x, y

    if HAS_SCIPY:
        try:
            sigma = max(1.0, len(y) * 0.03)
            y_smooth = gaussian_filter1d(y, sigma=sigma)
            x_out = np.linspace(x[0], x[-1], n_out)
            return x_out, np.interp(x_out, x, y_smooth)
        except Exception:
            pass
    return x, y


def stationary_kde(visit_counts: np.ndarray, N: int,
                   n_out: int = 300,
                   bandwidth: float = 0.06) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert integer visit counts (length N+1) to a smooth PDF via KDE.
    """
    samples = np.repeat(np.arange(N + 1) / N, visit_counts.astype(int))
    if len(samples) < 5:
        x = np.linspace(0, 1, n_out)
        return x, np.zeros(n_out)

    if HAS_SCIPY:
        try:
            kde = gaussian_kde(samples, bw_method=bandwidth)
            x = np.linspace(0, 1, n_out)
            return x, kde(x)
        except Exception:
            pass
    # Fallback: normalised bar heights
    counts = visit_counts.astype(float)
    total = counts.sum()
    return np.arange(N + 1) / N, counts / (total + 1e-12)


# ---------------------------------------------------------------------------
# Stationary distribution from long trajectories
# ---------------------------------------------------------------------------

def estimate_stationary(
    graph_builder,
    N: int,
    B: float,
    beta: float,
    mu: float,
    nb_networks: int,
    nb_generations: int,
    transitory: int,
    nb_runs_per_net: int,
    base_seed: int,
) -> np.ndarray:
    """
    Run multiple long trajectories and accumulate visit counts per j.
    Uses est.run() for pure-C++ efficiency (no Python callback per step).
    Returns normalised visit_counts array of length N+1.
    """
    visit_counts = np.zeros(N + 1)
    for net_idx in range(nb_networks):
        G = graph_builder(N, seed=base_seed + net_idx)
        est = NetworkMCEstimatorPC(PinheiroGame(B), graph_to_topology(G), 2, beta, mu)
        for run_idx in range(nb_runs_per_net):
            # Random start at different cooperation levels
            n_c0 = max(1, min(N - 1, round((run_idx + 0.5) / nb_runs_per_net * N)))
            init_state = np.array([n_c0, N - n_c0], dtype=np.uint64)
            traj = est.run(nb_generations, transitory, init_state)
            # traj is (nb_generations-transitory, 2), col 0 = cooperator count
            for counts in traj:
                visit_counts[int(counts[0])] += 1

    total = visit_counts.sum()
    return visit_counts / total if total > 0 else visit_counts


# ---------------------------------------------------------------------------
# Multi-network AGoS helpers
# ---------------------------------------------------------------------------

def average_agos_over_networks(
    graph_builder,
    N: int,
    B: float,
    beta: float,
    mu: float,
    nb_networks: int,
    nb_generations: int,
    transitory: int,
    runs_per_j: int,
    base_seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Average G^A(j) over multiple independent network realizations."""
    all_means = []
    for net_idx in range(nb_networks):
        G = graph_builder(N, seed=base_seed + net_idx)
        est = NetworkMCEstimatorPC(PinheiroGame(B), graph_to_topology(G), 2, beta, mu)
        mean_g, _ = est.estimate_agos(
            nb_runs=0,
            nb_generations=nb_generations,
            transitory=transitory,
            runs_per_j=runs_per_j,
        )
        all_means.append(mean_g[:, 0])

    stack = np.stack(all_means, axis=0)
    with np.errstate(invalid="ignore"):
        mean_G = np.nanmean(stack, axis=0)
        se_G   = np.nanstd(stack, axis=0) / max(1, np.sqrt(nb_networks))
    return np.arange(N + 1) / N, mean_G, se_G


def average_agos_td_over_networks(
    graph_builder,
    N: int,
    B: float,
    beta: float,
    mu: float,
    nb_networks: int,
    nb_generations: int,
    runs_per_j: int,
    base_seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Average time-dependent G^A(j,t) over multiple networks."""
    all_means = []
    for net_idx in range(nb_networks):
        G = graph_builder(N, seed=base_seed + net_idx)
        est = NetworkMCEstimatorPC(PinheiroGame(B), graph_to_topology(G), 2, beta, mu)
        nb_runs = runs_per_j * (N - 1)
        mg_t, _ = est.estimate_agos_time_dependent(nb_runs, nb_generations)
        all_means.append(mg_t[:, :, 0])

    stack = np.stack(all_means, axis=0)
    with np.errstate(invalid="ignore"):
        mean_G = np.nanmean(stack, axis=0)
        se_G   = np.nanstd(stack, axis=0) / max(1, np.sqrt(nb_networks))
    return mean_G, se_G


# ---------------------------------------------------------------------------
# Root-finding and trajectory helpers
# ---------------------------------------------------------------------------

def find_roots(j_vals: np.ndarray, G: np.ndarray) -> List[float]:
    roots = []
    mask = ~np.isnan(G)
    jv, Gv = j_vals[mask], G[mask]
    for i in range(len(Gv) - 1):
        if Gv[i] * Gv[i + 1] < 0:
            x = jv[i] - Gv[i] * (jv[i + 1] - jv[i]) / (Gv[i + 1] - Gv[i])
            roots.append(float(x))
    return roots


def trajectory_from_run(
    estimator: NetworkMCEstimatorPC,
    N: int,
    init_coop_frac: float,
    nb_generations: int,
) -> np.ndarray:
    n_c0 = round(init_coop_frac * N)
    init_state = np.array([n_c0, N - n_c0], dtype=np.uint64)
    traj: List[float] = []

    def cb(_t, pop):
        traj.append(sum(s == 0 for s in pop) / N)

    try:
        estimator.run_snapshots(nb_generations, 0, 1, init_state, cb)
    except Exception:
        pass
    return np.array(traj)


# ---------------------------------------------------------------------------
# Figure 1
# ---------------------------------------------------------------------------

def make_figure1(args, output_dir: str):
    N, k, beta = args.N, 4, 1.0
    B_vals = [1.005, 1.015]
    net_colors = ["#1f77b4", "#ff7f0e"]
    nb_gen = max(args.nb_generations, 50)
    trans  = nb_gen // 5

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # ── panel a: time-independent AGoS ──────────────────────────────────────
    for B, nc in zip(B_vals, net_colors):
        print(f"  Fig1 B={B} …", flush=True)
        j_vals, mean_g, _ = average_agos_over_networks(
            lambda N, seed, k=k: homogeneous_random_graph(N, k, seed),
            N, B, beta, args.mu,
            nb_networks=args.nb_networks,
            nb_generations=nb_gen,
            transitory=trans,
            runs_per_j=args.runs_per_j,
            base_seed=args.seed,
        )
        xs, ys = smooth_agos(j_vals, mean_g)
        axes[0].plot(xs, ys, color=nc, linewidth=2.2, label=fr"$B={B}$")

    axes[0].axhline(0, color="k", linewidth=0.8, linestyle="--", zorder=0)
    axes[0].set_xlabel(r"$x$", fontsize=11)
    axes[0].set_ylabel(r"$G^A(x)$", fontsize=11)
    axes[0].set_xlim(0, 1)
    axes[0].legend(fontsize=7, framealpha=0.7, edgecolor="none")
    axes[0].set_title(r"Homogeneous network  $(\beta=1,\,k=4)$", fontsize=10)

    # ── panel b: stationary distribution for both B values ──────────────────
    # Collect visit counts separately for each B
    stat_runs = max(3, args.runs_per_j * 2)
    stat_trans = max(nb_gen // 3, 10)
    for B, nc in zip(B_vals, net_colors):
        print(f"  Fig1 stat dist B={B} …", flush=True)
        vc = estimate_stationary(
            lambda N, seed, k=k: homogeneous_random_graph(N, k, seed),
            N, B, beta, args.mu,
            nb_networks=args.nb_networks,
            nb_generations=nb_gen,
            transitory=stat_trans,
            nb_runs_per_net=stat_runs,
            base_seed=args.seed + 100,
        )
        x_kde, y_kde = stationary_kde(
            (vc * 5000).astype(int), N, bandwidth=0.07
        )
        dx = x_kde[1] - x_kde[0]
        y_kde = y_kde / (y_kde.sum() * dx + 1e-12)
        axes[1].fill_between(x_kde, y_kde, alpha=0.25, color=nc)
        axes[1].plot(x_kde, y_kde, color=nc, linewidth=2.2, label=fr"$B={B}$")

    axes[1].set_xlabel(r"$x$", fontsize=11)
    axes[1].set_ylabel(r"$P_s(x)$", fontsize=11)
    axes[1].set_xlim(0, 1)
    axes[1].legend(fontsize=8, framealpha=0.7, edgecolor="none")
    axes[1].set_title(r"Stationary distribution  $(\beta=1,\,k=4)$", fontsize=10)

    plt.tight_layout()
    path = f"{output_dir}/figure1.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → saved {path}")


# ---------------------------------------------------------------------------
# Figure 2
# ---------------------------------------------------------------------------

def make_figure2(args, output_dir: str):
    N, k, beta, B = args.N, 4, 10.0, 1.01
    nb_gen  = max(args.nb_generations, 30)
    t_snap  = [4, 14, min(24, nb_gen - 1)]
    snap_colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    root_colors = {"xL": "#d62728", "xR": "#9467bd"}

    print(f"  Fig2 β={beta}, B={B} …", flush=True)
    mean_mg_t, _ = average_agos_td_over_networks(
        lambda N, seed, k=k: homogeneous_random_graph(N, k, seed),
        N, B, beta, args.mu,
        nb_networks=args.nb_networks,
        nb_generations=nb_gen,
        runs_per_j=args.runs_per_j,
        base_seed=args.seed + 200,
    )
    j_vals = np.arange(N + 1) / N

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # ── panel a: time-dependent G^A at three snapshots ──────────────────────
    for t_idx, col in zip(t_snap, snap_colors):
        if t_idx < mean_mg_t.shape[0]:
            G_raw = mean_mg_t[t_idx, :]
            xs, ys = smooth_agos(j_vals, G_raw)
            axes[0].plot(xs, ys, color=col, linewidth=2.0,
                         label=fr"$t={t_idx + 1}$")

    axes[0].axhline(0, color="k", linewidth=0.8, linestyle="--", zorder=0)
    axes[0].set_xlabel(r"$x$", fontsize=11)
    axes[0].set_ylabel(r"$G^A(x,\,t)$", fontsize=11)
    axes[0].set_xlim(0, 1)
    axes[0].legend(fontsize=8, framealpha=0.7, edgecolor="none")
    axes[0].set_title(fr"Time-dependent $G^A$ ($\beta={beta}$, $B={B}$)", fontsize=10)

    # ── panel b: root evolution + trajectories ───────────────────────────────
    # Roots plotted as circles: open for unstable (xL), filled for stable (xR)
    t_ticks = np.arange(1, nb_gen + 1)
    xL_vals, xR_vals = [], []
    for t_idx in range(nb_gen):
        G_t = mean_mg_t[t_idx, 1:N] if t_idx < mean_mg_t.shape[0] else np.full(N - 1, np.nan)
        roots = sorted(find_roots(j_vals[1:N], G_t))
        xL_vals.append(roots[0]  if roots          else np.nan)
        xR_vals.append(roots[-1] if len(roots) > 1 else np.nan)

    xL_arr = np.array(xL_vals)
    xR_arr = np.array(xR_vals)
    # Open circles for unstable xL
    axes[1].plot(t_ticks, xL_arr, "o", markerfacecolor="none",
                 markeredgecolor=root_colors["xL"], markeredgewidth=1.2,
                 markersize=5, label=r"$x_L$ (unstable)", zorder=3)
    # Filled circles for stable xR
    axes[1].plot(t_ticks, xR_arr, "o", markerfacecolor=root_colors["xR"],
                 markeredgecolor=root_colors["xR"], markeredgewidth=1.2,
                 markersize=5, label=r"$x_R$ (stable)", zorder=3)

    # Two independent trajectories starting from 50% C and 50% D
    G_net = homogeneous_random_graph(N, k, seed=args.seed + 200)
    est0 = NetworkMCEstimatorPC(PinheiroGame(B), graph_to_topology(G_net), 2, beta, args.mu)
    traj_styles = [
        ("#1f4e79", "-",  1.5, {}),      # dark blue solid line
        ("#7bbfea", None, 1.0, {"marker": "x", "markevery": max(1, nb_gen // 15),
                                 "markersize": 5, "markeredgewidth": 1.0}),  # light blue crosses
    ]
    for (color, ls, lw, kw), _ in zip(traj_styles, range(2)):
        traj = trajectory_from_run(est0, N, 0.5, nb_gen)
        if len(traj):
            axes[1].plot(np.arange(1, len(traj) + 1), traj,
                         color=color, linewidth=lw, linestyle=ls, alpha=0.85, **kw)

    axes[1].set_xlabel("Generation", fontsize=11)
    axes[1].set_ylabel(r"$x$", fontsize=11)
    axes[1].set_xlim(1, nb_gen)
    axes[1].set_ylim(0, 1)
    axes[1].legend(fontsize=8, framealpha=0.7, edgecolor="none")
    axes[1].set_title(fr"Root evolution ($\beta={beta}$, $B={B}$)", fontsize=10)

    plt.tight_layout()
    path = f"{output_dir}/figure2.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → saved {path}")


# ---------------------------------------------------------------------------
# Figure 3
# ---------------------------------------------------------------------------

def make_figure3(args, output_dir: str):
    N, m, beta = args.N, 2, 0.1
    B_vals = [1.15, 1.25, 1.35]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    nb_gen = max(args.nb_generations, 60)
    trans  = nb_gen // 5

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    j_vals = np.arange(N + 1) / N

    # ── panel a: time-independent G^A for three B values ────────────────────
    for B, col in zip(B_vals, colors):
        print(f"  Fig3 BA B={B} …", flush=True)
        _, mean_g, _ = average_agos_over_networks(
            lambda N, seed, m=m: ba_graph(N, m, seed),
            N, B, beta, args.mu,
            nb_networks=args.nb_networks,
            nb_generations=nb_gen,
            transitory=trans,
            runs_per_j=args.runs_per_j,
            base_seed=args.seed + 300,
        )
        xs, ys = smooth_agos(j_vals, mean_g)
        axes[0].plot(xs, ys, color=col, linewidth=2.2, label=fr"$B={B}$ (BA)")

    axes[0].axhline(0, color="k", linewidth=0.8, linestyle="--", zorder=0)
    axes[0].set_xlabel(r"$x$", fontsize=11)
    axes[0].set_ylabel(r"$G^A(x)$", fontsize=11)
    axes[0].set_xlim(0, 1)
    axes[0].legend(fontsize=8, framealpha=0.7, edgecolor="none")
    axes[0].set_title(fr"BA scale-free network  ($\beta={beta}$, $m=2$)", fontsize=10)

    # ── panel b: unstable root xL over time for B=1.15 ──────────────────────
    B_dyn = 1.15
    print(f"  Fig3 time-dep B={B_dyn} …", flush=True)
    mean_mg_t, _ = average_agos_td_over_networks(
        lambda N, seed, m=m: ba_graph(N, m, seed),
        N, B_dyn, beta, args.mu,
        nb_networks=args.nb_networks,
        nb_generations=nb_gen,
        runs_per_j=args.runs_per_j,
        base_seed=args.seed + 400,
    )

    # Open circles for unstable xL (paper style)
    t_ticks = np.arange(1, nb_gen + 1)
    xL_vals = []
    for t_idx in range(nb_gen):
        G_t = mean_mg_t[t_idx, 1:N] if t_idx < mean_mg_t.shape[0] else np.full(N - 1, np.nan)
        roots = find_roots(j_vals[1:N], G_t)
        xL_vals.append(roots[0] if roots else np.nan)

    axes[1].plot(t_ticks, np.array(xL_vals), "o", markerfacecolor="none",
                 markeredgecolor="#d62728", markeredgewidth=1.2,
                 markersize=5, label=fr"$x_L$ (unstable, $B={B_dyn}$)", zorder=3)

    # Two independent trajectories from 50% C/D: dark blue solid + light blue crosses
    G_ba0 = ba_graph(N, m, seed=args.seed + 400)
    est_dyn = NetworkMCEstimatorPC(PinheiroGame(B_dyn), graph_to_topology(G_ba0), 2, beta, args.mu)
    traj_styles = [
        ("#1f4e79", "-",  1.5, {}),
        ("#7bbfea", None, 1.0, {"marker": "x", "markevery": max(1, nb_gen // 15),
                                 "markersize": 5, "markeredgewidth": 1.0}),
    ]
    for (color, ls, lw, kw), _ in zip(traj_styles, range(2)):
        traj = trajectory_from_run(est_dyn, N, 0.5, nb_gen)
        if len(traj):
            axes[1].plot(np.arange(1, len(traj) + 1), traj,
                         color=color, linewidth=lw, linestyle=ls, alpha=0.85, **kw)

    axes[1].set_xlabel("Generation", fontsize=11)
    axes[1].set_ylabel(r"$x$", fontsize=11)
    axes[1].set_xlim(1, nb_gen)
    axes[1].set_ylim(0, 1)
    axes[1].legend(fontsize=8, framealpha=0.7, edgecolor="none")
    axes[1].set_title(fr"Unstable root evolution ($B={B_dyn}$, $\beta={beta}$)", fontsize=10)

    plt.tight_layout()
    path = f"{output_dir}/figure3.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → saved {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Reproduce Figures 1–3 of Pinheiro et al. (2012)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--N", type=int, default=100,
                        help="Network size (paper: 1000)")
    parser.add_argument("--nb_networks", type=int, default=3,
                        help="Independent network realizations (paper: 1000)")
    parser.add_argument("--runs_per_j", type=int, default=2,
                        help="Simulations per initial j per network (paper: ~20)")
    parser.add_argument("--nb_generations", type=int, default=50,
                        help="Generations per trajectory (paper: 100 for N=1000)")
    parser.add_argument("--mu", type=float, default=0.001,
                        help="Mutation probability")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default=".",
                        help="Directory for output PNG files")
    parser.add_argument("--figures", type=str, default="1,2,3",
                        help="Comma-separated figures to generate")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    figs = {int(f.strip()) for f in args.figures.split(",")}

    total_runs = args.runs_per_j * (args.N - 1) * args.nb_networks
    print(f"N={args.N}, nb_networks={args.nb_networks}, runs_per_j={args.runs_per_j}")
    print(f"nb_generations={args.nb_generations}, total_runs_per_panel≈{total_runs}")
    if not HAS_SCIPY:
        print("WARNING: scipy not found — curves will not be smoothed")

    if 1 in figs:
        print("Generating Figure 1 …")
        make_figure1(args, args.output_dir)

    if 2 in figs:
        print("Generating Figure 2 …")
        make_figure2(args, args.output_dir)

    if 3 in figs:
        print("Generating Figure 3 …")
        make_figure3(args, args.output_dir)

    print("Done.")


if __name__ == "__main__":
    main()

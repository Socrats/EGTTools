"""
Benchmark: numerically exact gradient vs MC estimator on the same network state.

Compares the gradient of selection computed by the exact formula (deterministic,
for a given per-node population assignment) against the stochastic MC estimator
for the same state, for increasing network sizes.

Usage::

    python scripts/network_gradient_benchmark.py [--N 50 100 200] [--nb_runs 500]

Output:
 - Console table: network size, mean |error|, exact_time, mc_time.
 - A saved figure ``gradient_benchmark.png`` with two subplots:
     Left:  gradient vectors side-by-side (exact vs MC) for the largest N.
     Right: relative L1 error vs network size.
"""

import argparse
import time
import warnings
from typing import List

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import networkx as nx
except ImportError:
    raise SystemExit("networkx is required: pip install networkx")

try:
    import egttools
    from egttools.numerical.structure import NetworkMCEstimatorPC
except ImportError:
    raise SystemExit("egttools is not installed in the current environment.")


# ---------------------------------------------------------------------------
# Minimal 2-strategy Prisoner's Dilemma spatial game
# ---------------------------------------------------------------------------

class PDGame(egttools.games.AbstractSpatialGame):
    """
    Prisoner's Dilemma with two strategies: Cooperate (0), Defect (1).

    Payoff to Cooperate in a neighbourhood of size d with k cooperators
    (excluding self):
        π_C = (R * k + S * (d - k)) / d        if d > 0, else 0

    Payoff to Defect:
        π_D = (T * k + P * (d - k)) / d        if d > 0, else P
    """

    def __init__(self, R: float = 3.0, S: float = 0.0,
                 T: float = 5.0, P: float = 1.0):
        super().__init__()
        self.R = R
        self.S = S
        self.T = T
        self.P = P

    def calculate_fitness(self, strategy_index: int, state) -> float:
        state = np.asarray(state, dtype=int)
        k = int(state[0])   # cooperators in neighbourhood
        d = int(state.sum())
        if d == 0:
            return self.R if strategy_index == 0 else self.P
        if strategy_index == 0:  # Cooperate
            return (self.R * k + self.S * (d - k)) / d
        else:                     # Defect
            return (self.T * k + self.P * (d - k)) / d

    def nb_strategies(self) -> int:
        return 2

    def toString(self) -> str:
        return f"PD(R={self.R}, S={self.S}, T={self.T}, P={self.P})"

    def type(self) -> str:
        return "PDGame"


# ---------------------------------------------------------------------------
# Benchmark helpers
# ---------------------------------------------------------------------------

def ring_graph(N: int) -> nx.Graph:
    G = nx.cycle_graph(N)
    return G


def random_population(N: int, nb_strategies: int, rng) -> List[int]:
    return list(rng.integers(0, nb_strategies, size=N))


def networkx_to_dict(G: nx.Graph):
    return {n: list(nbrs) for n, nbrs in G.adjacency()}


def run_benchmark(N: int, beta: float, mu: float, nb_runs: int,
                  nb_generations: int, rng) -> dict:
    G = ring_graph(N)
    topo = networkx_to_dict(G)
    game = PDGame()
    population = random_population(N, 2, rng)

    estimator = NetworkMCEstimatorPC(game, topo, 2, beta, mu)

    # ---- Numerically exact gradient -----------------------------------
    t0 = time.perf_counter()
    exact_grad = np.asarray(estimator.calculate_gradient_of_selection(population))
    exact_time = time.perf_counter() - t0

    # ---- MC gradient estimator ----------------------------------------
    # We use estimate_strategy_distribution starting from the given state
    # and measure the mean frequency change per generation (a proxy for
    # the gradient). For a direct gradient estimate we use multiple short runs.
    init_state = np.zeros(2, dtype=np.uint64)
    for s in population:
        init_state[s] += 1

    t0 = time.perf_counter()
    mean_freq, se_freq = estimator.estimate_strategy_distribution(
        nb_runs=nb_runs,
        nb_generations=nb_generations,
        transitory=max(1, nb_generations // 10),
    )
    mc_time = time.perf_counter() - t0

    # The MC estimate of the *gradient* (not stationary distribution)
    # from a single short trajectory starting from 'population':
    t0 = time.perf_counter()
    mc_grad_estimates = []
    for _ in range(nb_runs):
        traj = estimator.run(
            nb_generations=max(10, nb_generations // 10),
            transitory=0,
            init_state=init_state,
        )
        # Gradient ≈ mean Δfreq per generation
        traj_float = traj.astype(float) / N
        if len(traj_float) > 1:
            mc_grad_estimates.append(traj_float[-1] - traj_float[0])
    mc_grad_time = time.perf_counter() - t0

    mc_grad = np.mean(mc_grad_estimates, axis=0) if mc_grad_estimates else np.zeros(2)
    mc_grad_se = np.std(mc_grad_estimates, axis=0) / np.sqrt(len(mc_grad_estimates)) \
        if len(mc_grad_estimates) > 1 else np.zeros(2)

    l1_error = float(np.abs(exact_grad - mc_grad).sum())
    rel_error = l1_error / (np.abs(exact_grad).sum() + 1e-12)

    return {
        "N": N,
        "exact_grad": exact_grad,
        "mc_grad": mc_grad,
        "mc_grad_se": mc_grad_se,
        "l1_error": l1_error,
        "rel_error": rel_error,
        "exact_time": exact_time,
        "mc_grad_time": mc_grad_time,
        "mc_dist_time": mc_time,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Benchmark exact gradient vs MC estimator")
    parser.add_argument("--N", type=int, nargs="+", default=[20, 50, 100, 200, 500],
                        help="Network sizes to benchmark")
    parser.add_argument("--nb_runs", type=int, default=200,
                        help="MC runs per size")
    parser.add_argument("--nb_generations", type=int, default=50,
                        help="Generations per MC run (for gradient estimation)")
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--mu", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="gradient_benchmark.png")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    print(f"{'N':>8}  {'|exact grad|':>12}  {'|MC grad|':>10}  "
          f"{'rel_L1_err':>12}  {'exact_ms':>10}  {'MC_ms':>10}")
    print("-" * 70)

    results = []
    for N in args.N:
        r = run_benchmark(N, args.beta, args.mu, args.nb_runs, args.nb_generations, rng)
        results.append(r)
        print(f"{N:>8}  {np.abs(r['exact_grad']).sum():>12.5f}  "
              f"{np.abs(r['mc_grad']).sum():>10.5f}  "
              f"{r['rel_error']:>12.5f}  "
              f"{r['exact_time']*1000:>10.2f}  "
              f"{r['mc_grad_time']*1000:>10.2f}")

    # ---- Plot ----------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Left: gradient comparison for the largest N
    last = results[-1]
    N_last = last["N"]
    strategies = ["Cooperate", "Defect"]
    x = np.arange(2)
    w = 0.35
    axes[0].bar(x - w / 2, last["exact_grad"], w, label="Exact", color="#4C72B0", alpha=0.85)
    axes[0].bar(x + w / 2, last["mc_grad"], w, label="MC (mean ± SE)",
                color="#DD8452", alpha=0.85,
                yerr=last["mc_grad_se"], capsize=4)
    axes[0].axhline(0, color="k", linewidth=0.8, linestyle="--")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(strategies)
    axes[0].set_ylabel("Gradient of selection")
    axes[0].set_title(f"Gradient comparison (N={N_last}, β={args.beta})")
    axes[0].legend()

    # Right: relative L1 error vs N
    Ns = [r["N"] for r in results]
    rel_errs = [r["rel_error"] for r in results]
    exact_times = [r["exact_time"] * 1000 for r in results]
    mc_times = [r["mc_grad_time"] * 1000 for r in results]

    ax2 = axes[1]
    ax2.semilogy(Ns, rel_errs, "o-", color="#C44E52", label="Relative L1 error")
    ax2.set_xlabel("Network size N")
    ax2.set_ylabel("Relative L1 error", color="#C44E52")
    ax2.tick_params(axis="y", labelcolor="#C44E52")

    ax3 = ax2.twinx()
    ax3.plot(Ns, exact_times, "s--", color="#4C72B0", label="Exact (ms)", linewidth=1.2)
    ax3.plot(Ns, mc_times, "^--", color="#DD8452", label="MC (ms)", linewidth=1.2)
    ax3.set_ylabel("Wall time (ms)")

    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax3.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=8)
    ax2.set_title("Error and timing vs network size")

    plt.tight_layout()
    plt.savefig(args.output, dpi=150)
    print(f"\nFigure saved to {args.output}")


if __name__ == "__main__":
    main()

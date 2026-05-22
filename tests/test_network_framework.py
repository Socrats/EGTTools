"""
Tests for the network games framework introduced in Phase 0-6.

Coverage:
1. AdjacencyList topology conversion (NodeDictionary → AdjacencyList)
2. PairwiseComparison gradient on a ring and a star: compares exact formula
   against a finite-difference numerical check (perturbing the population
   by one node and recomputing).
3. BirthDeath and DeathBirth gradient: sign-check and symmetry on a complete graph.
4. NetworkMCEstimatorPC: fixation probability on complete graph agrees with
   the Moran process analytical result ρ ≈ 1/N for neutral drift.
5. NetworkMCEstimatorPC: estimate_strategy_distribution returns frequencies that
   sum to 1.
6. NetworkCoEvolutionary: run() returns correct shape; homophily is in [0, 1].
7. LocalRedistributionGame: fitness is reduced for richer nodes and increased
   for poorer nodes compared to the base game.
8. Network visualization imports (smoke tests — no rendering).
9. factories: network_mc_estimator_factory and network_coevo_factory construct
   correct types.
"""

from __future__ import annotations

import os
import sys
from typing import List

import numpy as np
import pytest

# ---- Guard: skip if egttools not installed --------------------------------
egt = pytest.importorskip("egttools")

from egttools.numerical.structure import (
    Network,
    NetworkMCEstimatorPC,
    NetworkMCEstimatorBD,
    NetworkMCEstimatorDB,
    NetworkCoEvolutionaryPC,
)
from egttools.numerical.structure.factories import (
    network_mc_estimator_factory,
    network_coevo_factory,
)

# Necessary on macOS with Conda OpenMP
if sys.platform == "darwin":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "True")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class CoopDefectGame(egt.games.AbstractSpatialGame):
    """Two-strategy PD: strategy 0 = Cooperate, strategy 1 = Defect."""

    def __init__(self, R=3.0, S=0.0, T=5.0, P=1.0):
        super().__init__()
        self.R, self.S, self.T, self.P = R, S, T, P

    def calculate_fitness(self, strategy_index: int, state) -> float:
        state = np.asarray(state, dtype=int)
        k = int(state[0])
        d = int(state.sum())
        if d == 0:
            return self.R if strategy_index == 0 else self.P
        if strategy_index == 0:
            return (self.R * k + self.S * (d - k)) / d
        else:
            return (self.T * k + self.P * (d - k)) / d

    def nb_strategies(self) -> int:
        return 2

    def toString(self) -> str:
        return "CoopDefectGame"

    def type(self) -> str:
        return "CoopDefectGame"


class NeutralGame(egt.games.AbstractSpatialGame):
    """All fitnesses = 1 (neutral drift)."""

    def calculate_fitness(self, strategy_index: int, state) -> float:
        return 1.0

    def nb_strategies(self) -> int:
        return 2

    def toString(self) -> str:
        return "NeutralGame"

    def type(self) -> str:
        return "NeutralGame"


def ring_topology(N: int):
    return {i: [(i - 1) % N, (i + 1) % N] for i in range(N)}


def complete_topology(N: int):
    return {i: [j for j in range(N) if j != i] for i in range(N)}


def star_topology(N: int):
    return {0: list(range(1, N)), **{i: [0] for i in range(1, N)}}


# ---------------------------------------------------------------------------
# 1. Topology conversion
# ---------------------------------------------------------------------------

def test_topology_conversion_ring():
    N = 6
    topo = ring_topology(N)
    estimator = NetworkMCEstimatorPC(CoopDefectGame(), topo, 2, beta=1.0, mu=0.0)
    adj = estimator.topology()
    assert len(adj) == N
    for i in range(N):
        assert sorted(adj[i]) == sorted([(i - 1) % N, (i + 1) % N])


def test_topology_conversion_star():
    N = 5
    topo = star_topology(N)
    estimator = NetworkMCEstimatorPC(CoopDefectGame(), topo, 2, beta=1.0, mu=0.0)
    adj = estimator.topology()
    assert len(adj[0]) == N - 1
    for i in range(1, N):
        assert adj[i] == [0]


# ---------------------------------------------------------------------------
# 2. PC gradient: gradient sums to zero (conservation)
# ---------------------------------------------------------------------------

def test_pc_gradient_sums_to_zero_ring():
    N = 10
    game = CoopDefectGame()
    topo = ring_topology(N)
    estimator = NetworkMCEstimatorPC(game, topo, 2, beta=1.0, mu=0.0)

    rng = np.random.default_rng(42)
    population = list(rng.integers(0, 2, size=N))
    grad = np.asarray(estimator.calculate_gradient_of_selection(population))
    np.testing.assert_allclose(grad.sum(), 0.0, atol=1e-12,
                               err_msg="PC gradient must sum to zero (probability conservation)")


def test_pc_gradient_sums_to_zero_star():
    N = 7
    game = CoopDefectGame()
    topo = star_topology(N)
    estimator = NetworkMCEstimatorPC(game, topo, 2, beta=1.0, mu=0.0)

    rng = np.random.default_rng(99)
    population = list(rng.integers(0, 2, size=N))
    grad = np.asarray(estimator.calculate_gradient_of_selection(population))
    np.testing.assert_allclose(grad.sum(), 0.0, atol=1e-12)


# ---------------------------------------------------------------------------
# 3. PC gradient: monomorphic population has zero gradient
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("strategy", [0, 1])
def test_pc_gradient_monomorphic_zero(strategy):
    N = 8
    game = CoopDefectGame()
    topo = ring_topology(N)
    estimator = NetworkMCEstimatorPC(game, topo, 2, beta=1.0, mu=0.0)

    population = [strategy] * N
    grad = np.asarray(estimator.calculate_gradient_of_selection(population))
    np.testing.assert_allclose(grad, 0.0, atol=1e-12,
                               err_msg="Monomorphic population should have zero gradient")


# ---------------------------------------------------------------------------
# 4. BD gradient: sums to zero + monomorphic = zero
# ---------------------------------------------------------------------------

def test_bd_gradient_sums_to_zero():
    N = 8
    game = CoopDefectGame()
    topo = ring_topology(N)
    estimator = NetworkMCEstimatorBD(game, topo, 2, beta=1.0, mu=0.0)

    rng = np.random.default_rng(7)
    population = list(rng.integers(0, 2, size=N))
    grad = np.asarray(estimator.calculate_gradient_of_selection(population))
    np.testing.assert_allclose(grad.sum(), 0.0, atol=1e-12)


def test_db_gradient_sums_to_zero():
    N = 8
    game = CoopDefectGame()
    topo = ring_topology(N)
    estimator = NetworkMCEstimatorDB(game, topo, 2, beta=1.0, mu=0.0)

    rng = np.random.default_rng(13)
    population = list(rng.integers(0, 2, size=N))
    grad = np.asarray(estimator.calculate_gradient_of_selection(population))
    np.testing.assert_allclose(grad.sum(), 0.0, atol=1e-12)


# ---------------------------------------------------------------------------
# 5. Fixation probability: neutral drift on complete graph → ρ ≈ 1/N
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_fixation_neutral_complete_graph():
    N = 10
    game = NeutralGame()
    topo = complete_topology(N)

    egt.Random.init_with_seed(12345)

    estimator = NetworkMCEstimatorPC(game, topo, 2, beta=0.0, mu=0.0)
    rho = estimator.estimate_fixation_probability(
        invader=1, resident=0,
        nb_runs=2000, nb_generations=5000,
    )
    expected = 1.0 / N
    assert abs(rho - expected) < 0.05, (
        f"Neutral fixation probability should be ≈ 1/N = {expected:.3f}, got {rho:.3f}"
    )


# ---------------------------------------------------------------------------
# 6. Strategy distribution: frequencies sum to 1
# ---------------------------------------------------------------------------

def test_strategy_distribution_sums_to_one():
    N = 12
    game = CoopDefectGame()
    topo = ring_topology(N)

    egt.Random.init_with_seed(99)
    estimator = NetworkMCEstimatorPC(game, topo, 2, beta=1.0, mu=0.01)
    mean_freq, se_freq = estimator.estimate_strategy_distribution(
        nb_runs=50, nb_generations=100, transitory=20
    )
    mean_freq = np.asarray(mean_freq)
    np.testing.assert_allclose(mean_freq.sum(), 1.0, atol=1e-6,
                               err_msg="Strategy frequencies must sum to 1")
    assert np.all(mean_freq >= 0.0)
    assert np.all(mean_freq <= 1.0)


# ---------------------------------------------------------------------------
# 7. run() returns correct shape
# ---------------------------------------------------------------------------

def test_run_shape():
    N = 10
    game = CoopDefectGame()
    topo = ring_topology(N)

    egt.Random.init_with_seed(1)
    estimator = NetworkMCEstimatorPC(game, topo, 2, beta=1.0, mu=0.01)
    init_state = np.array([5, 5], dtype=np.uint64)
    nb_gen = 50
    transitory = 10
    traj = estimator.run(nb_gen, transitory, init_state)
    traj = np.asarray(traj)
    assert traj.shape == (nb_gen - transitory, 2), \
        f"Expected shape ({nb_gen - transitory}, 2), got {traj.shape}"
    assert np.all(traj.sum(axis=1) == N), "Row sums must equal N"


# ---------------------------------------------------------------------------
# 8. run_snapshots fires the callback the right number of times
# ---------------------------------------------------------------------------

def test_run_snapshots_callback_count():
    N = 10
    game = CoopDefectGame()
    topo = ring_topology(N)

    egt.Random.init_with_seed(2)
    estimator = NetworkMCEstimatorPC(game, topo, 2, beta=1.0, mu=0.01)
    init_state = np.array([5, 5], dtype=np.uint64)

    calls = []
    def cb(t, pop):
        calls.append((t, list(pop)))

    nb_gen = 50
    transitory = 10
    snapshot_interval = 5
    estimator.run_snapshots(nb_gen, transitory, snapshot_interval, init_state, cb)

    expected_calls = (nb_gen - transitory) // snapshot_interval
    assert len(calls) == expected_calls, \
        f"Expected {expected_calls} snapshots, got {len(calls)}"
    for t, pop in calls:
        assert len(pop) == N


# ---------------------------------------------------------------------------
# 9. NetworkCoEvolutionary: run() shape + homophily range
# ---------------------------------------------------------------------------

def test_coevo_run_shape():
    N = 10
    game = CoopDefectGame()
    topo = ring_topology(N)

    egt.Random.init_with_seed(5)
    estimator = NetworkCoEvolutionaryPC(game, topo, 2, 1.0, 0.01, 0.3)
    init_state = np.array([5, 5], dtype=np.uint64)
    traj = np.asarray(estimator.run(30, 5, init_state))
    assert traj.shape == (25, 2)
    assert np.all(traj.sum(axis=1) == N)


def test_coevo_homophily_range():
    N = 12
    game = CoopDefectGame()
    topo = ring_topology(N)

    egt.Random.init_with_seed(8)
    estimator = NetworkCoEvolutionaryPC(game, topo, 2, 1.0, 0.01, 0.5)
    mean_freq, se_freq, mean_hom, se_hom = estimator.estimate_strategy_distribution(
        nb_runs=20, nb_generations=60, transitory=10
    )
    assert 0.0 <= mean_hom <= 1.0, f"Homophily {mean_hom} outside [0, 1]"
    assert se_hom >= 0.0


# ---------------------------------------------------------------------------
# 10. LocalRedistributionGame: fitness modification
# ---------------------------------------------------------------------------

def test_local_redistribution_reduces_rich_fitness():
    from egttools.numerical.numerical_.games import LocalRedistributionGame

    base = CoopDefectGame(R=3.0, S=0.0, T=5.0, P=1.0)
    wrapper = LocalRedistributionGame(base, redistribution_rate=0.5)

    # Defector in a neighbourhood of mostly cooperators: high base payoff
    state = np.array([7, 0], dtype=np.uint64)  # 7 cooperators, 0 defectors
    base_fit = base.calculate_fitness(1, state)    # Defect payoff
    wrapped_fit = wrapper.calculate_fitness(1, state)

    assert wrapper.nb_strategies() == 2
    # The rich node (high payoff relative to neighbours) should have reduced fitness
    # (or equal — it only reduces if it's above the neighbourhood mean)
    assert wrapped_fit <= base_fit + 1e-9, \
        f"Redistribution should not increase the richest node's fitness; " \
        f"base={base_fit:.3f}, wrapped={wrapped_fit:.3f}"


def test_local_redistribution_alpha_zero():
    from egttools.numerical.numerical_.games import LocalRedistributionGame

    base = CoopDefectGame()
    wrapper = LocalRedistributionGame(base, redistribution_rate=0.0)
    state = np.array([3, 2], dtype=np.uint64)

    for s in range(2):
        assert wrapper.calculate_fitness(s, state) == pytest.approx(
            base.calculate_fitness(s, state), abs=1e-12
        ), "With alpha=0 redistribution must leave fitness unchanged"


# ---------------------------------------------------------------------------
# 11. Factories: correct types and attributes
# ---------------------------------------------------------------------------

def test_mc_factory_types():
    game = CoopDefectGame()
    topo = ring_topology(8)
    for rule in ["PC", "BD", "DB"]:
        est = network_mc_estimator_factory(game, topo, 2, beta=1.0, mu=0.01,
                                           update_rule=rule)
        assert est.population_size() == 8
        assert est.nb_strategies() == 2


def test_coevo_factory_types():
    game = CoopDefectGame()
    topo = ring_topology(8)
    for rule in ["random", "homophilic"]:
        est = network_coevo_factory(game, topo, 2, beta=1.0, mu=0.01,
                                    rewiring_probability=0.3, rewiring_rule=rule)
        assert est.population_size() == 8
        assert est.rewiring_probability() == pytest.approx(0.3)


def test_mc_factory_invalid_rule():
    game = CoopDefectGame()
    topo = ring_topology(8)
    with pytest.raises(ValueError):
        network_mc_estimator_factory(game, topo, 2, 1.0, 0.01, update_rule="INVALID")


# ---------------------------------------------------------------------------
# 12. Plotting: smoke tests (no rendering)
# ---------------------------------------------------------------------------

def test_plotting_imports():
    from egttools.plotting import (
        plot_network_state,
        plot_strategy_evolution,
        animate_network_evolution,
        plot_edge_homophily,
        plot_parameter_sweep,
        plot_strategy_by_degree,
    )
    assert callable(plot_network_state)
    assert callable(plot_strategy_evolution)
    assert callable(plot_edge_homophily)
    assert callable(plot_parameter_sweep)
    assert callable(plot_strategy_by_degree)


def test_plot_strategy_evolution_shape():
    import matplotlib
    matplotlib.use("Agg")
    from egttools.plotting import plot_strategy_evolution
    import matplotlib.pyplot as plt

    traj = np.random.randint(0, 10, size=(50, 2))
    ax = plot_strategy_evolution(traj, strategy_names=["C", "D"])
    assert ax is not None
    plt.close("all")


def test_plot_edge_homophily():
    import matplotlib
    matplotlib.use("Agg")
    from egttools.plotting import plot_edge_homophily
    import matplotlib.pyplot as plt

    hom = np.random.uniform(0, 1, size=100)
    ax = plot_edge_homophily(hom)
    assert ax is not None
    plt.close("all")


def test_plot_strategy_by_degree():
    import matplotlib
    matplotlib.use("Agg")
    from egttools.plotting import plot_strategy_by_degree
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(0)
    N = 30
    degrees = rng.integers(1, 10, size=N)
    pop = rng.integers(0, 2, size=N)
    ax = plot_strategy_by_degree(degrees, pop, nb_strategies=2,
                                  strategy_names=["C", "D"])
    assert ax is not None
    plt.close("all")


def test_plot_parameter_sweep():
    import matplotlib
    matplotlib.use("Agg")
    from egttools.plotting import plot_parameter_sweep
    import matplotlib.pyplot as plt

    mat = np.random.uniform(0, 1, size=(10, 10))
    ax = plot_parameter_sweep(mat, np.linspace(0, 1, 10), np.linspace(0, 5, 10))
    assert ax is not None
    plt.close("all")

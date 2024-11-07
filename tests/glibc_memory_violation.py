import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import egttools as egt

nb_strategies = 3
markers = ["P", "o", "v", "h", "s", "X", "D"]
colors = sns.color_palette("icefire", nb_strategies)


def payoffs(p_o, p_c, e_c, e_o, b_c, b_cf, b_o, b_p, c_cf, c_c, c_o, c_po):
    return np.array([
        [0, p_o * b_o + (1 - p_o) * (-c_o) - e_o, c_po],  # O , C, P
        [p_o * (-c_c), (b_cf - c_cf) / 2, p_c * b_c - e_c],  # C
        [-c_po, -p_c * c_c, b_p]  # P
    ])


strategy_labels = ["O", "C", "P"]

# define the parameters
p_o = 0.2
p_c = 0.5
e_c = 1
e_o = 2
b_c = 15
b_cf = 2
b_o = 10
b_p = 10
c_cf = 5
c_c = 50
c_o = 3
c_po = 1

payoff_matrix_1 = payoffs(p_o, p_c, e_c, e_o, b_c, b_cf, b_o, b_p, c_cf, c_c, c_o, c_po)

from scipy.integrate import odeint

nb_runs = 1000
nb_time_steps = 1000
t = np.arange(0, 10, 10 / nb_time_steps)


def run_replicator_dynamics(t: np.ndarray, payoff_matrix: np.ndarray, nb_runs: int):
    results = []
    for i in range(nb_runs):
        x0 = egt.sample_unit_simplex(3)
        result = odeint(lambda x, t: egt.analytical.replicator_equation(x, payoff_matrix), x0, t)
        results.append(result)

    return results


results = run_replicator_dynamics(t, payoff_matrix_1, nb_runs)

results = np.asarray(results)

fig, ax = plt.subplots(figsize=(10, 3))

for run in results:
    for i in range(nb_strategies):
        ax.plot(t, run[:, i],
                linewidth=.05, alpha=0.6, color=colors[i])

for i in range(nb_strategies):
    ax.plot(t, np.mean(results[:, :, i], axis=0), linewidth=1.5,
            alpha=1, color=colors[i], label=strategy_labels[i])

ax.legend(frameon=False, bbox_to_anchor=(1.1, 1, 0, 0), loc='upper right')
ax.set_ylabel("frequency", fontsize=14)
ax.set_xlabel("time step, $t$", fontsize=14)
# ax.set_ylim(-0.2, 1.2)
sns.despine()
plt.show()

# reinitialize the changed values
p_o = 0.2
p_c = 0.5

from egttools.games import Matrix2PlayerGameHolder
from egttools.numerical import PairwiseComparisonNumerical

Z = 100
beta = 1
mu = 1e-3
cache_size = 1000
nb_population_states = egt.calculate_nb_states(Z, nb_strategies)

nb_runs = 1000
nb_generations = 1000

payoff_matrix = payoffs(p_o, p_c, e_c, e_o, b_c, b_cf, b_o, b_p, c_cf, c_c, c_o, c_po)
game = Matrix2PlayerGameHolder(nb_strategies, payoff_matrix)
evolver = PairwiseComparisonNumerical(Z, game, cache_size)

results = []

for i in range(nb_runs):
    index = np.random.randint(0, nb_population_states)
    x0 = egt.sample_simplex(index, Z, nb_strategies)
    result = evolver.run_with_mutation(nb_generations, beta, mu, x0)
    results.append(result)
results = np.asarray(results)

print("first done")

results = []

for i in range(nb_runs):
    index = np.random.randint(0, nb_population_states)
    x0 = egt.sample_simplex(index, Z, nb_strategies)
    result = evolver.run_with_mutation(nb_generations, beta, mu, x0)
    results.append(result)
results = np.asarray(results)

print("second done")

payoff_matrix2 = payoffs(p_o, p_c, e_c, e_o, b_c, b_cf, b_o, b_p, c_cf, c_c, c_o, c_po)
game2 = Matrix2PlayerGameHolder(nb_strategies, payoff_matrix2)
evolver2 = PairwiseComparisonNumerical(Z, game2, cache_size)

print("now we try to reinstantiate everything")

fig, ax = plt.subplots(figsize=(10, 3))

for run in results:
    for i in range(nb_strategies):
        ax.plot(range(nb_generations + 1), run[:, i] / Z,
                linewidth=.05, alpha=0.6, color=colors[i])

for i in range(nb_strategies):
    ax.plot(range(nb_generations + 1), np.mean(results[:, :, i] / Z, axis=0), linewidth=1.5,
            alpha=1, color=colors[i], label=strategy_labels[i])

ax.legend(frameon=False, bbox_to_anchor=(1.1, 1, 0, 0), loc='upper right')
ax.set_ylabel("proportion ($k/Z$)", fontsize=14)
ax.set_xlabel("generation", fontsize=14)
# ax.set_ylim(-0.2, 1.2)
sns.despine()
plt.show()

# This is the part that should break

payoff_matrix = payoffs(p_o, p_c, e_c, e_o, b_c, b_cf, b_o, b_p, c_cf, c_c, c_o, c_po)
game = Matrix2PlayerGameHolder(nb_strategies, payoff_matrix)
evolver = PairwiseComparisonNumerical(Z, game, cache_size)

results = []

for i in range(nb_runs):
    index = np.random.randint(0, nb_population_states)
    x0 = egt.sample_simplex(index, Z, nb_strategies)
    result = evolver.run_with_mutation(nb_generations, beta, mu, x0)
    results.append(result)
results = np.asarray(results)

fig, ax = plt.subplots(figsize=(10, 3))

for run in results:
    for i in range(nb_strategies):
        ax.plot(range(nb_generations + 1), run[:, i] / Z,
                linewidth=.05, alpha=0.6, color=colors[i])

for i in range(nb_strategies):
    ax.plot(range(nb_generations + 1), np.mean(results[:, :, i] / Z, axis=0), linewidth=1.5,
            alpha=1, color=colors[i], label=strategy_labels[i])

ax.legend(frameon=False, bbox_to_anchor=(1.1, 1, 0, 0), loc='upper right')
ax.set_ylabel("proportion ($k/Z$)", fontsize=14)
ax.set_xlabel("generation", fontsize=14)
# ax.set_ylim(-0.2, 1.2)
sns.despine()
plt.show()

"""Direct transcriptions of the definitions in ``docs/theory/``, shared by the spec tests.

Each function transcribes one numbered definition, with no shortcuts, so the tests compare
the implementations against the specification rather than against another implementation.
"""
import itertools
from math import comb

import numpy as np

import egttools as egt

# ---------------------------------------------------------------------------
# docs/theory/pairwise_comparison.md (cited as PC 1.1, PC 2.1, ...)
# ---------------------------------------------------------------------------

def all_states(z, k):
    """Definition 1.1, in EGTtools' state-index order."""
    # Signed counts: n - e_l must not wrap around for an absent strategy.
    return np.array([egt.sample_simplex(i, z, k) for i in range(egt.calculate_nb_states(z, k))], dtype=np.int64)


def fitness_two_player(a, n):
    """Definition 2.1."""
    z, k = n.sum(), len(n)
    return np.array([sum(a[l, m] * (n[m] - (l == m)) / (z - 1) for m in range(k)) for l in range(k)])


def fitness_n_player(payoff_table, group_size, n):
    """Definition 2.2. payoff_table[l, idx] is the payoff of l in the full group with state index idx."""
    z, k = n.sum(), len(n)
    out = np.zeros(k)
    for l in range(k):
        if n[l] == 0:
            continue
        others = n - np.eye(k, dtype=int)[l]
        for g in itertools.product(range(group_size), repeat=k):
            g = np.array(g)
            if g.sum() != group_size - 1:
                continue
            weight = np.prod([comb(int(others[m]), int(g[m])) for m in range(k)]) / comb(int(z - 1), group_size - 1)
            if weight == 0:
                continue
            full_group = g + np.eye(k, dtype=int)[l]
            out[l] += payoff_table[l, egt.calculate_state(group_size, full_group)] * weight
    return out


def fermi(beta, f_focal, f_model):
    """Definition 3.1."""
    return 1.0 / (1.0 + np.exp(-beta * (f_model - f_focal)))


def transition(n, j, l, f, beta, mu):
    """Definition 4.1: probability that one j-strategist becomes an l-strategist."""
    z, k = n.sum(), len(n)
    if j == l or n[j] == 0:
        return 0.0
    return n[j] / z * ((1 - mu) * n[l] / (z - 1) * fermi(beta, f[j], f[l]) + mu / (k - 1))


def gradient(n, f, beta, mu):
    """Definition 5.1, from the transitions (not from the closed form)."""
    k = len(n)
    return np.array([sum(transition(n, j, l, f, beta, mu) - transition(n, l, j, f, beta, mu)
                         for j in range(k) if j != l) for l in range(k)])




# ---------------------------------------------------------------------------
# docs/theory/fixation.md (cited as F 2.1, F 2.2, ...); mu = 0 throughout
# ---------------------------------------------------------------------------

def edge_state(z, k, a, b, i):
    """The state with i a-strategists and z - i b-strategists (F 2.1)."""
    n = np.zeros(k, dtype=int)
    n[a], n[b] = i, z - i
    return n


def fixation_probability(fit, z, k, a, b, beta):
    """F 2.2 for a single invader, from the ratios T-(j) / T+(j) of PC 4.1 (mu = 0)."""
    total, product = 0.0, 1.0
    for j in range(1, z):
        n = edge_state(z, k, a, b, j)
        f = fit(n)
        product *= transition(n, a, b, f, beta, 0.0) / transition(n, b, a, f, beta, 0.0)
        total += product
    return 1.0 / (1.0 + total)


def log_fixation_probability(fit, z, k, a, b, beta):
    """F 2.2 in log form: log rho = -log(1 + sum_m exp(L_m)), L_m = -beta sum_{j <= m} delta_f(j)."""
    deltas = []
    for j in range(1, z):
        f = fit(edge_state(z, k, a, b, j))
        deltas.append(f[a] - f[b])
    partial_sums = -beta * np.cumsum(deltas)
    return -np.logaddexp(0.0, np.logaddexp.reduce(partial_sums))


def absorbing_chain(fit, z, k, beta):
    """F 3.3: absorption probabilities B, mean absorption time t and Var[tau] for every transient state.

    The transition matrix is assembled from PC 4.1 (mu = 0) state by state. Returns the transient
    states (as rows) and B (one column per strategy), t and Var[tau] in the same order.
    """
    states = all_states(z, k)
    index = {tuple(n): i for i, n in enumerate(states)}
    p = np.zeros((len(states), len(states)))
    for i, n in enumerate(states):
        f = fit(n)
        for j, l in itertools.permutations(range(k), 2):
            if n[j] == 0:
                continue
            p[i, index[tuple(n - np.eye(k, dtype=int)[j] + np.eye(k, dtype=int)[l])]] += transition(n, j, l, f, beta, 0.0)
        p[i, i] = 1.0 - p[i].sum()
    homogeneous = [index[tuple(z * np.eye(k, dtype=int)[l])] for l in range(k)]
    transient = [i for i in range(len(states)) if i not in homogeneous]
    q = p[np.ix_(transient, transient)]
    r = p[np.ix_(transient, homogeneous)]
    fundamental = np.linalg.inv(np.eye(len(transient)) - q)
    t = fundamental @ np.ones(len(transient))
    variance = (2 * fundamental - np.eye(len(transient))) @ t - t * t
    return states[transient], fundamental @ r, t, variance

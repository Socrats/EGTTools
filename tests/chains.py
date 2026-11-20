"""Markov chains with known stationary distributions, shared by the tests (roadmap TEST-11).

Independent of egttools: these chains are references for testing the library's solvers (SOL-10)
and Monte Carlo estimators (MC-09). Every function returns a row-stochastic transition matrix P and,
except ``random_row_stochastic``, its stationary distribution pi (pi P = pi, sum(pi) = 1), computed
from a closed form or an exact argument, not by a solver.

    random_row_stochastic(n, rng)          irreducible, aperiodic; pi from ``reference_stationary``
    reference_stationary(P)                dense direct solve of pi P = pi, sum(pi) = 1 (small chains)
    two_state(a, b)                        pi = (b, a) / (a + b)
    doubly_stochastic(n, rng)              columns also sum to 1, so pi is uniform
    birth_death_chain(n, rng)              tridiagonal; pi from detailed balance
    neutral_pairwise_comparison(Z, mu)     2 strategies, beta = 0 (docs/theory/pairwise_comparison.md)
"""
from __future__ import annotations

import numpy as np
from scipy import sparse


def random_row_stochastic(n, rng, density=1.0):
    """Irreducible, aperiodic row-stochastic matrix.

    A random cycle guarantees irreducibility and a small diagonal aperiodicity; ``density`` < 1 zeroes
    a random fraction of the other entries.
    """
    p = rng.random((n, n)) * (rng.random((n, n)) < density)
    perm = rng.permutation(n)
    p[perm, np.roll(perm, 1)] += 0.5
    p += 0.01 * np.eye(n)  # avoid rows with a single entry equal to 1
    return p / p.sum(axis=1, keepdims=True)


def reference_stationary(p):
    """Solve pi P = pi, sum(pi) = 1 directly (dense; for small chains)."""
    p = p.toarray() if sparse.issparse(p) else np.asarray(p, dtype=float)
    n = p.shape[0]
    a = p.T - np.eye(n)
    a[-1] = 1.0
    b = np.zeros(n)
    b[-1] = 1.0
    return np.linalg.solve(a, b)


def two_state(a, b):
    """P = [[1 - a, a], [b, 1 - b]] with 0 < a, b <= 1; pi = (b, a) / (a + b)."""
    if not (0 < a <= 1 and 0 < b <= 1):
        raise ValueError("a and b must be in (0, 1]")
    p = np.array([[1.0 - a, a], [b, 1.0 - b]])
    return p, np.array([b, a]) / (a + b)


def doubly_stochastic(n, rng, nb_permutations=4):
    """Convex combination of permutation matrices (Birkhoff): rows and columns sum to 1, pi uniform.

    The identity (aperiodicity) and a cyclic shift (irreducibility) are always included.
    """
    permutations = [np.arange(n), np.roll(np.arange(n), 1)]
    permutations += [rng.permutation(n) for _ in range(nb_permutations)]
    weights = rng.random(len(permutations)) + 0.1
    weights /= weights.sum()
    p = np.zeros((n, n))
    for weight, permutation in zip(weights, permutations):
        p[np.arange(n), permutation] += weight
    return p, np.full(n, 1.0 / n)


def birth_death_chain(n, rng):
    """Row-stochastic birth-death chain (sparse) and its closed-form stationary distribution.

    The up/down ratio oscillates quickly, so pi stays within a small range and the chain has no large
    probability barriers. Random or slowly varying ratios make pi span many orders of magnitude and the
    problem so ill-conditioned that a vector with residual ~1e-17 can still differ from the closed form
    by percents.
    """
    i = np.arange(n - 1)
    up = rng.uniform(0.1, 0.25, size=n - 1)  # i -> i + 1
    down = up * np.exp(-0.5 * np.sin(0.5 * i))  # i + 1 -> i, at most 0.25 * e**0.5 < 0.42
    p = sparse.diags([down, np.zeros(n), up], offsets=[-1, 0, 1], format="lil")
    p.setdiag(1. - np.asarray(p.sum(axis=1)).ravel())
    # Detailed balance: pi_{i+1} / pi_i = up_i / down_i
    pi = np.concatenate([[1.], np.cumprod(up / down)])
    return p.tocsr(), pi / pi.sum()


def neutral_pairwise_comparison(population_size, mu):
    """Pairwise comparison process with 2 strategies at beta = 0 (neutral), with mutation mu > 0.

    States are n = 0, ..., Z, the number of individuals of the first strategy (the population state is
    (n, Z - n)). From Definition 4.1 of docs/theory/pairwise_comparison.md with k = 2 and
    phi = 1/2 at beta = 0:

        T+(n) = (Z - n)/Z * [(1 - mu) * n/(Z - 1) * 1/2 + mu]    (n -> n + 1)
        T-(n) = n/Z * [(1 - mu) * (Z - n)/(Z - 1) * 1/2 + mu]      (n -> n - 1)

    A birth-death chain, so pi follows from detailed balance: pi_{n+1} / pi_n = T+(n) / T-(n + 1),
    accumulated in log space. pi is symmetric (pi_n = pi_{Z - n}).
    """
    z = int(population_size)
    if z < 2 or not 0 < mu <= 1:
        raise ValueError("need Z >= 2 and 0 < mu <= 1")
    n = np.arange(z + 1, dtype=float)
    up = (z - n) / z * ((1 - mu) * n / (z - 1) / 2 + mu)
    down = n / z * ((1 - mu) * (z - n) / (z - 1) / 2 + mu)
    p = sparse.diags([down[1:], 1.0 - up - down, up[:-1]], offsets=[-1, 0, 1], format="csr")
    log_ratios = np.log(up[:-1]) - np.log(down[1:])
    log_pi = np.concatenate([[0.0], np.cumsum(log_ratios)])
    pi = np.exp(log_pi - log_pi.max())
    return p, pi / pi.sum()

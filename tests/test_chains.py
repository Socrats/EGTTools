"""Self-tests of tests/chains.py: every chain is row-stochastic and its pi is stationary.

The last test also checks egttools against the neutral chain: PairwiseComparison's transition matrix
at beta = 0 for two strategies must equal Definition 4.1 of docs/theory/pairwise_comparison.md.
"""
import numpy as np
import pytest
from scipy import sparse

from chains import (
    birth_death_chain,
    doubly_stochastic,
    neutral_pairwise_comparison,
    random_row_stochastic,
    reference_stationary,
    two_state,
)


def dense(p):
    return p.toarray() if sparse.issparse(p) else np.asarray(p)


def assert_stationary(p, pi, atol=1e-12):
    p = dense(p)
    assert np.all(p >= -1e-15)
    np.testing.assert_allclose(p.sum(axis=1), 1.0, atol=1e-12)
    assert np.all(pi >= 0)
    assert pi.sum() == pytest.approx(1.0)
    np.testing.assert_allclose(pi @ p, pi, atol=atol)


def test_random_row_stochastic_and_reference(rng):
    for n in (2, 5, 30):
        p = random_row_stochastic(n, rng, density=0.3)
        assert_stationary(p, reference_stationary(p))


def test_two_state():
    p, pi = two_state(0.2, 0.3)
    assert_stationary(p, pi)
    np.testing.assert_allclose(pi, [0.6, 0.4])


def test_doubly_stochastic_has_uniform_pi(rng):
    p, pi = doubly_stochastic(25, rng)
    np.testing.assert_allclose(p.sum(axis=0), 1.0, atol=1e-12)
    assert_stationary(p, pi)
    np.testing.assert_allclose(reference_stationary(p), pi, atol=1e-12)


def test_birth_death_chain(rng):
    p, pi = birth_death_chain(200, rng)
    assert_stationary(p, pi)


@pytest.mark.parametrize("population_size, mu", [(2, 0.5), (10, 0.01), (50, 0.1), (200, 1e-3)])
def test_neutral_pairwise_comparison(population_size, mu):
    p, pi = neutral_pairwise_comparison(population_size, mu)
    assert_stationary(p, pi)
    np.testing.assert_allclose(pi, pi[::-1], rtol=1e-10)  # symmetric in the two strategies
    np.testing.assert_allclose(reference_stationary(p), pi, rtol=1e-8, atol=1e-14)


def test_neutral_chain_matches_egttools_at_beta_zero():
    egt = pytest.importorskip("egttools")
    from egttools.analytical import PairwiseComparison

    population_size, mu = 12, 0.05
    payoffs = np.array([[1.0, 3.0], [0.5, 2.0]])  # any game: beta = 0 makes it irrelevant
    game = egt.games.Matrix2PlayerGameHolder(2, payoffs)
    transition = dense(PairwiseComparison(population_size, game).calculate_transition_matrix(0.0, mu))
    p, _ = neutral_pairwise_comparison(population_size, mu)
    p = dense(p)
    # egttools state index i -> count n of the first strategy.
    counts = [int(egt.sample_simplex(i, population_size, 2)[0]) for i in range(population_size + 1)]
    np.testing.assert_allclose(transition, p[np.ix_(counts, counts)], atol=1e-14)

"""Correctness tests for egttools.utils.calculate_stationary_distribution{,_non_hermitian}.

Both functions take a column-stochastic matrix (the transpose of a row-stochastic
transition matrix) and must return the unique stationary distribution of an
irreducible chain, for dense and for every scipy.sparse input type.
"""
import warnings

import numpy as np
import pytest
from scipy import sparse

from chains import birth_death_chain, random_row_stochastic, reference_stationary
from egttools.utils import (
    calculate_stationary_distribution,
    calculate_stationary_distribution_non_hermitian,
)

SOLVERS = [calculate_stationary_distribution, calculate_stationary_distribution_non_hermitian]
SPARSE_TYPES = [sparse.csr_matrix, sparse.csc_matrix, sparse.csr_array, sparse.csc_array, sparse.coo_array]


@pytest.mark.parametrize("solver", SOLVERS)
def test_random_dense_chains_match_reference(solver):
    rng = np.random.default_rng(12345)
    for _ in range(200):
        p = random_row_stochastic(int(rng.integers(2, 20)), rng)
        np.testing.assert_allclose(solver(p.T), reference_stationary(p), atol=1e-10)


@pytest.mark.parametrize("solver", SOLVERS)
@pytest.mark.parametrize("sparse_type", SPARSE_TYPES)
def test_sparse_inputs_match_reference(solver, sparse_type):
    rng = np.random.default_rng(7)
    for _ in range(20):
        p = random_row_stochastic(int(rng.integers(2, 40)), rng, density=0.3)
        np.testing.assert_allclose(solver(sparse_type(p.T)), reference_stationary(p), atol=1e-10)


@pytest.mark.parametrize("solver", SOLVERS)
def test_birth_death_closed_form(solver):
    p, pi = birth_death_chain(60, np.random.default_rng(3))
    np.testing.assert_allclose(solver(p.T.toarray()), pi, rtol=1e-8, atol=1e-12)
    np.testing.assert_allclose(solver(p.T.tocsr()), pi, rtol=1e-8, atol=1e-12)


def test_large_sparse_chain_is_not_densified():
    # A dense copy of this matrix would need 50_000**2 * 8 bytes = 20 GB.
    p, pi = birth_death_chain(50_000, np.random.default_rng(11))
    result = calculate_stationary_distribution(p.T.tocsr())
    assert np.abs(result @ p - result).sum() < 1e-12
    np.testing.assert_allclose(result, pi, rtol=1e-6, atol=1e-12)


@pytest.mark.parametrize("solver", SOLVERS)
def test_result_is_a_probability_vector_and_invariant(solver):
    p = random_row_stochastic(30, np.random.default_rng(99))
    pi = solver(p.T)
    assert pi.shape == (30,)
    assert np.all(pi >= 0)
    assert pi.sum() == pytest.approx(1.)
    np.testing.assert_allclose(pi @ p, pi, atol=1e-12)


def test_reducible_sparse_chain_warns():
    # Two absorbing states: no unique stationary distribution.
    p = np.array([[1., 0., 0.],
                  [0.3, 0.4, 0.3],
                  [0., 0., 1.]])
    with pytest.warns(RuntimeWarning):
        pi = calculate_stationary_distribution(sparse.csr_matrix(p.T))
    assert pi.sum() == pytest.approx(1.)


def test_non_hermitian_warns_when_eigenvalue_one_is_not_simple():
    p = np.array([[1., 0., 0.],
                  [0.3, 0.4, 0.3],
                  [0., 0., 1.]])
    with pytest.warns(RuntimeWarning, match="not simple"):
        calculate_stationary_distribution_non_hermitian(p.T)


@pytest.mark.parametrize("solver", SOLVERS)
def test_no_degeneracy_warning_for_regular_chain(solver):
    p = random_row_stochastic(10, np.random.default_rng(5))
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        solver(p.T)

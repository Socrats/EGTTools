"""Tests for PairwisePetscOperator (MPI-distributed SLEPc eigensolver).

These tests are skipped automatically when the ``numerical_mpi_`` C++ extension
is absent (i.e. EGTtools was not compiled with ``EGTTOOLS_ENABLE_PETSC=ON``).

Run as a standard pytest suite (single-rank):
    pytest tests/test_mpi_operator.py

Run with multiple MPI ranks:
    mpiexec -n 2 pytest tests/test_mpi_operator.py
"""
from __future__ import annotations

import numpy as np
import pytest

# Skip all tests in this module if numerical_mpi_ was not compiled.
petsc_op = pytest.importorskip(
    "egttools.numerical.numerical_mpi_",
    reason="numerical_mpi_ not compiled (EGTTOOLS_ENABLE_PETSC=OFF)",
)
PairwisePetscOperator = petsc_op.PairwisePetscOperator

from egttools.numerical.numerical_ import PairwiseComparison  # for reference
from egttools.numerical.linear_operator import stationary_distribution_from_sparse


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def hawk_dove_game():
    """2-strategy Hawk-Dove game (2x2 normal form).

    NormalFormGame(nb_interactions, payoffs) — nb_interactions=1 for pairwise games.
    """
    from egttools.games import NormalFormGame
    v, d = 2, 3
    payoffs = np.array([[(v - d) / 2, v], [0, v / 2]], dtype=np.float64)
    return NormalFormGame(1, payoffs)


@pytest.fixture(scope="module")
def rps_game():
    """3-strategy Rock-Paper-Scissors game.

    Uses Matrix2PlayerGameHolder which explicitly takes nb_strategies as first arg.
    """
    from egttools.games import Matrix2PlayerGameHolder
    payoffs = np.array([
        [0, -1, 1],
        [1,  0, -1],
        [-1, 1, 0],
    ], dtype=np.float64)
    return Matrix2PlayerGameHolder(3, payoffs)


@pytest.fixture(scope="module")
def hawk_dove_op(hawk_dove_game):
    return PairwisePetscOperator(
        population_size=10, game=hawk_dove_game, beta=1.0, mu=0.05
    )


@pytest.fixture(scope="module")
def rps_op(rps_game):
    return PairwisePetscOperator(
        population_size=8, game=rps_game, beta=0.5, mu=0.05
    )


# ---------------------------------------------------------------------------
# Construction tests
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_size_2strategy(self, hawk_dove_op):
        # C(10+2-1, 2-1) = C(11, 1) = 11
        assert hawk_dove_op.size == 11

    def test_size_3strategy(self, rps_op):
        # C(8+3-1, 3-1) = C(10, 2) = 45
        assert rps_op.size == 45

    def test_accessors(self, hawk_dove_op):
        assert hawk_dove_op.population_size == 10
        assert hawk_dove_op.nb_strategies  == 2
        assert hawk_dove_op.beta           == pytest.approx(1.0)
        assert hawk_dove_op.mu             == pytest.approx(0.05)

    def test_invalid_population_size(self, hawk_dove_game):
        with pytest.raises((ValueError, RuntimeError)):
            PairwisePetscOperator(1, hawk_dove_game, beta=1.0, mu=0.05)

    def test_invalid_mu(self, hawk_dove_game):
        with pytest.raises((ValueError, RuntimeError)):
            PairwisePetscOperator(10, hawk_dove_game, beta=1.0, mu=1.5)


# ---------------------------------------------------------------------------
# Stationary distribution tests
# ---------------------------------------------------------------------------

class TestStationaryDistribution:
    def test_sums_to_one_2strategy(self, hawk_dove_op):
        pi = hawk_dove_op.compute_stationary_distribution()
        assert pi.sum() == pytest.approx(1.0, abs=1e-8)

    def test_non_negative_2strategy(self, hawk_dove_op):
        pi = hawk_dove_op.compute_stationary_distribution()
        assert np.all(pi >= -1e-10)

    def test_length_2strategy(self, hawk_dove_op):
        pi = hawk_dove_op.compute_stationary_distribution()
        assert len(pi) == hawk_dove_op.size

    def test_matches_sparse_reference_2strategy(self, hawk_dove_game):
        """PETSc result must match scipy eigs on the explicit CSR matrix."""
        Z, beta, mu = 10, 1.0, 0.05
        pc_ref = PairwiseComparison(Z, hawk_dove_game)
        P      = pc_ref.calculate_transition_matrix(beta, mu)
        pi_ref = stationary_distribution_from_sparse(P)

        op = PairwisePetscOperator(Z, hawk_dove_game, beta=beta, mu=mu)
        pi = op.compute_stationary_distribution(tol=1e-12)

        np.testing.assert_allclose(pi, pi_ref, atol=1e-6)

    def test_matches_sparse_reference_3strategy(self, rps_game):
        """3-strategy RPS must match scipy eigs on the explicit CSR matrix."""
        Z, beta, mu = 8, 0.5, 0.05
        pc_ref = PairwiseComparison(Z, rps_game)
        P      = pc_ref.calculate_transition_matrix(beta, mu)
        pi_ref = stationary_distribution_from_sparse(P)

        op = PairwisePetscOperator(Z, rps_game, beta=beta, mu=mu)
        pi = op.compute_stationary_distribution(tol=1e-12)

        np.testing.assert_allclose(pi, pi_ref, atol=1e-6)

    def test_sums_to_one_3strategy(self, rps_op):
        pi = rps_op.compute_stationary_distribution()
        assert pi.sum() == pytest.approx(1.0, abs=1e-8)

    def test_non_negative_3strategy(self, rps_op):
        pi = rps_op.compute_stationary_distribution()
        assert np.all(pi >= -1e-10)

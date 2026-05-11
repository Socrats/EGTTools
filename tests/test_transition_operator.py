"""Tests for PairwiseComparisonTransitionOperator.

Verifies correctness by comparing the matrix-free operator against the
explicitly assembled sparse transition matrix produced by the analytical
PairwiseComparison class for small systems.

Test coverage
-------------
1. Construction and property accessors.
2. Column-stochastic property: (P^T x).sum() == x.sum() for random x.
3. Row-stochastic property: (P x).sum() == x.sum() for random x.
4. apply_transpose column-by-column matches columns of P^T from sparse matrix.
5. apply column-by-column matches columns of P from sparse matrix.
6. apply_residual == x - apply_transpose(x).
7. Stationary distribution via scipy eigs matches analytical stationary distribution.
8. LinearOperator wrappers (make_transition_operator, make_residual_operator).
"""
import os
from sys import platform

import numpy as np
import pytest
from scipy.sparse.linalg import eigs

egt = pytest.importorskip("egttools")

PairwiseComparisonTransitionOperator = egt.numerical.PairwiseComparisonTransitionOperator
PairwiseComparison = egt.numerical.numerical_.PairwiseComparison
NormalFormGame = egt.games.NormalFormGame
Matrix2PlayerGameHolder = egt.games.Matrix2PlayerGameHolder

from egttools.numerical.linear_operator import (
    make_transition_operator,
    make_residual_operator,
    stationary_distribution_from_sparse,
)


# ---------------------------------------------------------------------------
# Seed / platform fixture
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _setup():
    if platform == "darwin":
        os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
    egt.Random.init_with_seed(42)


# ---------------------------------------------------------------------------
# Small-system fixtures  (2 strategies, Z=10)
# ---------------------------------------------------------------------------

@pytest.fixture
def hawk_dove_params():
    v, d = 2, 3
    payoffs = np.array([[(v - d) / 2, v], [0, v / 2]], dtype=float)
    game = NormalFormGame(1, payoffs)
    return dict(game=game, Z=10, beta=1.0, mu=0.01)


@pytest.fixture
def hawk_dove_op(hawk_dove_params):
    p = hawk_dove_params
    return PairwiseComparisonTransitionOperator(p["Z"], p["game"], p["beta"], p["mu"])


@pytest.fixture
def hawk_dove_analytical(hawk_dove_params):
    p = hawk_dove_params
    return PairwiseComparison(p["Z"], p["game"])


@pytest.fixture
def rps_params():
    payoffs = np.array([[0, -1, 1], [1, 0, -1], [-1, 1, 0]], dtype=float)
    game = Matrix2PlayerGameHolder(3, payoffs)
    return dict(game=game, Z=8, beta=1.0, mu=0.02)


@pytest.fixture
def rps_op(rps_params):
    p = rps_params
    return PairwiseComparisonTransitionOperator(p["Z"], p["game"], p["beta"], p["mu"])


@pytest.fixture
def rps_analytical(rps_params):
    p = rps_params
    return PairwiseComparison(p["Z"], p["game"])


# ---------------------------------------------------------------------------
# Helper: assemble P^T as dense matrix via apply_transpose
# ---------------------------------------------------------------------------

def operator_to_dense_PT(op):
    """Return P^T as a dense matrix by applying op to each basis vector."""
    n = op.size
    PT = np.zeros((n, n))
    x = np.zeros(n)
    y = np.zeros(n)
    for i in range(n):
        x[:] = 0.0
        x[i] = 1.0
        op.apply_transpose(x, y)
        PT[:, i] = y  # column i of P^T = apply_transpose(e_i)
    return PT


def operator_to_dense_P(op):
    """Return P as a dense matrix by applying op to each basis vector."""
    n = op.size
    P = np.zeros((n, n))
    x = np.zeros(n)
    y = np.zeros(n)
    for i in range(n):
        x[:] = 0.0
        x[i] = 1.0
        op.apply(x, y)
        P[:, i] = y  # column i of P = apply(e_i)
    return P


# ---------------------------------------------------------------------------
# Construction tests
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_properties_2strategy(self, hawk_dove_op, hawk_dove_params):
        op = hawk_dove_op
        p = hawk_dove_params
        assert op.population_size == p["Z"]
        assert op.nb_strategies == 2
        assert op.beta == pytest.approx(p["beta"])
        assert op.mu == pytest.approx(p["mu"])
        # nb_states for Z=10, k=2: C(11,1) = 11
        assert op.size == 11

    def test_properties_3strategy(self, rps_op, rps_params):
        op = rps_op
        p = rps_params
        assert op.nb_strategies == 3
        # nb_states for Z=8, k=3: C(10,2) = 45
        assert op.size == 45

    def test_invalid_population_size(self, hawk_dove_params):
        p = hawk_dove_params
        with pytest.raises(Exception):
            PairwiseComparisonTransitionOperator(1, p["game"], p["beta"], p["mu"])

    def test_invalid_beta(self, hawk_dove_params):
        p = hawk_dove_params
        with pytest.raises(Exception):
            PairwiseComparisonTransitionOperator(p["Z"], p["game"], -0.1, p["mu"])

    def test_invalid_mu(self, hawk_dove_params):
        p = hawk_dove_params
        with pytest.raises(Exception):
            PairwiseComparisonTransitionOperator(p["Z"], p["game"], p["beta"], 1.5)


# ---------------------------------------------------------------------------
# Stochastic-property tests (no ground-truth matrix needed)
# ---------------------------------------------------------------------------

class TestStochasticProperties:
    @pytest.mark.parametrize("fixture_name", ["hawk_dove_op", "rps_op"])
    def test_PT_preserves_sum(self, fixture_name, hawk_dove_op, rps_op):
        """(P^T x).sum() == x.sum() for random x (column-stochastic P)."""
        op = hawk_dove_op if fixture_name == "hawk_dove_op" else rps_op
        rng = np.random.default_rng(0)
        x = rng.random(op.size)
        y = np.zeros(op.size)
        op.apply_transpose(x, y)
        assert y.sum() == pytest.approx(x.sum(), rel=1e-10)

    @pytest.mark.parametrize("fixture_name", ["hawk_dove_op", "rps_op"])
    def test_P_row_stochastic(self, fixture_name, hawk_dove_op, rps_op):
        """P * ones == ones (row-stochastic property)."""
        op = hawk_dove_op if fixture_name == "hawk_dove_op" else rps_op
        x = np.ones(op.size)
        y = np.zeros(op.size)
        op.apply(x, y)
        np.testing.assert_allclose(y, np.ones(op.size), atol=1e-12)

    @pytest.mark.parametrize("fixture_name", ["hawk_dove_op", "rps_op"])
    def test_P_diagonal_is_complement(self, fixture_name, hawk_dove_op, rps_op):
        """For each state i: P[i,i] == 1 - sum_{j!=i} P[i,j] (diagonal is the complement)."""
        op = hawk_dove_op if fixture_name == "hawk_dove_op" else rps_op
        n = op.size
        # Apply to each basis vector e_i: the diagonal element is P[i,i] = (P * e_i)[i]
        # The off-diagonal row sum is sum_{j!=i} P[i,j] = (P * ones)[i] - P[i,i]
        # Row-stochastic: (P * ones)[i] = 1, so P[i,i] = 1 - off-diagonal row sum.
        ones = np.ones(n)
        y_ones = np.zeros(n)
        op.apply(ones, y_ones)  # y_ones[i] = sum_j P[i,j] = 1 for all i
        np.testing.assert_allclose(y_ones, ones, atol=1e-12,
                                   err_msg="Row sums deviate from 1")

        # Check diagonal specifically via basis vectors
        y = np.zeros(n)
        for i in range(n):
            ei = np.zeros(n)
            ei[i] = 1.0
            op.apply(ei, y)
            diag = y[i]  # P[i,i]
            off_diag_sum = y_ones[i] - diag  # sum_{j!=i} P[i,j]
            assert diag == pytest.approx(1.0 - off_diag_sum, abs=1e-12), \
                f"Diagonal P[{i},{i}] = {diag} != 1 - off_diag_sum = {1.0 - off_diag_sum}"

    @pytest.mark.parametrize("fixture_name", ["hawk_dove_op", "rps_op"])
    def test_probability_vector_stays_positive(self, fixture_name, hawk_dove_op, rps_op):
        """P^T applied to a probability vector stays non-negative."""
        op = hawk_dove_op if fixture_name == "hawk_dove_op" else rps_op
        x = np.ones(op.size) / op.size
        y = np.zeros(op.size)
        op.apply_transpose(x, y)
        assert np.all(y >= -1e-14)

    @pytest.mark.parametrize("fixture_name", ["hawk_dove_op", "rps_op"])
    def test_residual_is_x_minus_PT_x(self, fixture_name, hawk_dove_op, rps_op):
        """apply_residual(x) == x - apply_transpose(x) for random x."""
        op = hawk_dove_op if fixture_name == "hawk_dove_op" else rps_op
        rng = np.random.default_rng(2)
        x = rng.random(op.size)
        y_res = np.zeros(op.size)
        y_PT = np.zeros(op.size)
        op.apply_residual(x, y_res)
        op.apply_transpose(x, y_PT)
        np.testing.assert_allclose(y_res, x - y_PT, atol=1e-14)


# ---------------------------------------------------------------------------
# Ground-truth comparison tests (vs. analytical transition matrix)
# ---------------------------------------------------------------------------

class TestAgainstAnalytical:
    def test_PT_columns_match_sparse_2strategy(
        self, hawk_dove_op, hawk_dove_analytical, hawk_dove_params
    ):
        """Column-by-column check: operator P^T == sparse P^T (2-strategy)."""
        p = hawk_dove_params
        sparse_P = hawk_dove_analytical.calculate_transition_matrix(p["beta"], p["mu"])
        PT_dense = sparse_P.toarray().T  # ground-truth P^T as dense

        op_PT = operator_to_dense_PT(hawk_dove_op)
        np.testing.assert_allclose(op_PT, PT_dense, atol=1e-12)

    def test_P_columns_match_sparse_2strategy(
        self, hawk_dove_op, hawk_dove_analytical, hawk_dove_params
    ):
        """Column-by-column check: operator P == sparse P (2-strategy)."""
        p = hawk_dove_params
        sparse_P = hawk_dove_analytical.calculate_transition_matrix(p["beta"], p["mu"])
        P_dense = sparse_P.toarray()

        op_P = operator_to_dense_P(hawk_dove_op)
        np.testing.assert_allclose(op_P, P_dense, atol=1e-12)

    def test_PT_columns_match_sparse_3strategy(
        self, rps_op, rps_analytical, rps_params
    ):
        """Column-by-column check: operator P^T == sparse P^T (3-strategy RPS)."""
        p = rps_params
        sparse_P = rps_analytical.calculate_transition_matrix(p["beta"], p["mu"])
        PT_dense = sparse_P.toarray().T

        op_PT = operator_to_dense_PT(rps_op)
        np.testing.assert_allclose(op_PT, PT_dense, atol=1e-12)

    def _stationary_from_sparse(self, sparse_P):
        """Compute stationary distribution from sparse P via eigenvector of P^T."""
        n = sparse_P.shape[0]
        PT = sparse_P.T.tocsr()
        vals, vecs = eigs(PT, k=1, which="LM", tol=1e-14)
        pi = vecs[:, 0].real
        pi = np.abs(pi)
        pi /= pi.sum()
        return pi

    def test_stationary_distribution_2strategy(
        self, hawk_dove_op, hawk_dove_analytical, hawk_dove_params
    ):
        """Leading eigenvector of P^T via eigs matches analytical stationary dist."""
        p = hawk_dove_params
        # Reference: eigenvector of P^T from the explicit sparse matrix.
        sparse_P = hawk_dove_analytical.calculate_transition_matrix(p["beta"], p["mu"])
        pi_exact = self._stationary_from_sparse(sparse_P)

        # Operator: eigenvector of P^T via matrix-free LinearOperator.
        L = make_transition_operator(hawk_dove_op)
        eigenvalues, eigenvectors = eigs(L, k=1, which="LM", tol=1e-12)
        pi_iter = eigenvectors[:, 0].real
        pi_iter = np.abs(pi_iter)
        pi_iter /= pi_iter.sum()

        np.testing.assert_allclose(pi_iter, pi_exact, atol=1e-6)

    def test_residual_zero_at_stationary(
        self, hawk_dove_op, hawk_dove_analytical, hawk_dove_params
    ):
        """(I - P^T) pi ≈ 0 when pi is the exact stationary distribution."""
        p = hawk_dove_params
        sparse_P = hawk_dove_analytical.calculate_transition_matrix(p["beta"], p["mu"])
        pi = self._stationary_from_sparse(sparse_P)
        y = np.zeros(hawk_dove_op.size)
        hawk_dove_op.apply_residual(pi, y)
        np.testing.assert_allclose(y, 0.0, atol=1e-12)


# ---------------------------------------------------------------------------
# LinearOperator wrapper tests
# ---------------------------------------------------------------------------

class TestPowerIteration:
    """Tests for compute_stationary_distribution (C++ power iteration)."""

    def _reference_stationary(self, analytical, p):
        sparse_P = analytical.calculate_transition_matrix(p["beta"], p["mu"])
        PT = sparse_P.T.tocsr()
        vals, vecs = eigs(PT, k=1, which="LM", tol=1e-14)
        pi = vecs[:, 0].real
        pi = np.abs(pi)
        pi /= pi.sum()
        return pi

    def test_power_iteration_2strategy(self, hawk_dove_op, hawk_dove_analytical, hawk_dove_params):
        """Power iteration result matches sparse eigenvector (2-strategy)."""
        pi_ref = self._reference_stationary(hawk_dove_analytical, hawk_dove_params)
        pi = hawk_dove_op.compute_stationary_distribution(tol=1e-12, max_iter=50000)
        np.testing.assert_allclose(pi, pi_ref, atol=1e-6)

    def test_power_iteration_3strategy(self, rps_op, rps_analytical, rps_params):
        """Power iteration result matches sparse eigenvector (3-strategy RPS)."""
        pi_ref = self._reference_stationary(rps_analytical, rps_params)
        pi = rps_op.compute_stationary_distribution(tol=1e-12, max_iter=50000)
        np.testing.assert_allclose(pi, pi_ref, atol=1e-6)

    def test_residual_zero_at_power_iteration_result(self, hawk_dove_op, hawk_dove_params):
        """(I - P^T) pi ≈ 0 for the power-iteration stationary distribution."""
        pi = hawk_dove_op.compute_stationary_distribution()
        y = np.zeros(hawk_dove_op.size)
        hawk_dove_op.apply_residual(pi, y)
        np.testing.assert_allclose(y, 0.0, atol=1e-9)

    def test_power_iteration_is_normalized(self, hawk_dove_op):
        """Returned distribution sums to 1."""
        pi = hawk_dove_op.compute_stationary_distribution()
        assert pi.sum() == pytest.approx(1.0, rel=1e-12)

    def test_power_iteration_is_nonnegative(self, hawk_dove_op):
        """Returned distribution is non-negative."""
        pi = hawk_dove_op.compute_stationary_distribution()
        assert np.all(pi >= 0.0)

    def test_power_iteration_not_converged_raises(self, hawk_dove_op):
        """RuntimeError raised when max_iter=1 (cannot converge)."""
        with pytest.raises(RuntimeError, match="not converged"):
            hawk_dove_op.compute_stationary_distribution(tol=1e-15, max_iter=1)


class TestARPACK:
    """Tests for compute_stationary_arpack (native C++ ARPACK eigensolver)."""

    _arpack_available = hasattr(PairwiseComparisonTransitionOperator, "compute_stationary_arpack")

    def _reference_stationary(self, analytical, p):
        sparse_P = analytical.calculate_transition_matrix(p["beta"], p["mu"])
        PT = sparse_P.T.tocsr()
        vals, vecs = eigs(PT, k=1, which="LM", tol=1e-14)
        pi = vecs[:, 0].real
        pi = np.abs(pi)
        pi /= pi.sum()
        return pi

    @pytest.mark.skipif(
        not hasattr(PairwiseComparisonTransitionOperator, "compute_stationary_arpack"),
        reason="ARPACK not compiled in (build without EGTTOOLS_ENABLE_ARPACK=ON)"
    )
    def test_arpack_2strategy(self, hawk_dove_op, hawk_dove_analytical, hawk_dove_params):
        """ARPACK result matches sparse eigenvector (2-strategy)."""
        pi_ref = self._reference_stationary(hawk_dove_analytical, hawk_dove_params)
        pi = hawk_dove_op.compute_stationary_arpack()
        np.testing.assert_allclose(pi, pi_ref, atol=1e-8)

    @pytest.mark.skipif(
        not hasattr(PairwiseComparisonTransitionOperator, "compute_stationary_arpack"),
        reason="ARPACK not compiled in (build without EGTTOOLS_ENABLE_ARPACK=ON)"
    )
    def test_arpack_3strategy(self, rps_op, rps_analytical, rps_params):
        """ARPACK result matches sparse eigenvector (3-strategy RPS)."""
        pi_ref = self._reference_stationary(rps_analytical, rps_params)
        pi = rps_op.compute_stationary_arpack()
        np.testing.assert_allclose(pi, pi_ref, atol=1e-8)

    @pytest.mark.skipif(
        not hasattr(PairwiseComparisonTransitionOperator, "compute_stationary_arpack"),
        reason="ARPACK not compiled in"
    )
    def test_arpack_matches_power_iteration(self, hawk_dove_op):
        """ARPACK result agrees with power iteration to high precision."""
        pi_pow = hawk_dove_op.compute_stationary_distribution(tol=1e-12, max_iter=50000)
        pi_arp = hawk_dove_op.compute_stationary_arpack()
        np.testing.assert_allclose(pi_arp, pi_pow, atol=1e-8)

    @pytest.mark.skipif(
        not hasattr(PairwiseComparisonTransitionOperator, "compute_stationary_arpack"),
        reason="ARPACK not compiled in"
    )
    def test_arpack_result_is_normalized(self, hawk_dove_op):
        """ARPACK distribution sums to 1."""
        pi = hawk_dove_op.compute_stationary_arpack()
        assert pi.sum() == pytest.approx(1.0, rel=1e-12)

    @pytest.mark.skipif(
        not hasattr(PairwiseComparisonTransitionOperator, "compute_stationary_arpack"),
        reason="ARPACK not compiled in"
    )
    def test_arpack_residual_near_zero(self, hawk_dove_op):
        """(I - P^T) pi ≈ 0 for ARPACK stationary distribution."""
        pi = hawk_dove_op.compute_stationary_arpack()
        y = np.zeros(hawk_dove_op.size)
        hawk_dove_op.apply_residual(pi, y)
        np.testing.assert_allclose(y, 0.0, atol=1e-8)

    def test_arpack_absent_when_not_compiled(self):
        """When ARPACK is not compiled in, the method is absent (not just None)."""
        if self._arpack_available:
            pytest.skip("ARPACK is compiled in — absence test not applicable")
        assert not hasattr(PairwiseComparisonTransitionOperator, "compute_stationary_arpack")


class TestStationaryDistributionFromSparse:
    """Tests for stationary_distribution_from_sparse helper."""

    def test_2strategy_matches_power_iteration(
        self, hawk_dove_op, hawk_dove_analytical, hawk_dove_params
    ):
        """Result agrees with C++ power iteration (2-strategy)."""
        p = hawk_dove_params
        P = hawk_dove_analytical.calculate_transition_matrix(p["beta"], p["mu"])
        pi_sparse = stationary_distribution_from_sparse(P)
        pi_power = hawk_dove_op.compute_stationary_distribution(tol=1e-12, max_iter=50000)
        np.testing.assert_allclose(pi_sparse, pi_power, atol=1e-6)

    def test_3strategy_matches_power_iteration(
        self, rps_op, rps_analytical, rps_params
    ):
        """Result agrees with C++ power iteration (3-strategy RPS)."""
        p = rps_params
        P = rps_analytical.calculate_transition_matrix(p["beta"], p["mu"])
        pi_sparse = stationary_distribution_from_sparse(P)
        pi_power = rps_op.compute_stationary_distribution(tol=1e-12, max_iter=50000)
        np.testing.assert_allclose(pi_sparse, pi_power, atol=1e-6)

    def test_result_is_normalized(self, hawk_dove_analytical, hawk_dove_params):
        """Returned distribution sums to 1."""
        p = hawk_dove_params
        P = hawk_dove_analytical.calculate_transition_matrix(p["beta"], p["mu"])
        pi = stationary_distribution_from_sparse(P)
        assert pi.sum() == pytest.approx(1.0, rel=1e-12)

    def test_result_is_nonnegative(self, hawk_dove_analytical, hawk_dove_params):
        """Returned distribution is non-negative."""
        p = hawk_dove_params
        P = hawk_dove_analytical.calculate_transition_matrix(p["beta"], p["mu"])
        pi = stationary_distribution_from_sparse(P)
        assert np.all(pi >= 0.0)

    def test_residual_near_zero(
        self, hawk_dove_op, hawk_dove_analytical, hawk_dove_params
    ):
        """(I - P^T) pi ≈ 0 for the result."""
        p = hawk_dove_params
        P = hawk_dove_analytical.calculate_transition_matrix(p["beta"], p["mu"])
        pi = stationary_distribution_from_sparse(P)
        y = np.zeros(hawk_dove_op.size)
        hawk_dove_op.apply_residual(pi, y)
        np.testing.assert_allclose(y, 0.0, atol=1e-10)


class TestLinearOperatorWrappers:
    def test_make_transition_operator_shape(self, hawk_dove_op):
        L = make_transition_operator(hawk_dove_op)
        n = hawk_dove_op.size
        assert L.shape == (n, n)

    def test_make_residual_operator_shape(self, hawk_dove_op):
        A = make_residual_operator(hawk_dove_op)
        n = hawk_dove_op.size
        assert A.shape == (n, n)

    def test_transition_operator_matvec(self, hawk_dove_op):
        """make_transition_operator matvec matches direct apply_transpose."""
        L = make_transition_operator(hawk_dove_op)
        rng = np.random.default_rng(3)
        x = rng.random(hawk_dove_op.size)
        y_wrapper = L @ x
        y_direct = np.zeros(hawk_dove_op.size)
        hawk_dove_op.apply_transpose(x, y_direct)
        np.testing.assert_allclose(y_wrapper, y_direct, atol=1e-14)

    def test_transition_operator_rmatvec(self, hawk_dove_op):
        """make_transition_operator rmatvec matches direct apply."""
        L = make_transition_operator(hawk_dove_op)
        rng = np.random.default_rng(4)
        x = rng.random(hawk_dove_op.size)
        y_wrapper = L.T @ x
        y_direct = np.zeros(hawk_dove_op.size)
        hawk_dove_op.apply(x, y_direct)
        np.testing.assert_allclose(y_wrapper, y_direct, atol=1e-14)

    def test_residual_operator_matvec(self, hawk_dove_op):
        """make_residual_operator matvec matches direct apply_residual."""
        A = make_residual_operator(hawk_dove_op)
        rng = np.random.default_rng(5)
        x = rng.random(hawk_dove_op.size)
        y_wrapper = A @ x
        y_direct = np.zeros(hawk_dove_op.size)
        hawk_dove_op.apply_residual(x, y_direct)
        np.testing.assert_allclose(y_wrapper, y_direct, atol=1e-14)

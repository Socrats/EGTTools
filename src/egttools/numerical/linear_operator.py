"""
Matrix-free linear operator wrappers and stationary-distribution solvers.

Provides:
- Factory functions that wrap a PairwiseComparisonTransitionOperator as
  scipy LinearOperator objects for iterative eigensolvers and Krylov solvers.
- ``stationary_distribution_from_sparse``: convenience solver for an explicit
  sparse transition matrix (e.g. from PairwiseComparison.calculate_transition_matrix).

Example usage
-------------
>>> import numpy as np
>>> from egttools.games import NormalFormGame
>>> from egttools.numerical import PairwiseComparisonTransitionOperator
>>> from egttools.numerical.numerical_ import PairwiseComparison
>>> from egttools.numerical.linear_operator import (
...     make_transition_operator, make_residual_operator,
...     stationary_distribution_from_sparse)
>>>
>>> # Matrix-free path (no Python callbacks)
>>> game = NormalFormGame(...)
>>> op = PairwiseComparisonTransitionOperator(
...         population_size=50, game=game, beta=1.0, mu=0.01)
>>> pi = op.compute_stationary_distribution()          # power iteration
>>> pi = op.compute_stationary_arpack()                # ARPACK (if compiled)
>>>
>>> # Explicit-sparse path (best for moderate state spaces)
>>> pc = PairwiseComparison(50, game)
>>> P  = pc.calculate_transition_matrix(beta=1.0, mu=0.01)
>>> pi = stationary_distribution_from_sparse(P)
"""
from __future__ import annotations

import numpy as np
from scipy.sparse.linalg import LinearOperator, eigs


def make_transition_operator(operator) -> LinearOperator:
    """Wrap a PairwiseComparisonTransitionOperator as a scipy LinearOperator for P^T.

    The returned LinearOperator A satisfies:
    - ``A @ x`` computes ``y = P^T x``  (``matvec``)
    - ``A.T @ x`` computes ``y = P x``  (``rmatvec``)

    The stationary distribution π is the leading eigenvector of P^T
    (eigenvalue = 1). Use ``scipy.sparse.linalg.eigs(A, k=1, which='LM')``
    to find it.

    Parameters
    ----------
    operator : PairwiseComparisonTransitionOperator
        Fully constructed C++ transition operator.

    Returns
    -------
    scipy.sparse.linalg.LinearOperator
        Shape ``(n, n)`` operator where ``n = operator.size``.
    """
    n = operator.size
    buf = np.empty(n, dtype=np.float64)

    def matvec(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        operator.apply_transpose(x, buf)
        return buf.copy()

    def rmatvec(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        operator.apply(x, buf)
        return buf.copy()

    return LinearOperator(
        shape=(n, n),
        matvec=matvec,
        rmatvec=rmatvec,
        dtype=np.float64,
    )


def make_residual_operator(operator) -> LinearOperator:
    """Wrap a PairwiseComparisonTransitionOperator as a scipy LinearOperator for (I - P^T).

    The returned LinearOperator A satisfies:
    - ``A @ x`` computes ``y = (I - P^T) x``  (``matvec``)

    Useful for iterative linear solvers (GMRES, LGMRES) that seek π with
    ``(I - P^T) π = 0``.  Because the system is singular, a normalization
    constraint must be added externally (e.g. replace one equation with
    ``Σ π_i = 1``).

    Parameters
    ----------
    operator : PairwiseComparisonTransitionOperator
        Fully constructed C++ transition operator.

    Returns
    -------
    scipy.sparse.linalg.LinearOperator
        Shape ``(n, n)`` operator where ``n = operator.size``.
    """
    n = operator.size
    buf = np.empty(n, dtype=np.float64)

    def matvec(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        operator.apply_residual(x, buf)
        return buf.copy()

    return LinearOperator(
        shape=(n, n),
        matvec=matvec,
        dtype=np.float64,
    )


def stationary_distribution_from_sparse(
    P,
    tol: float = 1e-12,
    max_iter: int = 1000,
) -> np.ndarray:
    """Compute the stationary distribution of an explicit sparse transition matrix.

    Uses ``scipy.sparse.linalg.eigs`` (ARPACK) on the transpose of *P* to find
    the leading eigenvector, which is the stationary distribution π satisfying
    ``P^T π = π``.

    This is the fastest available local method for moderate state spaces (up to
    the RAM limit for storing P) and is the recommended approach when *P* has
    already been assembled via ``PairwiseComparison.calculate_transition_matrix``.

    Parameters
    ----------
    P : scipy.sparse matrix
        Row-stochastic transition matrix of shape ``(n, n)``.  Typically the
        output of ``PairwiseComparison.calculate_transition_matrix(beta, mu)``.
    tol : float
        ARPACK convergence tolerance (default 1e-12; 0 → machine precision).
    max_iter : int
        Maximum number of ARPACK iterations (default 1000).

    Returns
    -------
    numpy.ndarray
        Normalised stationary distribution of length ``n``, non-negative and
        summing to 1.

    Raises
    ------
    scipy.sparse.linalg.ArpackNoConvergence
        If ARPACK fails to converge within *max_iter* iterations.
    """
    PT = P.T.tocsr()
    vals, vecs = eigs(PT, k=1, which="LM", tol=tol, maxiter=max_iter)
    pi = vecs[:, 0].real
    pi = np.abs(pi)
    pi /= pi.sum()
    return pi

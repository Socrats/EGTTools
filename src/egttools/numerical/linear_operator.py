"""
Matrix-free linear operator wrappers for scipy.sparse.linalg.

Provides factory functions that wrap a PairwiseComparisonTransitionOperator
as scipy LinearOperator objects for use with iterative eigensolvers (ARPACK,
LOBPCG) and Krylov linear solvers (GMRES, LGMRES).

Example usage
-------------
>>> import numpy as np
>>> from egttools.games import NormalFormGame
>>> from egttools.numerical import PairwiseComparisonTransitionOperator
>>> from egttools.numerical.linear_operator import (
...     make_transition_operator, make_residual_operator)
>>> from scipy.sparse.linalg import eigs
>>>
>>> game = NormalFormGame(...)
>>> op = PairwiseComparisonTransitionOperator(
...         population_size=50, game=game, beta=1.0, mu=0.01)
>>> L = make_transition_operator(op)
>>> # Find stationary distribution as leading eigenvector of P^T
>>> eigenvalues, eigenvectors = eigs(L, k=1, which='LM')
>>> pi = eigenvectors[:, 0].real
>>> pi /= pi.sum()
"""
from __future__ import annotations

import numpy as np
from scipy.sparse.linalg import LinearOperator


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

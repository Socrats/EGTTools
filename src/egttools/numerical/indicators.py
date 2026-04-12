"""
Indicator result types and statistics helpers for PairwiseComparisonNumerical.

The high-level ``estimate_stationary_indicators`` method is implemented as a
pybind11 binding directly on the C++ class (see ``cpp/src/pybind11_files/methods.cpp``).
This module supplies:

* ``StationaryIndicatorResult`` — the dataclass returned by that method.
* ``_bootstrap_ci`` — the non-parametric bootstrap helper called from the
  C++ binding (imported at runtime to avoid circular-import issues).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np

__all__ = ["StationaryIndicatorResult"]


@dataclass
class StationaryIndicatorResult:
    """Result of ``PairwiseComparisonNumerical.estimate_stationary_indicators``.

    Attributes
    ----------
    mean : np.ndarray
        Grand mean across all completed runs, shape ``(nb_indicators,)``.
    confidence_interval : tuple[np.ndarray, np.ndarray]
        ``(low, high)`` non-parametric bootstrap confidence interval at the
        requested confidence level, each of shape ``(nb_indicators,)``.
        No normality assumption is made; this is appropriate for skewed or
        bimodal indicator distributions (e.g. rare-event group success).
    nb_runs_used : int
        Number of simulation runs actually completed.  Less than ``nb_runs``
        when tolerance-based early stopping triggered.
    converged : bool
        ``True`` if the simulation stopped early because the L1 norm of the
        change in column-means between consecutive batches fell below
        ``tolerance``.
    per_run_values : np.ndarray or None
        Raw per-run means of shape ``(nb_runs_used, nb_indicators)`` when
        ``verbose=True``, otherwise ``None``.  Use this for custom downstream
        statistics (quantiles, KDE, empirical CDF, etc.).
    """

    mean: np.ndarray
    confidence_interval: tuple[np.ndarray, np.ndarray]
    nb_runs_used: int
    converged: bool
    per_run_values: np.ndarray | None = field(default=None, repr=False)


def _bootstrap_ci(
    per_run: np.ndarray,
    confidence: float,
    n_bootstrap: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Non-parametric percentile bootstrap CI on column means.

    Called from the pybind11 binding of
    ``PairwiseComparisonNumerical.estimate_stationary_indicators``.

    Parameters
    ----------
    per_run : (nb_runs, nb_indicators)
    confidence : float in (0, 1)
    n_bootstrap : int

    Returns
    -------
    (ci_low, ci_high) each of shape (nb_indicators,)
    """
    alpha = 1.0 - confidence
    n = per_run.shape[0]

    if n < 2:
        warnings.warn(
            f"Too few simulation runs (nb_runs_used={n}) to compute a bootstrap "
            "confidence interval.  Returning NaN for CI bounds.",
            RuntimeWarning,
            stacklevel=4,
        )
        nan = np.full(per_run.shape[1], np.nan)
        return nan, nan

    rng = np.random.default_rng()
    indices = rng.integers(0, n, size=(n_bootstrap, n))
    boot_means = per_run[indices].mean(axis=1)  # (n_bootstrap, nb_indicators)

    ci_low = np.percentile(boot_means, 100.0 * alpha / 2.0, axis=0)
    ci_high = np.percentile(boot_means, 100.0 * (1.0 - alpha / 2.0), axis=0)
    return ci_low, ci_high

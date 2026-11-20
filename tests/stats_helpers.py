"""Helpers for statistical tests of Monte Carlo estimators (roadmap TEST-06).

Deliberately independent of egttools: these helpers check the library's estimates (and, later,
its own error estimates), so they must not reuse the code under test.

Comparing an estimate with an exact value
    ``standard_error_iid``           independent draws (e.g. fixation runs, absorption runs).
    ``standard_error_batch_means``   correlated Markov-chain output; flags batches that are too short.
    ``assert_within_k_se``           |estimate - exact| < k * SE, per component, with z-scores in the message.
    ``assert_detects``               power check: a deliberately wrong variant must fall outside k * SE.

Choosing how long to run
    ``integrated_autocorrelation_time``   tau_int from data (Sokal's self-consistent window).
    ``required_run_length``               burn-in and number of steps for a target SE, from a pilot run.

Runs are sized from a pilot run: tau_int and Var(f) from the pilot give n ~ tau_int * Var(f) / SE^2.
A-priori spectral bounds are not used: they need an eigenvalue computation at least as hard as solving
for the stationary distribution itself.

Choice of k: with one comparison, k = 4 gives a false-failure probability of about 6e-5 for a correct
implementation (normal approximation). Checking d components at k = 4 keeps the family-wise rate below
about d * 6e-5, i.e. under 1% up to ~150 components; use k = 5 for larger vectors.

Constants and their sources
    c = 5 (window), reliable if n > 50 tau   Sokal's self-consistent window, as implemented in emcee
                                             (integrated_time(c=5, tol=50)); A. D. Sokal, Cargese lectures
                                             (1996/97); D. Foreman-Mackey et al., PASP 125 (2013) 306.
    batch size b = floor(sqrt(n)), OBM      J. M. Flegal and G. L. Jones, Ann. Statist. 38 (2010) 1034,
                                             Sec. 4.3: b = floor(n^(1/3)) "seemed to give very poor results";
                                             b = floor(n^(1/2)) recommended as default; OBM less variable
                                             than non-overlapping batch means.
    burn-in 20 tau                          W. Janke, Lect. Notes Phys. 739 (2008) 79: wait "at least about
                                             20 tau_exp". tau_exp is not estimated here: 20 * tau_int is used,
                                             conservative when one exponential mode dominates
                                             (tau_int ~ 2 tau_exp with tau_int = 1 + 2 sum rho), but not when
                                             the slowest mode barely couples to the observable.
    sequential fixed-width stopping         G. L. Jones, M. Haran, B. S. Caffo, R. Neath, JASA 101 (2006) 1537.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

DEFAULT_K = 4.0


# --- Standard errors --------------------------------------------------------------------------------

def standard_error_iid(samples):
    """Mean and standard error of independent draws."""
    x = np.asarray(samples, dtype=float)
    if x.size < 2:
        raise ValueError("need at least 2 samples")
    return float(x.mean()), float(x.std(ddof=1) / math.sqrt(x.size))


@dataclass(frozen=True)
class BatchMeans:
    mean: float
    se: float
    batch_size: int
    overlapping: bool


def standard_error_batch_means(series, batch_size=None, overlapping=True):
    """Standard error of the mean of a correlated series by batch means.

    Default batch size b = floor(sqrt(n)) and overlapping batch means (OBM), following Flegal and Jones
    (2010, Sec. 4.3). OBM: sigma^2 = n b / ((n - b)(n - b + 1)) * sum_j (Ybar_j(b) - Ybar)^2 over all n - b + 1
    windows of length b; non-overlapping (BM): sigma^2 = b / (a - 1) * sum_k (Ybar_k - Ybar)^2 over a = n // b
    batches. SE = sqrt(sigma^2 / n).
    """
    x = np.asarray(series, dtype=float)
    n = x.size
    b = int(math.isqrt(n)) if batch_size is None else int(batch_size)
    if b < 1 or n < 2 * b:
        raise ValueError(f"need at least two batches: n = {n}, batch size = {b}")
    mean = float(x.mean())
    if overlapping:
        cumulative = np.concatenate(([0.0], np.cumsum(x)))
        window_means = (cumulative[b:] - cumulative[:-b]) / b
        variance = n * b / ((n - b) * (n - b + 1)) * float(np.sum((window_means - mean) ** 2))
    else:
        a = n // b
        y = x[n - a * b:]
        batch_means = y.reshape(a, b).mean(axis=1)
        mean = float(y.mean())
        variance = b / (a - 1) * float(np.sum((batch_means - mean) ** 2))
    return BatchMeans(mean=mean, se=math.sqrt(variance / n), batch_size=b, overlapping=overlapping)


# --- Assertions -------------------------------------------------------------------------------------

def _z_scores(estimate, exact, se):
    estimate, exact, se = np.broadcast_arrays(*(np.asarray(v, dtype=float) for v in (estimate, exact, se)))
    if np.any(se < 0) or not np.all(np.isfinite(se)):
        raise ValueError("standard errors must be finite and non-negative")
    difference = estimate - exact
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(se > 0, difference / np.where(se > 0, se, 1.0), np.where(difference == 0, 0.0, np.inf))
    return estimate, exact, se, z


def assert_within_k_se(estimate, exact, se, k=DEFAULT_K, label="estimate"):
    """Assert |estimate - exact| < k * se for every component (se = 0 requires equality)."""
    estimate, exact, se, z = _z_scores(estimate, exact, se)
    bad = np.flatnonzero(np.abs(z).ravel() >= k)
    if bad.size:
        worst = bad[np.argsort(-np.abs(z.ravel()[bad]))][:5]
        details = "; ".join(
            f"[{i}] estimate={estimate.ravel()[i]:.6g}, exact={exact.ravel()[i]:.6g}, "
            f"se={se.ravel()[i]:.3g}, z={z.ravel()[i]:+.2f}" for i in worst)
        raise AssertionError(f"{label}: {bad.size} of {z.size} component(s) outside {k} standard errors: {details}")


def assert_detects(wrong_estimate, exact, se, k=DEFAULT_K, label="wrong variant"):
    """Power check: a deliberately wrong variant must fall outside k * se in at least one component."""
    _, _, _, z = _z_scores(wrong_estimate, exact, se)
    if not np.any(np.abs(z) >= k):
        raise AssertionError(f"{label} is within {k} standard errors of the exact value "
                             f"(max |z| = {np.max(np.abs(z)):.2f}): the test cannot detect this error")


# --- Run length -------------------------------------------------------------------------------------

@dataclass(frozen=True)
class AutocorrelationTime:
    tau: float  # integrated autocorrelation time, in steps
    window: int  # summation window chosen by Sokal's rule
    reliable: bool  # series longer than 50 * tau


def integrated_autocorrelation_time(series, c=5.0):
    """Integrated autocorrelation time tau_int = 1 + 2 sum_t rho(t), with Sokal's automatic window.

    The sum stops at the smallest window M with M >= c * tau(M); c = 5 and the reliability criterion
    n > 50 * tau follow Sokal's notes as implemented in emcee (see the module docstring). For an AR(1)
    process with coefficient phi the exact value is (1 + phi) / (1 - phi).
    """
    x = np.asarray(series, dtype=float)
    n = x.size
    if n < 4:
        raise ValueError("series too short")
    centred = x - x.mean()
    size = 1 << (2 * n - 1).bit_length()
    spectrum = np.fft.rfft(centred, size)
    autocovariance = np.fft.irfft(spectrum * np.conj(spectrum), size)[:n] / n
    if autocovariance[0] == 0:
        return AutocorrelationTime(tau=1.0, window=0, reliable=True)
    rho = autocovariance / autocovariance[0]
    taus = 1.0 + 2.0 * np.cumsum(rho[1:])
    windows = np.arange(1, n)
    acceptable = np.flatnonzero(windows >= c * taus)
    window = int(windows[acceptable[0]]) if acceptable.size else n - 1
    tau = float(max(taus[window - 1], 1e-12))
    return AutocorrelationTime(tau=tau, window=window, reliable=n > 50 * tau)


@dataclass(frozen=True)
class RunLength:
    burn_in: int
    nb_steps: int  # steps to record after the burn-in
    tau: float
    variance: float  # stationary variance of the observable, estimated from the pilot


def required_run_length(pilot_series, target_se, burn_in_taus=20.0, safety=1.5):
    """Burn-in and number of recorded steps for a target SE of the mean, from a pilot run.

    The variance of the mean of n correlated samples is about tau_int * Var(f) / n, so
    n = safety * tau_int * Var(f) / target_se^2. The burn-in is ``burn_in_taus`` * tau_int (20, after
    Janke's rule of 20 tau_exp; see the module docstring for when this is conservative). The pilot
    should itself be past its burn-in and longer than ~50 tau_int (see ``integrated_autocorrelation_time``).
    """
    x = np.asarray(pilot_series, dtype=float)
    act = integrated_autocorrelation_time(x)
    if not act.reliable:
        raise ValueError(f"pilot too short for its autocorrelation time (tau = {act.tau:.1f}, "
                         f"length {x.size} < 50 tau); run a longer pilot")
    variance = float(x.var(ddof=1))
    nb_steps = math.ceil(safety * act.tau * variance / target_se ** 2)
    return RunLength(burn_in=math.ceil(burn_in_taus * act.tau), nb_steps=max(nb_steps, 1), tau=act.tau,
                     variance=variance)

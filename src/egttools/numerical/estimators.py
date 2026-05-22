"""
Python wrapper classes for egttools MC estimators.

These wrappers add two things the underlying C++ classes do not provide:

1. **Rich result objects** — every method returns a typed dataclass
   (FixationResult, StrategyDistributionResult, …) that carries the point
   estimate, standard error, 95 % CI, and prints a readable summary.

2. **Optional progress bar** — pass ``verbose=1`` to any estimation method
   and a ``tqdm`` progress bar is shown over the internal chunks.  ``tqdm``
   is an optional dependency; if it is absent the call runs silently.

The wrappers use *chunked execution*: ``nb_runs`` is split into ``n_chunks``
equal pieces, each piece runs as a single C++ call, and the SE is computed as
``std(chunk_means) / sqrt(n_chunks)``.  This gives SE estimates for methods
that otherwise return only a point estimate (e.g.
``PairwiseComparisonNumerical.estimate_fixation_probability``).

All other attributes and methods of the underlying C++ object are accessible
directly via ``__getattr__``, so these wrappers are drop-in replacements.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from .results import (
    AbsorptionProbabilityResult,
    AbsorptionTimeResult,
    AGoSResult,
    FixationResult,
    StationaryDistributionResult,
    StrategyDistributionResult,
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _chunk_iterator(n_chunks: int, verbose: int, desc: str):
    """Yield chunk indices 0..n_chunks-1, optionally wrapped in a tqdm bar."""
    it = range(n_chunks)
    if verbose >= 1:
        try:
            from tqdm import tqdm
            return tqdm(it, desc=desc, unit="chunk")
        except ImportError:
            pass
    return it


def _split_runs(nb_runs: int, n_chunks: int):
    """Split nb_runs into at most n_chunks balanced sizes; return list of ints."""
    n_chunks = max(1, min(n_chunks, nb_runs))
    base, remainder = divmod(nb_runs, n_chunks)
    return [base + (1 if i < remainder else 0) for i in range(n_chunks)]


def _combine_chunks(chunk_means: list, chunk_ses: list | None = None):
    """
    Combine per-chunk (mean, se) arrays.
    SE is the maximum of between-chunk variation and mean within-chunk SE.
    """
    stack = np.stack(chunk_means)
    mean = stack.mean(axis=0)
    n = len(chunk_means)
    between = stack.std(axis=0, ddof=1) / np.sqrt(n) if n > 1 else np.zeros_like(mean)
    if chunk_ses:
        within = np.stack(chunk_ses).mean(axis=0)
        se = np.maximum(between, within)
    else:
        se = between
    return mean, se


# ---------------------------------------------------------------------------
# PairwiseComparisonEstimator
# ---------------------------------------------------------------------------

class PairwiseComparisonEstimator:
    """
    Wraps :class:`egttools.numerical.PairwiseComparisonNumerical` and returns
    rich result objects with standard errors and optional progress bars.

    Parameters
    ----------
    game : AbstractGame
        The game to associate with the estimator.  ``nb_strategies`` is read
        from the game object.
    pop_size : int
        Population size.
    cache_size : int, optional
        LRU cache size for fitness computations (default 100 000).
    strategy_names : list[str], optional
        Human-readable strategy names used in result reprs.
    """

    def __init__(self, game, pop_size: int,
                 cache_size: int = 100_000,
                 strategy_names: Optional[Sequence[str]] = None):
        from egttools.numerical import PairwiseComparisonNumerical
        self._est = PairwiseComparisonNumerical(pop_size, game, cache_size)
        self._strategy_names = list(strategy_names) if strategy_names else None

    def __getattr__(self, name):
        return getattr(self._est, name)

    # -------------------------------------------------------------------------

    def estimate_fixation_probability(
        self,
        invader: int,
        resident: int,
        nb_runs: int,
        nb_generations: int,
        beta: float,
        verbose: int = 0,
        n_chunks: int = 10,
    ) -> FixationResult:
        sizes = _split_runs(nb_runs, n_chunks)
        estimates = []
        for idx in _chunk_iterator(len(sizes), verbose, "estimate_fixation_probability"):
            p = self._est.estimate_fixation_probability(
                invader, resident, sizes[idx], nb_generations, beta
            )
            estimates.append(float(p))
        mean, se = float(np.mean(estimates)), 0.0
        if len(estimates) > 1:
            se = float(np.std(estimates, ddof=1) / np.sqrt(len(estimates)))
        result = FixationResult(estimate=mean, stderr=se, nb_runs=nb_runs,
                                invader=invader, resident=resident)
        if verbose >= 1:
            print(result)
        return result

    def estimate_mean_absorption_time(
        self,
        beta: float,
        init_state,
        nb_runs: int,
        verbose: int = 0,
        n_chunks: int = 10,
    ) -> AbsorptionTimeResult:
        sizes = _split_runs(nb_runs, n_chunks)
        means = []
        for idx in _chunk_iterator(len(sizes), verbose, "estimate_mean_absorption_time"):
            d = self._est.estimate_mean_absorption_time(beta, init_state, sizes[idx])
            means.append(float(d["mean"]))
        m = float(np.mean(means))
        se = float(np.std(means, ddof=1) / np.sqrt(len(means))) if len(means) > 1 else 0.0
        result = AbsorptionTimeResult(mean=m, stderr=se, nb_runs=nb_runs)
        if verbose >= 1:
            print(result)
        return result

    def estimate_absorption_probabilities(
        self,
        beta: float,
        init_state,
        nb_runs: int,
        verbose: int = 0,
        n_chunks: int = 10,
    ) -> AbsorptionProbabilityResult:
        sizes = _split_runs(nb_runs, n_chunks)
        chunks = []
        for idx in _chunk_iterator(len(sizes), verbose, "estimate_absorption_probabilities"):
            v = np.asarray(
                self._est.estimate_absorption_probabilities(beta, init_state, sizes[idx]),
                dtype=float,
            )
            chunks.append(v)
        mean, se = _combine_chunks(chunks)
        result = AbsorptionProbabilityResult(
            probabilities=mean, stderr=se, nb_runs=nb_runs,
            strategy_names=self._strategy_names,
        )
        if verbose >= 1:
            print(result)
        return result

    def estimate_stationary_distribution(
        self,
        nb_runs: int,
        nb_generations: int,
        transitory: int,
        beta: float,
        mu: float,
        verbose: int = 0,
        n_chunks: int = 10,
        **kwargs,
    ) -> StationaryDistributionResult:
        sizes = _split_runs(nb_runs, n_chunks)
        chunks = []
        for idx in _chunk_iterator(len(sizes), verbose, "estimate_stationary_distribution"):
            d = np.asarray(
                self._est.estimate_stationary_distribution(
                    sizes[idx], nb_generations, transitory, beta, mu, **kwargs
                ),
                dtype=float,
            )
            chunks.append(d)
        distribution = np.stack(chunks).mean(axis=0)
        result = StationaryDistributionResult(distribution=distribution, nb_runs=nb_runs)
        if verbose >= 1:
            print(result)
        return result

    def estimate_strategy_distribution(
        self,
        nb_runs: int,
        nb_generations: int,
        transitory: int,
        beta: float,
        mu: float,
        verbose: int = 0,
        n_chunks: int = 10,
        **kwargs,
    ) -> StrategyDistributionResult:
        sizes = _split_runs(nb_runs, n_chunks)
        chunks = []
        for idx in _chunk_iterator(len(sizes), verbose, "estimate_strategy_distribution"):
            v = np.asarray(
                self._est.estimate_strategy_distribution(
                    sizes[idx], nb_generations, transitory, beta, mu, **kwargs
                ),
                dtype=float,
            )
            chunks.append(v)
        mean, se = _combine_chunks(chunks)
        result = StrategyDistributionResult(
            mean=mean, stderr=se, nb_runs=nb_runs,
            strategy_names=self._strategy_names,
        )
        if verbose >= 1:
            print(result)
        return result


# ---------------------------------------------------------------------------
# NetworkEstimator
# ---------------------------------------------------------------------------

class NetworkEstimator:
    """
    Wraps any ``NetworkMCEstimator*`` C++ instance and returns rich result
    objects with optional progress bars.

    Parameters
    ----------
    cpp_estimator
        A C++ estimator returned by e.g. ``network_mc_estimator_factory()``.
    strategy_names : list[str], optional
        Human-readable strategy names used in result reprs.
    """

    def __init__(self, cpp_estimator, strategy_names: Optional[Sequence[str]] = None):
        self._est = cpp_estimator
        self._strategy_names = list(strategy_names) if strategy_names else None

    def __getattr__(self, name):
        return getattr(self._est, name)

    # -------------------------------------------------------------------------

    def estimate_fixation_probability(
        self,
        invader: int,
        resident: int,
        nb_runs: int,
        nb_generations: int,
        verbose: int = 0,
        n_chunks: int = 10,
    ) -> FixationResult:
        sizes = _split_runs(nb_runs, n_chunks)
        estimates = []
        for idx in _chunk_iterator(len(sizes), verbose, "estimate_fixation_probability"):
            p = self._est.estimate_fixation_probability(
                invader, resident, sizes[idx], nb_generations
            )
            estimates.append(float(p))
        mean, se = float(np.mean(estimates)), 0.0
        if len(estimates) > 1:
            se = float(np.std(estimates, ddof=1) / np.sqrt(len(estimates)))
        result = FixationResult(estimate=mean, stderr=se, nb_runs=nb_runs,
                                invader=invader, resident=resident)
        if verbose >= 1:
            print(result)
        return result

    def estimate_strategy_distribution(
        self,
        nb_runs: int,
        nb_generations: int,
        transitory: int = 0,
        verbose: int = 0,
        n_chunks: int = 10,
        **kwargs,
    ) -> StrategyDistributionResult:
        sizes = _split_runs(nb_runs, n_chunks)
        chunk_means, chunk_ses = [], []
        for idx in _chunk_iterator(len(sizes), verbose, "estimate_strategy_distribution"):
            m, s = self._est.estimate_strategy_distribution(
                sizes[idx], nb_generations, transitory, **kwargs
            )
            chunk_means.append(np.asarray(m, dtype=float))
            chunk_ses.append(np.asarray(s, dtype=float))
        mean, se = _combine_chunks(chunk_means, chunk_ses)
        result = StrategyDistributionResult(
            mean=mean, stderr=se, nb_runs=nb_runs,
            strategy_names=self._strategy_names,
        )
        if verbose >= 1:
            print(result)
        return result

    def estimate_agos(
        self,
        nb_runs: int,
        nb_generations: int,
        transitory: int = 0,
        runs_per_j: int = 0,
        verbose: int = 0,
        n_chunks: int = 10,
    ) -> AGoSResult:
        if runs_per_j > 0:
            # runs_per_j mode: nb_runs is ignored by C++; chunk by repeating
            # runs_per_j calls n_chunks times and averaging.
            chunk_means, chunk_ses = [], []
            for _ in _chunk_iterator(n_chunks, verbose, "estimate_agos"):
                m, s = self._est.estimate_agos(0, nb_generations, transitory, runs_per_j)
                chunk_means.append(np.asarray(m, dtype=float))
                chunk_ses.append(np.asarray(s, dtype=float))
            total_runs = runs_per_j * n_chunks
        else:
            sizes = _split_runs(nb_runs, n_chunks)
            chunk_means, chunk_ses = [], []
            for idx in _chunk_iterator(len(sizes), verbose, "estimate_agos"):
                m, s = self._est.estimate_agos(sizes[idx], nb_generations, transitory, 0)
                chunk_means.append(np.asarray(m, dtype=float))
                chunk_ses.append(np.asarray(s, dtype=float))
            total_runs = nb_runs

        mean, se = _combine_chunks(chunk_means, chunk_ses)
        result = AGoSResult(
            mean_gradient=mean, se_gradient=se, nb_runs=total_runs,
            strategy_names=self._strategy_names,
        )
        if verbose >= 1:
            print(result)
        return result

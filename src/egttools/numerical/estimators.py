"""
Python wrapper classes for egttools MC estimators.

These wrappers add three things the underlying C++ classes do not provide:

1. **Rich result objects** — every method returns a typed dataclass
   (FixationResult, StrategyDistributionResult, …) that carries the point
   estimate, standard error, 95 % CI, and prints a readable summary.

2. **Optional progress bar** — pass ``verbose=1`` to any estimation method
   and a ``tqdm`` progress bar is shown over the internal chunks.  ``tqdm``
   is an optional dependency; if it is absent the call runs silently.

3. **Optional checkpointing** — pass ``checkpoint_path=<file>`` to any
   estimation method and progress is persisted to disk after every chunk
   (atomically, so a kill mid-write can't corrupt it). If the same
   ``checkpoint_path`` is passed again with identical parameters, the call
   resumes from the last completed chunk instead of starting over — useful
   for long runs on clusters with job time limits. Resuming is statistically
   independent (never repeats a chunk's random draws) and, if
   ``egttools.Random`` state capture succeeds, bit-exact — see
   ``egttools.Random.get_state``/``set_state``. Passing a ``checkpoint_path``
   with parameters that don't match an existing checkpoint raises rather
   than silently resuming incorrectly. The checkpoint file is removed once
   the call completes normally, so a later call with the same path starts
   fresh again.

The wrappers use *chunked execution*: ``nb_runs`` is split into ``n_chunks``
equal pieces, each piece runs as a single C++ call, and the SE is computed as
``std(chunk_means) / sqrt(n_chunks)``.  This gives SE estimates for methods
that otherwise return only a point estimate (e.g.
``PairwiseComparisonNumerical.estimate_fixation_probability``), and doubles
as the natural checkpoint boundary described above.

All other attributes and methods of the underlying C++ object are accessible
directly via ``__getattr__``, so these wrappers are drop-in replacements.
"""

from __future__ import annotations

import hashlib
import os
import pickle
import warnings
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

def _chunk_iterator(iterable, verbose: int, desc: str):
    """Wrap `iterable` in a tqdm progress bar when verbose>=1, else iterate as-is."""
    if verbose >= 1:
        try:
            from tqdm import tqdm
            return tqdm(iterable, desc=desc, unit="chunk")
        except ImportError:
            pass
    return iterable


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


def _params_key(*args, **kwargs) -> str:
    """Stable hash of a call's defining parameters, used to detect a
    mismatched/stale checkpoint (e.g. resuming with different nb_generations)."""
    payload = repr((args, sorted(kwargs.items())))
    return hashlib.sha256(payload.encode()).hexdigest()


def _rng_get_state():
    """Best-effort capture of the global RNG state for the checkpoint payload.
    Never fails the caller — checkpointing degrades gracefully to
    statistically-independent-but-not-bit-exact resume if this is unavailable."""
    try:
        import egttools
        return egttools.Random.get_state()
    except Exception:
        return None


def _rng_set_state(state) -> None:
    if state is None:
        return
    try:
        import egttools
        egttools.Random.set_state(state)
    except Exception:
        pass


def _load_checkpoint(checkpoint_path, params_key):
    if checkpoint_path is None or not os.path.exists(checkpoint_path):
        return None
    with open(checkpoint_path, "rb") as f:
        data = pickle.load(f)
    if data.get("params_key") != params_key:
        raise ValueError(
            f"Checkpoint at {checkpoint_path!r} was created with different "
            "parameters than this call (nb_runs/nb_generations/beta/mu/... "
            "differ). Refusing to resume from a mismatched checkpoint — use "
            "a different checkpoint_path, or delete the existing file if "
            "starting over is intentional."
        )
    return data


def _save_checkpoint(checkpoint_path, params_key, chunk_results, next_chunk_idx, rng_state) -> None:
    """Atomically persist checkpoint state: write to a temp file then
    os.replace into place, so a kill mid-write can't corrupt it."""
    data = {
        "params_key": params_key,
        "chunk_results": chunk_results,
        "next_chunk_idx": next_chunk_idx,
        "rng_state": rng_state,
    }
    tmp_path = f"{checkpoint_path}.tmp"
    with open(tmp_path, "wb") as f:
        pickle.dump(data, f)
    os.replace(tmp_path, checkpoint_path)


def _run_chunks_with_checkpoint(
    compute_chunk,
    sizes,
    verbose: int,
    desc: str,
    checkpoint_path=None,
    params_key=None,
):
    """
    Calls ``compute_chunk(sizes[idx])`` once per chunk index, returning the
    list of per-chunk results in the same order as `sizes`.

    When `checkpoint_path` is given, persists progress atomically after every
    chunk and resumes from an existing, matching checkpoint instead of
    starting over — see the module docstring's "Optional checkpointing"
    section. The checkpoint file is removed once all chunks complete.
    """
    if checkpoint_path is None:
        return [compute_chunk(sizes[idx]) for idx in _chunk_iterator(range(len(sizes)), verbose, desc)]

    if len(sizes) < 5:
        warnings.warn(
            f"checkpoint_path is set but n_chunks resolved to only {len(sizes)} "
            "chunk(s) — a single giant chunk gives checkpointing nothing to "
            "resume from until it finishes. Consider a larger n_chunks.",
            RuntimeWarning,
            stacklevel=3,
        )

    checkpoint = _load_checkpoint(checkpoint_path, params_key)
    if checkpoint is not None:
        chunk_results = list(checkpoint["chunk_results"])
        start_idx = checkpoint["next_chunk_idx"]
        _rng_set_state(checkpoint.get("rng_state"))
    else:
        chunk_results = []
        start_idx = 0

    for idx in _chunk_iterator(range(start_idx, len(sizes)), verbose, desc):
        chunk_results.append(compute_chunk(sizes[idx]))
        _save_checkpoint(checkpoint_path, params_key, chunk_results, idx + 1, _rng_get_state())

    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)

    return chunk_results


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
    mutation_weights : array_like, optional
        Biases the mutation kernel away from uniform. Either a 1D array of
        length ``nb_strategies`` (target-strategy bias shared by all source
        strategies) or a 2D array of shape ``(nb_strategies, nb_strategies)``
        (row ``i`` is the bias over target strategies when mutating away
        from strategy ``i``). See
        ``PairwiseComparisonNumerical.set_mutation_weights`` for details.
        Defaults to uniform mutation when omitted.
    """

    def __init__(self, game, pop_size: int,
                 cache_size: int = 100_000,
                 strategy_names: Optional[Sequence[str]] = None,
                 mutation_weights=None):
        from egttools.numerical import PairwiseComparisonNumerical
        self._est = PairwiseComparisonNumerical(pop_size, game, cache_size)
        self._strategy_names = list(strategy_names) if strategy_names else None
        if mutation_weights is not None:
            self._est.set_mutation_weights(mutation_weights)

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
        checkpoint_path: Optional[str] = None,
    ) -> FixationResult:
        sizes = _split_runs(nb_runs, n_chunks)
        params_key = _params_key(
            "estimate_fixation_probability", invader, resident, nb_runs, nb_generations, beta, n_chunks
        )
        estimates = _run_chunks_with_checkpoint(
            lambda size: float(self._est.estimate_fixation_probability(
                invader, resident, size, nb_generations, beta
            )),
            sizes, verbose, "estimate_fixation_probability",
            checkpoint_path=checkpoint_path, params_key=params_key,
        )
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
        checkpoint_path: Optional[str] = None,
    ) -> AbsorptionTimeResult:
        sizes = _split_runs(nb_runs, n_chunks)
        params_key = _params_key(
            "estimate_mean_absorption_time", beta, tuple(np.asarray(init_state).tolist()), nb_runs, n_chunks
        )
        means = _run_chunks_with_checkpoint(
            lambda size: float(self._est.estimate_mean_absorption_time(beta, init_state, size)["mean"]),
            sizes, verbose, "estimate_mean_absorption_time",
            checkpoint_path=checkpoint_path, params_key=params_key,
        )
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
        checkpoint_path: Optional[str] = None,
    ) -> AbsorptionProbabilityResult:
        sizes = _split_runs(nb_runs, n_chunks)
        params_key = _params_key(
            "estimate_absorption_probabilities", beta, tuple(np.asarray(init_state).tolist()), nb_runs, n_chunks
        )
        chunks = _run_chunks_with_checkpoint(
            lambda size: np.asarray(
                self._est.estimate_absorption_probabilities(beta, init_state, size), dtype=float
            ),
            sizes, verbose, "estimate_absorption_probabilities",
            checkpoint_path=checkpoint_path, params_key=params_key,
        )
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
        checkpoint_path: Optional[str] = None,
        **kwargs,
    ) -> StationaryDistributionResult:
        sizes = _split_runs(nb_runs, n_chunks)
        params_key = _params_key(
            "estimate_stationary_distribution", nb_runs, nb_generations, transitory, beta, mu, n_chunks, **kwargs
        )
        chunks = _run_chunks_with_checkpoint(
            lambda size: np.asarray(
                self._est.estimate_stationary_distribution(
                    size, nb_generations, transitory, beta, mu, **kwargs
                ),
                dtype=float,
            ),
            sizes, verbose, "estimate_stationary_distribution",
            checkpoint_path=checkpoint_path, params_key=params_key,
        )
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
        checkpoint_path: Optional[str] = None,
        **kwargs,
    ) -> StrategyDistributionResult:
        sizes = _split_runs(nb_runs, n_chunks)
        params_key = _params_key(
            "estimate_strategy_distribution", nb_runs, nb_generations, transitory, beta, mu, n_chunks, **kwargs
        )
        chunks = _run_chunks_with_checkpoint(
            lambda size: np.asarray(
                self._est.estimate_strategy_distribution(
                    size, nb_generations, transitory, beta, mu, **kwargs
                ),
                dtype=float,
            ),
            sizes, verbose, "estimate_strategy_distribution",
            checkpoint_path=checkpoint_path, params_key=params_key,
        )
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
        checkpoint_path: Optional[str] = None,
    ) -> FixationResult:
        sizes = _split_runs(nb_runs, n_chunks)
        params_key = _params_key(
            "network_estimate_fixation_probability", invader, resident, nb_runs, nb_generations, n_chunks
        )
        estimates = _run_chunks_with_checkpoint(
            lambda size: float(self._est.estimate_fixation_probability(
                invader, resident, size, nb_generations
            )),
            sizes, verbose, "estimate_fixation_probability",
            checkpoint_path=checkpoint_path, params_key=params_key,
        )
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
        checkpoint_path: Optional[str] = None,
        **kwargs,
    ) -> StrategyDistributionResult:
        sizes = _split_runs(nb_runs, n_chunks)
        params_key = _params_key(
            "network_estimate_strategy_distribution", nb_runs, nb_generations, transitory, n_chunks, **kwargs
        )
        chunk_pairs = _run_chunks_with_checkpoint(
            lambda size: tuple(
                np.asarray(x, dtype=float)
                for x in self._est.estimate_strategy_distribution(size, nb_generations, transitory, **kwargs)
            ),
            sizes, verbose, "estimate_strategy_distribution",
            checkpoint_path=checkpoint_path, params_key=params_key,
        )
        chunk_means = [m for m, _ in chunk_pairs]
        chunk_ses = [s for _, s in chunk_pairs]
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
        checkpoint_path: Optional[str] = None,
    ) -> AGoSResult:
        if runs_per_j > 0:
            # runs_per_j mode: nb_runs is ignored by C++; chunk by repeating
            # runs_per_j calls n_chunks times and averaging.
            sizes = [runs_per_j] * n_chunks
            params_key = _params_key(
                "network_estimate_agos_per_j", nb_generations, transitory, runs_per_j, n_chunks
            )
            chunk_pairs = _run_chunks_with_checkpoint(
                lambda size: tuple(
                    np.asarray(x, dtype=float)
                    for x in self._est.estimate_agos(0, nb_generations, transitory, size)
                ),
                sizes, verbose, "estimate_agos",
                checkpoint_path=checkpoint_path, params_key=params_key,
            )
            total_runs = runs_per_j * n_chunks
        else:
            sizes = _split_runs(nb_runs, n_chunks)
            params_key = _params_key(
                "network_estimate_agos", nb_runs, nb_generations, transitory, n_chunks
            )
            chunk_pairs = _run_chunks_with_checkpoint(
                lambda size: tuple(
                    np.asarray(x, dtype=float)
                    for x in self._est.estimate_agos(size, nb_generations, transitory, 0)
                ),
                sizes, verbose, "estimate_agos",
                checkpoint_path=checkpoint_path, params_key=params_key,
            )
            total_runs = nb_runs

        chunk_means = [m for m, _ in chunk_pairs]
        chunk_ses = [s for _, s in chunk_pairs]
        mean, se = _combine_chunks(chunk_means, chunk_ses)
        result = AGoSResult(
            mean_gradient=mean, se_gradient=se, nb_runs=total_runs,
            strategy_names=self._strategy_names,
        )
        if verbose >= 1:
            print(result)
        return result

"""
Rich result objects returned by egttools MC estimator wrappers.

Each object carries the point estimate, standard error, number of runs used,
and a ``ci95`` property (95 % CI via z = 1.96).  All objects print a
human-readable summary via ``__repr__``, which makes them convenient both in
interactive sessions and in log output.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence, Tuple

import numpy as np

_Z95 = 1.959964  # scipy.stats.norm.ppf(0.975) — avoid runtime scipy dep


# ---------------------------------------------------------------------------
# FixationResult
# ---------------------------------------------------------------------------

@dataclass
class FixationResult:
    """Fixation probability of an invading strategy into a resident population."""
    estimate: float
    stderr: float
    nb_runs: int
    invader: int = -1
    resident: int = -1

    @property
    def ci95(self) -> Tuple[float, float]:
        w = _Z95 * self.stderr
        return (self.estimate - w, self.estimate + w)

    def __repr__(self) -> str:
        lo, hi = self.ci95
        lines = ["FixationProbability"]
        if self.invader >= 0:
            lines.append(f"  invader  : {self.invader}  →  resident : {self.resident}")
        lines.append(f"  estimate : {self.estimate:.6f}")
        lines.append(f"  std_err  : {self.stderr:.6f}")
        lines.append(f"  95% CI   : [{lo:.6f}, {hi:.6f}]")
        lines.append(f"  nb_runs  : {self.nb_runs:,}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# AbsorptionTimeResult
# ---------------------------------------------------------------------------

@dataclass
class AbsorptionTimeResult:
    """Mean absorption (fixation) time from a given initial state."""
    mean: float
    stderr: float
    nb_runs: int

    @property
    def ci95(self) -> Tuple[float, float]:
        w = _Z95 * self.stderr
        return (self.mean - w, self.mean + w)

    def __repr__(self) -> str:
        lo, hi = self.ci95
        return "\n".join([
            "MeanAbsorptionTime",
            f"  mean     : {self.mean:.2f}",
            f"  std_err  : {self.stderr:.4f}",
            f"  95% CI   : [{lo:.2f}, {hi:.2f}]",
            f"  nb_runs  : {self.nb_runs:,}",
        ])


# ---------------------------------------------------------------------------
# AbsorptionProbabilityResult
# ---------------------------------------------------------------------------

@dataclass
class AbsorptionProbabilityResult:
    """Per-strategy fixation (absorption) probabilities."""
    probabilities: np.ndarray
    stderr: np.ndarray
    nb_runs: int
    strategy_names: Optional[Sequence[str]] = field(default=None, repr=False)

    @property
    def ci95(self) -> Tuple[np.ndarray, np.ndarray]:
        w = _Z95 * self.stderr
        return (self.probabilities - w, self.probabilities + w)

    def __repr__(self) -> str:
        K = len(self.probabilities)
        names = list(self.strategy_names) if self.strategy_names else [f"s{i}" for i in range(K)]
        lo, hi = self.ci95
        w = max(len(n) for n in names)
        header = f"  {'strategy':<{w}}   prob       std_err    95% CI"
        sep = "  " + "-" * (len(header) - 2)
        rows = [
            f"  {n:<{w}}   {p:.6f}   {s:.6f}   [{l:.6f}, {h:.6f}]"
            for n, p, s, l, h in zip(names, self.probabilities, self.stderr, lo, hi)
        ]
        return "\n".join(["AbsorptionProbabilities", header, sep] + rows
                         + [f"  nb_runs  : {self.nb_runs:,}"])


# ---------------------------------------------------------------------------
# StrategyDistributionResult
# ---------------------------------------------------------------------------

@dataclass
class StrategyDistributionResult:
    """Time-averaged strategy frequencies with uncertainty."""
    mean: np.ndarray
    stderr: np.ndarray
    nb_runs: int
    strategy_names: Optional[Sequence[str]] = field(default=None, repr=False)

    @property
    def ci95(self) -> Tuple[np.ndarray, np.ndarray]:
        w = _Z95 * self.stderr
        return (self.mean - w, self.mean + w)

    def __repr__(self) -> str:
        K = len(self.mean)
        names = list(self.strategy_names) if self.strategy_names else [f"s{i}" for i in range(K)]
        lo, hi = self.ci95
        w = max(len(n) for n in names)
        header = f"  {'strategy':<{w}}   mean       std_err    95% CI"
        sep = "  " + "-" * (len(header) - 2)
        rows = [
            f"  {n:<{w}}   {m:.6f}   {s:.6f}   [{l:.6f}, {h:.6f}]"
            for n, m, s, l, h in zip(names, self.mean, self.stderr, lo, hi)
        ]
        return "\n".join(["StrategyDistribution", header, sep] + rows
                         + [f"  nb_runs  : {self.nb_runs:,}"])


# ---------------------------------------------------------------------------
# StationaryDistributionResult
# ---------------------------------------------------------------------------

@dataclass
class StationaryDistributionResult:
    """Estimated stationary distribution over population states."""
    distribution: np.ndarray
    nb_runs: int

    def __repr__(self) -> str:
        n = len(self.distribution)
        nonzero = int(np.count_nonzero(self.distribution > 1e-9))
        peak = int(np.argmax(self.distribution))
        return "\n".join([
            "StationaryDistribution",
            f"  nb_states : {n:,}",
            f"  nonzero   : {nonzero:,}",
            f"  peak_state: {peak}",
            f"  nb_runs   : {self.nb_runs:,}",
        ])


# ---------------------------------------------------------------------------
# AGoSResult
# ---------------------------------------------------------------------------

@dataclass
class AGoSResult:
    """Average gradient of selection G^A(x) from a network MC estimator."""
    mean_gradient: np.ndarray   # shape (N+1, nb_strategies)
    se_gradient: np.ndarray     # shape (N+1, nb_strategies)
    nb_runs: int
    strategy_names: Optional[Sequence[str]] = field(default=None, repr=False)

    @property
    def roots(self) -> list:
        """Zero-crossings of mean_gradient[:, 0] (first strategy's gradient)."""
        G = self.mean_gradient[:, 0]
        N = len(G) - 1
        x = np.arange(N + 1) / N
        result = []
        for i in range(len(G) - 1):
            gi, gi1 = G[i], G[i + 1]
            if not (np.isnan(gi) or np.isnan(gi1)) and gi * gi1 < 0:
                xi = x[i] - gi * (x[i + 1] - x[i]) / (gi1 - gi)
                result.append(float(xi))
        return result

    def __repr__(self) -> str:
        N = len(self.mean_gradient) - 1
        roots = self.roots
        lines = [
            "AverageGradientOfSelection",
            f"  N        : {N}",
            f"  nb_runs  : {self.nb_runs:,}",
        ]
        if roots:
            lines.append("  roots    : " + ", ".join(f"{r:.4f}" for r in roots))
        else:
            lines.append("  roots    : none detected")
        return "\n".join(lines)

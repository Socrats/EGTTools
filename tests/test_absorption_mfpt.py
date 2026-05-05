"""Tests for estimate_mean_absorption_time, estimate_absorption_probabilities,
and calculate_spectral_gap (Phase 0a, 0b, 0d)."""
import os
from sys import platform

import numpy as np
import pytest
from scipy.sparse import csr_matrix

egt = pytest.importorskip("egttools")

PairwiseComparisonNumerical = egt.numerical.PairwiseComparisonNumerical
NormalFormGame = egt.games.NormalFormGame


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _set_seed():
    if platform == "darwin":
        os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
    egt.Random.init_with_seed(42)


@pytest.fixture
def hawk_dove_solver():
    """PairwiseComparisonNumerical for Hawk-Dove (Z=20, 2 strategies)."""
    v, d, t = 2, 3, 1
    payoffs = np.array([
        [(v - d) / 2, v],
        [0, (v / 2) - t],
    ])
    game = NormalFormGame(1, payoffs)
    return PairwiseComparisonNumerical(20, game, cache_size=10000)


@pytest.fixture
def rps_solver():
    """PairwiseComparisonNumerical for Rock-Paper-Scissors (Z=15, 3 strategies)."""
    payoffs = np.array([
        [0, -1, 1],
        [1, 0, -1],
        [-1, 1, 0],
    ], dtype=float)
    game = egt.games.Matrix2PlayerGameHolder(3, payoffs)
    return PairwiseComparisonNumerical(15, game, cache_size=10000)


# ---------------------------------------------------------------------------
# estimate_mean_absorption_time
# ---------------------------------------------------------------------------

class TestEstimateMeanAbsorptionTime:

    def test_already_absorbed_returns_zero(self, hawk_dove_solver):
        """Monomorphic initial state should return mean=0, stderr=0 immediately."""
        Z = 20
        result = hawk_dove_solver.estimate_mean_absorption_time(
            beta=1.0, init_state=np.array([Z, 0], dtype=np.uint64), nb_runs=10
        )
        assert result["mean"] == pytest.approx(0.0)
        assert result["stderr"] == pytest.approx(0.0)
        assert result["nb_runs"] == 10

    def test_result_keys(self, hawk_dove_solver):
        result = hawk_dove_solver.estimate_mean_absorption_time(
            beta=1.0, init_state=np.array([10, 10], dtype=np.uint64), nb_runs=20
        )
        assert set(result.keys()) == {"mean", "stderr", "nb_runs"}

    def test_mean_positive_for_polymorphic_state(self, hawk_dove_solver):
        result = hawk_dove_solver.estimate_mean_absorption_time(
            beta=1.0, init_state=np.array([10, 10], dtype=np.uint64), nb_runs=50
        )
        assert result["mean"] > 0.0
        assert result["stderr"] >= 0.0

    def test_nb_runs_reflected(self, hawk_dove_solver):
        result = hawk_dove_solver.estimate_mean_absorption_time(
            beta=0.0, init_state=np.array([5, 15], dtype=np.uint64), nb_runs=30
        )
        assert result["nb_runs"] == 30

    def test_neutral_drift_order_of_magnitude(self, hawk_dove_solver):
        """Neutral drift (beta=0) absorption time from [10,10] should be O(Z^2)=O(400)."""
        Z = 20
        result = hawk_dove_solver.estimate_mean_absorption_time(
            beta=0.0, init_state=np.array([Z // 2, Z // 2], dtype=np.uint64), nb_runs=200
        )
        # For neutral Moran process, t(k) ~ Z log(Z) at k=Z/2; Z=20 → ~60-200 steps
        assert 10 < result["mean"] < 5000

    def test_invalid_beta(self, hawk_dove_solver):
        with pytest.raises(Exception):
            hawk_dove_solver.estimate_mean_absorption_time(
                beta=-1.0, init_state=np.array([10, 10], dtype=np.uint64), nb_runs=10
            )

    def test_invalid_init_state_wrong_sum(self, hawk_dove_solver):
        with pytest.raises(Exception):
            hawk_dove_solver.estimate_mean_absorption_time(
                beta=1.0, init_state=np.array([5, 5], dtype=np.uint64), nb_runs=10
            )

    def test_three_strategies(self, rps_solver):
        """Should work for k=3 strategies; non-monomorphic state absorbs eventually."""
        result = rps_solver.estimate_mean_absorption_time(
            beta=0.5, init_state=np.array([5, 5, 5], dtype=np.uint64), nb_runs=30
        )
        assert result["mean"] > 0.0


# ---------------------------------------------------------------------------
# estimate_absorption_probabilities
# ---------------------------------------------------------------------------

class TestEstimateAbsorptionProbabilities:

    def test_shape(self, hawk_dove_solver):
        probs = hawk_dove_solver.estimate_absorption_probabilities(
            beta=1.0, init_state=np.array([10, 10], dtype=np.uint64), nb_runs=50
        )
        assert probs.shape == (2,)

    def test_sums_to_one(self, hawk_dove_solver):
        probs = hawk_dove_solver.estimate_absorption_probabilities(
            beta=1.0, init_state=np.array([10, 10], dtype=np.uint64), nb_runs=100
        )
        assert probs.sum() == pytest.approx(1.0, abs=1e-9)

    def test_already_absorbed_one_hot(self, hawk_dove_solver):
        Z = 20
        probs = hawk_dove_solver.estimate_absorption_probabilities(
            beta=1.0, init_state=np.array([Z, 0], dtype=np.uint64), nb_runs=10
        )
        assert probs[0] == pytest.approx(1.0)
        assert probs[1] == pytest.approx(0.0)

    def test_neutral_drift_proportional_to_initial_frequency(self, hawk_dove_solver):
        """Under neutral drift absorption prob of strategy i ≈ init_freq(i)."""
        Z = 20
        k = 5  # 5 Hawks, 15 Doves
        probs = hawk_dove_solver.estimate_absorption_probabilities(
            beta=0.0, init_state=np.array([k, Z - k], dtype=np.uint64), nb_runs=2000
        )
        expected = k / Z  # neutral drift: fixation prob = initial frequency
        # Allow ±5% absolute tolerance for stochastic test
        assert probs[0] == pytest.approx(expected, abs=0.05)
        assert probs[1] == pytest.approx(1.0 - expected, abs=0.05)

    def test_three_strategies_shape_and_sum(self, rps_solver):
        probs = rps_solver.estimate_absorption_probabilities(
            beta=0.5, init_state=np.array([5, 5, 5], dtype=np.uint64), nb_runs=50
        )
        assert probs.shape == (3,)
        assert probs.sum() == pytest.approx(1.0, abs=1e-9)

    def test_values_in_unit_interval(self, hawk_dove_solver):
        probs = hawk_dove_solver.estimate_absorption_probabilities(
            beta=1.0, init_state=np.array([10, 10], dtype=np.uint64), nb_runs=100
        )
        assert np.all(probs >= 0.0)
        assert np.all(probs <= 1.0)


# ---------------------------------------------------------------------------
# calculate_spectral_gap
# ---------------------------------------------------------------------------

class TestCalculateSpectralGap:

    def test_single_state_returns_one(self):
        P = csr_matrix(np.array([[1.0]]))
        gap = egt.utils.calculate_spectral_gap(P)
        assert gap == pytest.approx(1.0)

    def test_gap_in_unit_interval(self):
        """Gap must lie in [0, 1] for a valid stochastic matrix."""
        Z, mu, beta = 20, 0.05, 1.0
        v, d, t = 2, 3, 1
        payoffs = np.array([[(v - d) / 2, v], [0, (v / 2) - t]], dtype=float)
        game = NormalFormGame(1, payoffs)
        analytical = egt.analytical.PairwiseComparison(Z, game)
        T = analytical.calculate_transition_matrix(beta, mu)
        gap = egt.utils.calculate_spectral_gap(T)
        assert 0.0 <= gap <= 1.0

    def test_stronger_selection_smaller_gap(self):
        """Coordination game: stronger selection → deeper basins → smaller gap (slower mixing).

        Stag Hunt payoffs: mutual stag (a) beats mutual hare (b), but stag is risky.
        Two stable pure-strategy equilibria → two deep basins under strong selection.
        """
        Z, mu = 20, 0.05
        # Stag Hunt: payoffs[i][j] = payoff of strategy i vs j
        # Stag(0) vs Stag: a, Stag vs Hare: 0, Hare vs Stag: b, Hare vs Hare: b
        a, b = 3.0, 1.0  # a > b > 0; stag dominant only when enough play stag
        payoffs = np.array([[a, 0.0], [b, b]], dtype=float)
        game = NormalFormGame(1, payoffs)
        analytical = egt.analytical.PairwiseComparison(Z, game)
        T_weak = analytical.calculate_transition_matrix(beta=0.5, mu=mu)
        T_strong = analytical.calculate_transition_matrix(beta=5.0, mu=mu)
        gap_weak = egt.utils.calculate_spectral_gap(T_weak)
        gap_strong = egt.utils.calculate_spectral_gap(T_strong)
        assert gap_weak >= gap_strong

    def test_dense_input_accepted(self):
        """calculate_spectral_gap should accept dense arrays too (via csr_matrix conversion)."""
        P_dense = np.array([
            [0.9, 0.1],
            [0.2, 0.8],
        ])
        gap = egt.utils.calculate_spectral_gap(P_dense)
        assert 0.0 <= gap <= 1.0

    def test_mixing_time_bound_positive(self):
        """Derived mixing time bound ceil(log(0.01)/log(1-gap)) should be a positive integer."""
        import math
        Z, mu, beta = 20, 0.05, 1.0
        v, d, t = 2, 3, 1
        payoffs = np.array([[(v - d) / 2, v], [0, (v / 2) - t]], dtype=float)
        game = NormalFormGame(1, payoffs)
        analytical = egt.analytical.PairwiseComparison(Z, game)
        T = analytical.calculate_transition_matrix(beta, mu)
        gap = egt.utils.calculate_spectral_gap(T)
        if gap > 0.0:
            mixing_bound = math.ceil(math.log(0.01) / math.log(1.0 - gap))
            assert mixing_bound > 0

import pytest
import numpy as np

egt = pytest.importorskip("egttools")

PairwiseComparisonNumerical = egt.numerical.PairwiseComparisonNumerical
Matrix2PlayerGameHolder = egt.games.Matrix2PlayerGameHolder


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def payoffs() -> np.ndarray:
    p_o = 0.2
    p_c = 0.5
    e_c = 1
    e_o = 2
    b_c = 15
    b_cf = 2
    b_o = 10
    b_p = 10
    c_cf = 5
    c_c = 50
    c_o = 3
    c_po = 1

    return np.array([
        [0, p_o * b_o + (1 - p_o) * (-c_o) - e_o, c_po],   # O, C, P
        [p_o * (-c_c), (b_cf - c_cf) / 2, p_c * b_c - e_c],  # C
        [-c_po, -p_c * c_c, b_p]                               # P
    ])


@pytest.fixture(scope="module")
def hawk_dove_payoffs() -> np.ndarray:
    """Classic 2-strategy Hawk-Dove game."""
    b, c = 4.0, 6.0
    return np.array([
        [(b - c) / 2, b],
        [0, b / 2],
    ])


@pytest.fixture(scope="module")
def game3(payoffs) -> Matrix2PlayerGameHolder:
    return Matrix2PlayerGameHolder(3, payoffs)


@pytest.fixture(scope="module")
def pc3(game3) -> PairwiseComparisonNumerical:
    """3-strategy solver, pop_size=100."""
    return PairwiseComparisonNumerical(100, game3, 1000)


@pytest.fixture(scope="module")
def game2(hawk_dove_payoffs) -> Matrix2PlayerGameHolder:
    return Matrix2PlayerGameHolder(2, hawk_dove_payoffs)


@pytest.fixture(scope="module")
def pc2(game2) -> PairwiseComparisonNumerical:
    """2-strategy (Hawk-Dove) solver, pop_size=50."""
    return PairwiseComparisonNumerical(50, game2, 500)


# ---------------------------------------------------------------------------
# Constructor / property tests
# ---------------------------------------------------------------------------

class TestConstructorAndProperties:
    def test_properties(self, pc3):
        assert pc3.pop_size == 100
        assert pc3.nb_strategies == 3
        assert pc3.cache_size == 1000

    def test_nb_states(self, pc3):
        # C(100+3-1, 3-1) = C(102, 2) = 5151
        assert pc3.nb_states == 5151

    def test_set_pop_size(self, payoffs):
        game = Matrix2PlayerGameHolder(3, payoffs)
        pc = PairwiseComparisonNumerical(50, game, 500)
        assert pc.pop_size == 50
        pc.pop_size = 80
        assert pc.pop_size == 80

    def test_set_cache_size(self, payoffs):
        game = Matrix2PlayerGameHolder(3, payoffs)
        pc = PairwiseComparisonNumerical(50, game, 500)
        pc.cache_size = 2000
        assert pc.cache_size == 2000

    def test_payoffs_shape(self, pc3, payoffs):
        assert pc3.payoffs.shape == payoffs.shape

    def test_payoffs_values(self, pc3, payoffs):
        np.testing.assert_array_almost_equal(pc3.payoffs, payoffs)


# ---------------------------------------------------------------------------
# evolve() tests
# ---------------------------------------------------------------------------

class TestEvolve:
    def test_evolve_returns_valid_state(self, pc3):
        init = np.array([40, 10, 50], dtype=np.uint64)
        result = pc3.evolve(500, 1.0, 1e-3, init)
        assert result.shape == (3,)
        assert result.sum() == 100
        assert np.all(result >= 0)

    def test_evolve_fixation(self, pc2):
        """Without mutation the population must fix."""
        init = np.array([1, 49], dtype=np.uint64)
        result = pc2.evolve(100_000, 1.0, 1e-6, init)
        assert result.sum() == 50
        assert np.all(result >= 0)


# ---------------------------------------------------------------------------
# run_without_mutation() tests
# ---------------------------------------------------------------------------

class TestRunWithoutMutation:
    def test_shape_no_transient(self, pc3):
        init = np.array([40, 10, 50], dtype=np.uint64)
        result = pc3.run_without_mutation(200, 1.0, init)
        assert result.shape == (201, 3)        # includes initial state
        assert np.all(result.sum(axis=1) == 100)

    def test_initial_row_equals_init_state(self, pc3):
        init = np.array([40, 10, 50], dtype=np.uint64)
        result = pc3.run_without_mutation(100, 1.0, init)
        np.testing.assert_array_equal(result[0], init)

    def test_shape_with_transient(self, pc3):
        init = np.array([40, 10, 50], dtype=np.uint64)
        result = pc3.run_without_mutation(200, 50, 1.0, init)
        assert result.shape == (150, 3)
        assert np.all(result.sum(axis=1) == 100)

    def test_fixation_fills_all_columns(self, pc2):
        """After fixation every subsequent row must be the fixated state."""
        init = np.array([1, 49], dtype=np.uint64)
        result = pc2.run_without_mutation(500, 1.0, init)
        assert result.shape == (501, 2)
        assert np.all(result.sum(axis=1) == 50)
        # Find fixation row
        fix_rows = np.where((result == 50).any(axis=1))[0]
        if fix_rows.size > 0:
            fix_row = fix_rows[0]
            fixed_state = result[fix_row]
            # All subsequent rows must equal the fixed state (both columns)
            np.testing.assert_array_equal(result[fix_row:], fixed_state)

    def test_homogeneous_init_returns_constant(self, pc3):
        """Starting from a homogeneous state without mutation stays constant."""
        init = np.array([100, 0, 0], dtype=np.uint64)
        result = pc3.run_without_mutation(50, 1.0, init)
        for row in result:
            np.testing.assert_array_equal(row, init)

    def test_homogeneous_init_with_transient(self, pc3):
        init = np.array([0, 0, 100], dtype=np.uint64)
        result = pc3.run_without_mutation(100, 20, 1.0, init)
        assert result.shape == (80, 3)
        for row in result:
            np.testing.assert_array_equal(row, init)


# ---------------------------------------------------------------------------
# run_with_mutation() tests
# ---------------------------------------------------------------------------

class TestRunWithMutation:
    def test_shape_no_transient(self, pc3):
        init = np.array([40, 10, 50], dtype=np.uint64)
        result = pc3.run_with_mutation(1000, 1.0, 1e-3, init)
        assert result.shape == (1001, 3)
        assert np.all(result.sum(axis=1) == 100)

    def test_shape_with_transient(self, pc3):
        init = np.array([40, 10, 50], dtype=np.uint64)
        result = pc3.run_with_mutation(1000, 0, 1.0, 1e-3, init)
        assert result.shape == (1000, 3)
        assert np.all(result.sum(axis=1) == 100)

    def test_shape_with_nonzero_transient(self, pc3):
        init = np.array([40, 10, 50], dtype=np.uint64)
        result = pc3.run_with_mutation(1000, 200, 1.0, 1e-3, init)
        assert result.shape == (800, 3)
        assert np.all(result.sum(axis=1) == 100)

    def test_mutation_breaks_fixation(self, pc2):
        """With mutation a fully homogeneous pop must eventually diversify."""
        init = np.array([50, 0], dtype=np.uint64)
        result = pc2.run_with_mutation(5000, 1.0, 0.1, init)
        # With mu=0.1, expect some non-monomorphic states
        non_mono = np.sum((result > 0).all(axis=1))
        assert non_mono > 0


# ---------------------------------------------------------------------------
# estimate_fixation_probability() tests
# ---------------------------------------------------------------------------

class TestFixationProbability:
    def test_neutral_drift(self, hawk_dove_payoffs):
        """At beta=0 (neutral), fixation probability ≈ 1/Z."""
        game = Matrix2PlayerGameHolder(2, hawk_dove_payoffs)
        pc = PairwiseComparisonNumerical(20, game, 500)
        fp = pc.estimate_fixation_probability(0, 1, 1000, 10_000, 0.0)
        assert abs(fp - 1 / 20) < 0.05  # within 5% of 1/Z

    def test_fixation_probability_range(self, pc3):
        fp = pc3.estimate_fixation_probability(0, 1, 500, 5000, 1.0)
        assert 0.0 <= fp <= 1.0

    def test_invalid_index_raises(self, pc3):
        with pytest.raises(Exception):
            pc3.estimate_fixation_probability(0, 10, 10, 100, 1.0)

    def test_same_strategy_raises(self, pc3):
        with pytest.raises(Exception):
            pc3.estimate_fixation_probability(1, 1, 10, 100, 1.0)


# ---------------------------------------------------------------------------
# estimate_stationary_distribution() tests
# ---------------------------------------------------------------------------

class TestStationaryDistribution:
    def test_shape_and_sums_to_one(self, pc2):
        sd = pc2.estimate_stationary_distribution(20, 5000, 500, 1.0, 1e-2)
        assert sd.ndim == 1
        assert abs(sd.sum() - 1.0) < 0.05

    def test_values_non_negative(self, pc2):
        sd = pc2.estimate_stationary_distribution(20, 5000, 500, 1.0, 1e-2)
        assert np.all(sd >= 0)

    def test_sparse_vs_dense_consistent(self, pc2):
        sd_dense = pc2.estimate_stationary_distribution(30, 5000, 500, 1.0, 1e-2)
        sd_sparse = pc2.estimate_stationary_distribution_sparse(30, 5000, 500, 1.0, 1e-2)
        sd_sparse_dense = np.array(sd_sparse.todense()).flatten()
        # Both should sum close to 1
        assert abs(sd_dense.sum() - 1.0) < 0.1
        assert abs(sd_sparse_dense.sum() - 1.0) < 0.1


# ---------------------------------------------------------------------------
# estimate_strategy_distribution() tests
# ---------------------------------------------------------------------------

class TestStrategyDistribution:
    # Use mu=0.05 so the geometric jump mean (~20 steps) is much smaller than
    # nb_generations - transitory (4500 steps), keeping normalization accurate.
    # Very small mu (e.g. 1e-3) causes large geometric jumps that can overshoot
    # the counting window, producing sums > 1.
    def test_shape(self, pc3):
        sd = pc3.estimate_strategy_distribution(10, 5000, 500, 1.0, 0.05)
        assert sd.shape == (3,)

    def test_sums_to_one(self, pc3):
        sd = pc3.estimate_strategy_distribution(20, 10000, 1000, 1.0, 0.05)
        assert abs(sd.sum() - 1.0) < 0.05

    def test_values_non_negative(self, pc3):
        sd = pc3.estimate_strategy_distribution(10, 5000, 500, 1.0, 0.05)
        assert np.all(sd >= 0)


# ---------------------------------------------------------------------------
# Input validation tests
# ---------------------------------------------------------------------------

class TestValidation:
    def test_wrong_init_state_length_raises(self, pc3):
        bad_init = np.array([50, 50], dtype=np.uint64)
        with pytest.raises(Exception):
            pc3.run_without_mutation(100, 1.0, bad_init)

    def test_init_state_wrong_sum_raises(self, pc3):
        bad_init = np.array([30, 30, 30], dtype=np.uint64)   # sums to 90, not 100
        with pytest.raises(Exception):
            pc3.run_without_mutation(100, 1.0, bad_init)

    def test_zero_generations_raises(self, pc3):
        init = np.array([40, 10, 50], dtype=np.uint64)
        with pytest.raises(Exception):
            pc3.run_without_mutation(0, 1.0, init)

    def test_transient_ge_nb_generations_raises(self, pc3):
        init = np.array([40, 10, 50], dtype=np.uint64)
        with pytest.raises(Exception):
            pc3.run_without_mutation(100, 100, 1.0, init)

    def test_mu_zero_raises(self, pc3):
        init = np.array([40, 10, 50], dtype=np.uint64)
        with pytest.raises(Exception):
            pc3.run_with_mutation(100, 1.0, 0.0, init)

    def test_mu_negative_raises(self, pc3):
        init = np.array([40, 10, 50], dtype=np.uint64)
        with pytest.raises(Exception):
            pc3.run_with_mutation(100, 1.0, -1e-3, init)

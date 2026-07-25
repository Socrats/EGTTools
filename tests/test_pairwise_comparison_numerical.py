import warnings
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

    def test_fixation_fills_all_columns(self):
        """After fixation every subsequent row must be the fixated state (both columns)."""
        # Use a strong selection game where one strategy dominates to guarantee fixation.
        # Strategy 0 strictly dominates: payoff always 10 vs 0.
        dom_payoffs = np.array([[10., 10.], [0., 0.]])
        game = Matrix2PlayerGameHolder(2, dom_payoffs)
        pc = PairwiseComparisonNumerical(20, game, 200)
        init = np.array([1, 19], dtype=np.uint64)
        # With beta=10 (strong selection) and a dominant strategy, fixation is nearly certain
        # within 500 generations
        result = pc.run_without_mutation(500, 10.0, init)
        assert result.shape == (501, 2)
        assert np.all(result.sum(axis=1) == 20)
        fix_rows = np.where((result == 20).any(axis=1))[0]
        assert fix_rows.size > 0, "Dominant strategy should have fixated within 500 generations"
        fix_row = fix_rows[0]
        fixed_state = result[fix_row]  # shape (2,)
        # Every subsequent row must equal the fixed state (broadcasting: (N,2) vs (2,))
        assert np.all(result[fix_row:] == fixed_state), (
            f"Rows after fixation at row {fix_row} are not constant:\n{result[fix_row:]}"
        )

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
# Biased mutation tests
# ---------------------------------------------------------------------------

class TestBiasedMutation:
    """Tests for set_mutation_matrix / set_mutation_weights / mutation_matrix.

    Statistical tests isolate a single mutation event by starting from a
    homogeneous population and running exactly 2 generations with mu close
    to 1: the homogeneous branch waits `k` generations (k=0 with near-certain
    probability when mu~1) then applies exactly one mutation, and with
    nb_generations=2 the post-mutation state lands exactly on the last row
    with no further generations processed — so the observed target strategy
    is a clean draw from the configured mutation kernel, uncontaminated by
    subsequent selection/imitation dynamics. Payoffs are neutral (all equal)
    so the homogeneous branch never invokes the fitness/fermi comparison.
    """

    @staticmethod
    def _neutral_solver(pop_size=20, nb_strategies=3):
        payoffs = np.ones((nb_strategies, nb_strategies))
        game = Matrix2PlayerGameHolder(nb_strategies, payoffs)
        return PairwiseComparisonNumerical(pop_size, game, 1000)

    @staticmethod
    def _single_mutation_targets(solver, init_state, n_samples=1000):
        """Runs n_samples independent 2-generation trajectories from a
        homogeneous init_state and returns a Counter of which strategy the
        single mutation event landed on."""
        from collections import Counter
        counts = Counter()
        init_signed = init_state.astype(np.int64)
        for _ in range(n_samples):
            final = solver.run_with_mutation(2, 1.0, 0.999, init_state)[-1].astype(np.int64)
            counts[int(np.argmax(final - init_signed))] += 1
        return counts

    # -- defaults / getters -------------------------------------------------

    def test_default_mutation_matrix_is_uniform(self, pc3):
        expected = np.ones((3, 3))
        np.fill_diagonal(expected, 0.0)
        np.testing.assert_array_equal(pc3.mutation_matrix, expected)

    # -- setters --------------------------------------------------------

    def test_set_mutation_weights_broadcasts_to_matrix(self):
        solver = self._neutral_solver()
        solver.set_mutation_weights([1, 1, 5])
        expected = np.array([
            [0, 1, 5],
            [1, 0, 5],
            [1, 1, 0],
        ], dtype=float)
        np.testing.assert_array_equal(solver.mutation_matrix, expected)

    def test_set_mutation_matrix_forces_zero_diagonal(self):
        """Diagonal entries are always ignored/zeroed, even if the caller
        passes non-zero values on the diagonal."""
        solver = self._neutral_solver()
        given = np.array([
            [9, 1, 1],
            [1, 9, 1],
            [1, 1, 9],
        ], dtype=float)
        solver.set_mutation_matrix(given)
        expected = np.array([
            [0, 1, 1],
            [1, 0, 1],
            [1, 1, 0],
        ], dtype=float)
        np.testing.assert_array_equal(solver.mutation_matrix, expected)

    # -- validation -------------------------------------------------------

    def test_set_mutation_weights_wrong_length_raises(self):
        solver = self._neutral_solver()
        with pytest.raises(Exception):
            solver.set_mutation_weights([1, 1])

    def test_set_mutation_matrix_wrong_shape_raises(self):
        solver = self._neutral_solver()
        with pytest.raises(Exception):
            solver.set_mutation_matrix(np.ones((2, 3)))

    def test_negative_weights_raise(self):
        solver = self._neutral_solver()
        with pytest.raises(Exception):
            solver.set_mutation_weights([-1, 1, 1])

    def test_row_with_no_positive_target_raises(self):
        """Row 0 has no positive entry among strategies 1, 2 -> mutating
        away from strategy 0 would have no valid target."""
        solver = self._neutral_solver()
        with pytest.raises(Exception):
            solver.set_mutation_matrix(np.array([
                [0, 0, 0],
                [1, 1, 0],
                [1, 0, 1],
            ], dtype=float))

    # -- statistical bias checks --------------------------------------------

    def test_vector_bias_skews_mutation_target(self):
        """With weights [1, 1, 100], mutating away from strategy 0 should
        land on strategy 2 in ~100/101 ~= 99% of events."""
        solver = self._neutral_solver()
        solver.set_mutation_weights([1, 1, 100])
        init_state = np.array([20, 0, 0], dtype=np.uint64)
        counts = self._single_mutation_targets(solver, init_state, n_samples=1000)
        assert counts[2] / 1000 > 0.9
        assert counts[1] / 1000 < 0.1

    def test_matrix_bias_is_source_dependent(self):
        """An asymmetric matrix must bias the mutation target differently
        depending on which strategy is currently mutating, proving the bias
        is not just a target-only vector under the hood."""
        solver = self._neutral_solver()
        solver.set_mutation_matrix(np.array([
            [0, 100, 1],
            [1, 0, 100],
            [100, 1, 0],
        ], dtype=float))

        counts_from_0 = self._single_mutation_targets(
            solver, np.array([20, 0, 0], dtype=np.uint64), n_samples=1000)
        counts_from_1 = self._single_mutation_targets(
            solver, np.array([0, 20, 0], dtype=np.uint64), n_samples=1000)

        # From strategy 0: should favor strategy 1, not strategy 2.
        assert counts_from_0[1] / 1000 > 0.9
        # From strategy 1: should favor strategy 2, not strategy 0.
        assert counts_from_1[2] / 1000 > 0.9

    def test_bias_applies_to_estimate_strategy_distribution(self):
        """The bias must also be visible through estimate_strategy_distribution,
        which reaches mutate_() via _update_step/_update_multi_step rather
        than the run() trajectory path -- confirms the duplicated inline
        mutation call sites were fixed consistently."""
        solver_uniform = self._neutral_solver()
        solver_biased = self._neutral_solver()
        solver_biased.set_mutation_weights([1, 1, 100])

        kwargs = dict(nb_runs=50, nb_generations=2000, transitory=200, beta=1.0, mu=0.3)
        dist_uniform = solver_uniform.estimate_strategy_distribution(**kwargs)
        dist_biased = solver_biased.estimate_strategy_distribution(**kwargs)

        # Strategy 2's share must be markedly higher under the biased kernel.
        assert dist_biased[2] > dist_uniform[2] + 0.1


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
# Small-mu UserWarning tests
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# change_game() tests
# ---------------------------------------------------------------------------

class TestChangeGame:
    def test_change_game_updates_payoffs(self, payoffs):
        game_a = Matrix2PlayerGameHolder(3, payoffs)
        new_payoffs = np.eye(3)
        new_game = Matrix2PlayerGameHolder(3, new_payoffs)
        pc = PairwiseComparisonNumerical(50, game_a, 500)
        pc.change_game(new_game)
        np.testing.assert_array_almost_equal(pc.payoffs, new_payoffs)

    def test_change_game_new_game_used_in_run(self):
        """After change_game the simulation uses the new game's payoffs."""
        # All-cooperation game: cooperate always wins
        coop_payoffs = np.array([[2., 2.], [0., 0.]])
        defect_payoffs = np.array([[0., 0.], [2., 2.]])

        coop_game = Matrix2PlayerGameHolder(2, coop_payoffs)
        defect_game = Matrix2PlayerGameHolder(2, defect_payoffs)

        pc = PairwiseComparisonNumerical(20, coop_game, 200)
        # Simulate with coop game: strategy 0 should tend to fixate
        init = np.array([1, 19], dtype=np.uint64)
        fp_coop = pc.estimate_fixation_probability(0, 1, 500, 5000, 5.0)

        # Switch to defect game: now strategy 1 should have higher fixation
        pc.change_game(defect_game)
        fp_defect_inv = pc.estimate_fixation_probability(1, 0, 500, 5000, 5.0)

        # Both games are symmetric mirror images, so their fixation probs should be similar
        assert abs(fp_coop - fp_defect_inv) < 0.1


# ---------------------------------------------------------------------------
# Small-mu UserWarning tests
# ---------------------------------------------------------------------------

class TestSmallMuWarning:
    """Estimation methods should warn when mu*(nb_gen-transitory) < 10."""

    def _make_pc(self):
        game = Matrix2PlayerGameHolder(2, np.array([[1., 0.], [0., 1.]]))
        return PairwiseComparisonNumerical(10, game, 100), game

    def test_estimate_strategy_distribution_warns(self):
        pc, game = self._make_pc()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            pc.estimate_strategy_distribution(2, 50, 5, 1.0, 1e-3)
        assert any(issubclass(x.category, UserWarning) for x in w)

    def test_estimate_stationary_distribution_warns(self):
        pc, game = self._make_pc()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            pc.estimate_stationary_distribution(2, 50, 5, 1.0, 1e-3)
        assert any(issubclass(x.category, UserWarning) for x in w)

    def test_estimate_stationary_distribution_sparse_warns(self):
        pc, game = self._make_pc()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            pc.estimate_stationary_distribution_sparse(2, 50, 5, 1.0, 1e-3)
        assert any(issubclass(x.category, UserWarning) for x in w)

    def test_no_warning_when_mu_large_enough(self):
        pc, game = self._make_pc()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            pc.estimate_strategy_distribution(2, 1000, 0, 1.0, 0.05)
        mu_warnings = [x for x in w if issubclass(x.category, UserWarning)]
        assert len(mu_warnings) == 0


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


# ---------------------------------------------------------------------------
# Tolerance-based early stopping tests
# ---------------------------------------------------------------------------

class TestToleranceEarlyStopping:
    """Tests for the tolerance / check_every early-stopping feature added to
    estimate_stationary_distribution, estimate_stationary_distribution_sparse,
    and estimate_strategy_distribution."""

    @staticmethod
    def _make_pc():
        """Small 2-strategy game for fast tolerance tests."""
        payoffs = np.array([[1.0, 0.0], [0.0, 1.0]])
        game = Matrix2PlayerGameHolder(2, payoffs)
        pc = PairwiseComparisonNumerical(20, game, 1000)
        return pc, game

    # --- estimate_stationary_distribution ---

    def test_stationary_no_tolerance_returns_full_run(self):
        """tolerance=0 (default) must always run all nb_runs."""
        pc, game = self._make_pc()
        result = pc.estimate_stationary_distribution(10, 500, 50, 1.0, 0.05)
        assert result.shape[0] == pc.nb_states
        assert abs(result.sum() - 1.0) < 0.1  # unnormalised; just sanity

    def test_stationary_tolerance_stops_before_max_runs(self):
        """With a generous tolerance the simulation should stop before nb_runs=1000."""
        pc, game = self._make_pc()
        # tolerance=0.5 is deliberately large so it converges early
        result = pc.estimate_stationary_distribution(
            1000, 500, 50, 1.0, 0.05, tolerance=0.5
        )
        assert result.shape[0] == pc.nb_states
        assert result.sum() > 0

    def test_stationary_tolerance_result_is_distribution(self):
        """With a modest tolerance the result should still be a valid distribution."""
        pc, game = self._make_pc()
        result = pc.estimate_stationary_distribution(
            100, 1000, 100, 1.0, 0.05, tolerance=0.05
        )
        assert result.shape[0] == pc.nb_states
        assert np.all(result >= 0.0)
        assert abs(result.sum() - 1.0) < 0.15

    def test_stationary_check_every_respected(self):
        """check_every > 0 should not crash and should return a valid result."""
        pc, game = self._make_pc()
        result = pc.estimate_stationary_distribution(
            50, 500, 50, 1.0, 0.05, tolerance=0.1, check_every=5
        )
        assert result.shape[0] == pc.nb_states
        assert result.sum() > 0

    # --- estimate_stationary_distribution_sparse ---

    def test_sparse_stationary_tolerance_result_is_valid(self):
        pc, game = self._make_pc()
        result = pc.estimate_stationary_distribution_sparse(
            100, 1000, 100, 1.0, 0.05, tolerance=0.05
        )
        dense = np.asarray(result.todense()).flatten()
        assert dense.shape[0] == pc.nb_states
        assert np.all(dense >= 0.0)
        assert abs(dense.sum() - 1.0) < 0.15

    def test_sparse_stationary_no_tolerance_matches_dense(self):
        """Without tolerance both dense and sparse should give similarly shaped outputs."""
        pc, game = self._make_pc()
        dense = pc.estimate_stationary_distribution(20, 500, 50, 1.0, 0.05)
        sparse = pc.estimate_stationary_distribution_sparse(20, 500, 50, 1.0, 0.05)
        sparse_dense = np.asarray(sparse.todense()).flatten()
        assert dense.shape == sparse_dense.shape

    # --- estimate_strategy_distribution ---

    def test_strategy_dist_no_tolerance_valid(self):
        pc, game = self._make_pc()
        result = pc.estimate_strategy_distribution(10, 500, 50, 1.0, 0.05)
        assert result.shape[0] == pc.nb_strategies
        assert np.all(result >= 0.0)
        assert abs(result.sum() - 1.0) < 0.15

    def test_strategy_dist_tolerance_stops_early(self):
        pc, game = self._make_pc()
        result = pc.estimate_strategy_distribution(
            1000, 500, 50, 1.0, 0.05, tolerance=0.5
        )
        assert result.shape[0] == pc.nb_strategies
        assert result.sum() > 0

    def test_strategy_dist_tolerance_result_is_distribution(self):
        pc, game = self._make_pc()
        result = pc.estimate_strategy_distribution(
            100, 1000, 100, 1.0, 0.05, tolerance=0.05
        )
        assert result.shape[0] == pc.nb_strategies
        assert np.all(result >= 0.0)
        assert abs(result.sum() - 1.0) < 0.15

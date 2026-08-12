import numpy as np
import pytest

egt = pytest.importorskip("egttools")

MLSTraulsen = egt.numerical.MLSTraulsen
MLSGarcia = egt.numerical.MLSGarcia


def donation_game(b: float, c: float = 1.0) -> np.ndarray:
    """2-strategy donation game payoff matrix. Row 0 = Cooperator, row 1 = Defector."""
    return np.array([[b - c, -c],
                      [b, 0.0]])


# ---------------------------------------------------------------------------
# MLSTraulsen — Traulsen & Nowak (2006)
# ---------------------------------------------------------------------------

@pytest.fixture
def traulsen_evolver() -> MLSTraulsen:
    payoff = donation_game(b=3.0)
    freq = np.array([0.5, 0.5])
    return MLSTraulsen(2000, 2, 4, 3, 0.1, freq, payoff)


class TestMLSTraulsenConstruction:
    def test_properties_after_construction(self, traulsen_evolver):
        e = traulsen_evolver
        assert e.nb_strategies == 2
        assert e.group_size == 4
        assert e.nb_groups == 3
        assert e.max_pop_size == 12
        assert e.generations == 2000
        np.testing.assert_allclose(e.payoff_matrix, donation_game(3.0))

    def test_setters_update_dependent_properties(self, traulsen_evolver):
        e = traulsen_evolver
        e.generations = 500
        assert e.generations == 500

        e.group_size = 5
        assert e.group_size == 5
        assert e.max_pop_size == 5 * e.nb_groups

        e.nb_groups = 4
        assert e.nb_groups == 4
        assert e.max_pop_size == e.group_size * 4

        e.selection_intensity = 0.2
        assert e.selection_intensity == pytest.approx(0.2)

    def test_group_size_below_minimum_raises(self, traulsen_evolver):
        with pytest.raises(Exception):
            traulsen_evolver.group_size = 3

    def test_repr_reports_population_structure(self, traulsen_evolver):
        text = repr(traulsen_evolver)
        assert "Z = 12" in text
        assert "m = 3" in text
        assert "n = 4" in text


class TestMLSTraulsenEvolve:
    def test_evolve_no_migration_returns_valid_distribution(self, traulsen_evolver):
        init_state = np.array([6, 6], dtype=np.uint64)
        freqs = traulsen_evolver.evolve(0.0, 0.1, init_state)
        assert len(freqs) == 2
        assert np.all(freqs >= 0.0) and np.all(freqs <= 1.0)
        assert freqs.sum() == pytest.approx(1.0, abs=1e-9)

    def test_evolve_with_migration_returns_valid_distribution(self, traulsen_evolver):
        init_state = np.array([6, 6], dtype=np.uint64)
        freqs = traulsen_evolver.evolve(0.0, 0.1, 0.05, init_state)
        assert len(freqs) == 2
        assert freqs.sum() == pytest.approx(1.0, abs=1e-9)

    def test_evolve_wrong_init_state_length_raises(self, traulsen_evolver):
        with pytest.raises(Exception):
            traulsen_evolver.evolve(0.0, 0.1, np.array([12], dtype=np.uint64))

    def test_evolve_init_state_not_summing_to_pop_size_raises(self, traulsen_evolver):
        with pytest.raises(Exception):
            traulsen_evolver.evolve(0.0, 0.1, np.array([1, 1], dtype=np.uint64))


class TestMLSTraulsenFixationProbability:
    def test_no_migration_overload_in_bounds(self, traulsen_evolver):
        fp = traulsen_evolver.fixation_probability(0, 1, 100, 0.0, 0.1)
        assert 0.0 <= fp <= 1.0

    def test_with_migration_overload_in_bounds(self, traulsen_evolver):
        fp = traulsen_evolver.fixation_probability(0, 1, 100, 0.0, 0.05, 0.1)
        assert 0.0 <= fp <= 1.0

    def test_invalid_strategy_index_raises(self, traulsen_evolver):
        with pytest.raises(Exception):
            traulsen_evolver.fixation_probability(5, 1, 10, 0.0, 0.1)

    def test_nonzero_splitting_with_single_group_raises(self):
        payoff = donation_game(b=3.0)
        freq = np.array([0.5, 0.5])
        evolver = MLSTraulsen(2000, 2, 4, 1, 0.1, freq, payoff)
        with pytest.raises(Exception):
            evolver.fixation_probability(0, 1, 10, 0.001, 0.1)


class TestMLSTraulsenGradientOfSelection:
    def test_shape_matches_population_size(self, traulsen_evolver):
        gradient = traulsen_evolver.gradient_of_selection(0, 1, 50, 0.1)
        assert len(gradient) == traulsen_evolver.max_pop_size + 1


# ---------------------------------------------------------------------------
# MLSGarcia — Garcia & van den Bergh (2011)
# ---------------------------------------------------------------------------

@pytest.fixture
def garcia_payoffs():
    b, c = 3.0, 1.0
    payoff_in = donation_game(b, c)
    payoff_out = donation_game(b, c)
    return payoff_in, payoff_out


@pytest.fixture
def garcia_evolver() -> MLSGarcia:
    return MLSGarcia(2000, 2, 4, 3)


class TestMLSGarciaConstruction:
    def test_properties_after_construction(self, garcia_evolver):
        e = garcia_evolver
        assert e.nb_strategies == 2
        assert e.group_size == 4
        assert e.nb_groups == 3
        assert e.max_pop_size == 12

    def test_setters_update_dependent_properties(self, garcia_evolver):
        e = garcia_evolver
        e.generations = 500
        assert e.generations == 500

        e.group_size = 5
        assert e.group_size == 5

        e.nb_groups = 4
        assert e.nb_groups == 4
        assert e.max_pop_size == e.group_size * 4

    def test_group_size_below_minimum_raises(self, garcia_evolver):
        with pytest.raises(Exception):
            garcia_evolver.group_size = 3

    def test_repr_reports_population_structure(self, garcia_evolver):
        text = repr(garcia_evolver)
        assert "Z = 12" in text
        assert "m = 3" in text


class TestMLSGarciaFixationProbability:
    def test_in_bounds(self, garcia_evolver, garcia_payoffs):
        payoff_in, payoff_out = garcia_payoffs
        fp = garcia_evolver.fixation_probability(
            0, 1, 100, 0.01, 0.0, 0.1, 0.8, 0.0, 0.0, payoff_in, payoff_out)
        assert 0.0 <= fp <= 1.0

    def test_invalid_strategy_index_raises(self, garcia_evolver, garcia_payoffs):
        payoff_in, payoff_out = garcia_payoffs
        with pytest.raises(Exception):
            garcia_evolver.fixation_probability(
                5, 1, 10, 0.01, 0.0, 0.1, 0.8, 0.0, 0.0, payoff_in, payoff_out)

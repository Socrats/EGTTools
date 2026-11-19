"""Input validation at the user-facing (Python) entry points.

Rules (roadmap MEM-09):
- a strategy index outside [0, S) raises ``IndexError``;
- an invading strategy equal to the resident one raises ``ValueError``;
- a population state must have one entry per strategy, each between 0 and Z, summing to Z;
  otherwise ``ValueError``.

The population has Z = 100 individuals and S = 4 strategies.
"""
import numpy as np
import pytest

egt = pytest.importorskip("egttools")

from egttools.analytical import PairwiseComparison  # noqa: E402
from egttools.numerical import PairwiseComparisonNumerical  # noqa: E402

Z, S = 100, 4
VALID_STATE = np.array([25, 25, 25, 25], dtype=np.uint64)

BAD_INDEX_PAIRS = [(7, 0), (0, 7), (S, 0), (-1, 0), (0, -1)]

BAD_STATES = {
    "too short": np.array([50, 50], dtype=np.uint64),
    "too long": np.array([25, 25, 25, 25, 0], dtype=np.uint64),
    "sum above Z": np.array([50, 50, 10, 0], dtype=np.uint64),
    "sum below Z": np.array([10, 10, 10, 10], dtype=np.uint64),
    # Unsigned wrap-around: the entries sum to Z modulo 2^64, but one is larger than Z.
    "entry above Z": np.array([Z + 1, 2 ** 64 - 1, 0, 0], dtype=np.uint64),
}


@pytest.fixture(scope="module")
def game():
    payoffs = np.random.default_rng(0).random((S, S))
    return egt.games.Matrix2PlayerGameHolder(S, payoffs)


@pytest.fixture(scope="module")
def analytical(game):
    return PairwiseComparison(Z, game)


@pytest.fixture(scope="module")
def numerical(game):
    return PairwiseComparisonNumerical(Z, game, 1000)


# --- PairwiseComparison: strategy indices ---------------------------------------------------------

FIXATION_METHODS = {
    "calculate_fixation_probability": lambda pc, i, j: pc.calculate_fixation_probability(i, j, 1.0),
    "calculate_fixation_probability(high_precision)":
        lambda pc, i, j: pc.calculate_fixation_probability(i, j, 1.0, high_precision=True),
    "calculate_log_fixation_probability": lambda pc, i, j: pc.calculate_log_fixation_probability(i, j, 1.0),
}


@pytest.mark.parametrize("method", FIXATION_METHODS)
@pytest.mark.parametrize("invader, resident", BAD_INDEX_PAIRS)
def test_analytical_fixation_rejects_out_of_range_index(analytical, method, invader, resident):
    with pytest.raises(IndexError):
        FIXATION_METHODS[method](analytical, invader, resident)


@pytest.mark.parametrize("method", FIXATION_METHODS)
def test_analytical_fixation_rejects_equal_strategies(analytical, method):
    with pytest.raises(ValueError):
        FIXATION_METHODS[method](analytical, 1, 1)


@pytest.mark.parametrize("method", FIXATION_METHODS)
def test_analytical_fixation_accepts_valid_indices(analytical, method):
    value = FIXATION_METHODS[method](analytical, 1, 0)
    assert np.isfinite(value)


# --- PairwiseComparison: population states --------------------------------------------------------

GRADIENT_METHODS = {
    "calculate_gradient_of_selection": lambda pc, state: pc.calculate_gradient_of_selection(1.0, state),
    "calculate_gradient_of_selection_with_mutation":
        lambda pc, state: pc.calculate_gradient_of_selection_with_mutation(1.0, 0.01, state),
}


@pytest.mark.parametrize("method", GRADIENT_METHODS)
@pytest.mark.parametrize("label", BAD_STATES)
def test_analytical_gradient_rejects_invalid_state(analytical, method, label):
    with pytest.raises(ValueError):
        GRADIENT_METHODS[method](analytical, BAD_STATES[label])


@pytest.mark.parametrize("method", GRADIENT_METHODS)
def test_analytical_gradient_accepts_valid_state(analytical, method):
    gradient = GRADIENT_METHODS[method](analytical, VALID_STATE)
    assert gradient.shape == (S,)


# --- PairwiseComparisonNumerical ------------------------------------------------------------------

@pytest.mark.parametrize("invader, resident", BAD_INDEX_PAIRS)
def test_numerical_fixation_rejects_out_of_range_index(numerical, invader, resident):
    with pytest.raises(IndexError):
        numerical.estimate_fixation_probability(invader, resident, 2, 100, 1.0)


def test_numerical_fixation_rejects_equal_strategies(numerical):
    with pytest.raises(ValueError):
        numerical.estimate_fixation_probability(1, 1, 2, 100, 1.0)


STATE_METHODS = {
    "evolve": lambda pcn, state: pcn.evolve(10, 1.0, 0.01, state),
    "run_with_mutation": lambda pcn, state: pcn.run_with_mutation(10, 1.0, 0.01, state),
    "estimate_mean_absorption_time": lambda pcn, state: pcn.estimate_mean_absorption_time(1.0, state, 2),
    "estimate_absorption_probabilities": lambda pcn, state: pcn.estimate_absorption_probabilities(1.0, state, 2),
}


@pytest.mark.parametrize("method", STATE_METHODS)
@pytest.mark.parametrize("label", BAD_STATES)
def test_numerical_rejects_invalid_state(numerical, method, label):
    with pytest.raises(ValueError):
        STATE_METHODS[method](numerical, BAD_STATES[label])


@pytest.mark.parametrize("method", ["evolve", "run_with_mutation"])
def test_numerical_accepts_valid_state(numerical, method):
    STATE_METHODS[method](numerical, VALID_STATE)


def test_error_messages_name_the_argument_and_range(analytical):
    with pytest.raises(IndexError, match=r"\[0, 4\)"):
        analytical.calculate_fixation_probability(7, 0, 1.0)
    with pytest.raises(ValueError, match="sum to the population size"):
        analytical.calculate_gradient_of_selection(1.0, BAD_STATES["sum above Z"])

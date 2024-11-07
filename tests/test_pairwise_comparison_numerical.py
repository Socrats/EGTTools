import pytest
import numpy as np

egt = pytest.importorskip("egttools")

PairwiseComparisonNumerical = egt.numerical.PairwiseComparisonNumerical
Matrix2PlayerGameHolder = egt.games.Matrix2PlayerGameHolder


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
        [0, p_o * b_o + (1 - p_o) * (-c_o) - e_o, c_po],  # O , C, P
        [p_o * (-c_c), (b_cf - c_cf) / 2, p_c * b_c - e_c],  # C
        [-c_po, -p_c * c_c, b_p]  # P
    ])


@pytest.fixture(scope="module")
def parameters() -> tuple[int, int, int, np.uint64, float, float, np.ndarray]:
    pop_size = 100
    nb_strategies = 3
    cache_size = 1000
    nb_population_states = egt.calculate_nb_states(pop_size, nb_strategies)

    beta = 1
    mu = 1e-3
    init_state = np.array([40, 10, 50], dtype=np.uint64)

    return pop_size, nb_strategies, cache_size, nb_population_states, beta, mu, init_state


def test_pairwise_comparison_numerical_run_v1(payoffs, parameters) -> None:
    """
    This test checks that the run method of PairwiseComparisonNumerical executes.
    """
    game = Matrix2PlayerGameHolder(parameters[1], payoffs)

    pc = PairwiseComparisonNumerical(parameters[0], game, parameters[2])

    nb_generations = 1000


    result = pc.run_with_mutation(nb_generations, parameters[4], parameters[5], parameters[6])

    assert result.shape == (nb_generations + 1, game.nb_strategies())
    assert np.all(np.sum(result, axis=1) == parameters[0])


def test_pairwise_comparison_numerical_run_v2(payoffs, parameters) -> None:
    """
    This test checks that the run method of PairwiseComparisonNumerical executes.
    """
    game = Matrix2PlayerGameHolder(parameters[1], payoffs)

    pc = PairwiseComparisonNumerical(parameters[0], game, parameters[2])

    nb_generations = 1000


    result = pc.run_with_mutation(nb_generations, 0, parameters[4], parameters[5], parameters[6])

    assert result.shape == (nb_generations, game.nb_strategies())
    assert np.all(np.sum(result, axis=1) == parameters[0])

import numpy as np
import egttools as egt


def test_analytical_fitness_calculation():
    N = 6
    M = 3
    c = 0.1
    b = 1
    Z = 50
    risk = 1.
    game = egt.games.OneShotCRD(b, c, risk, N, M)
    payoffs = egt.utils.transform_payoffs_to_pairwise(game.nb_strategies(), game)
    evolver = egt.analytical.StochDynamics(game.nb_strategies(), payoffs, Z, N)
    result = evolver.fitness_group(Z - 1, 0, 1)

    assert result > 0


def test_analytical_pairwise_comparison_generic_game():
    N = 6
    M = 3
    c = 0.1
    b = 1
    Z = 50
    risk = 1.
    game = egt.games.OneShotCRD(b, c, risk, N, M)
    evolver = egt.analytical.PairwiseComparison(Z, game)
    result = evolver.calculate_fixation_probability(0, 1, 1)

    assert np.isclose(0.003682100470889826, result)


def test_analytical_pairwise_comparison_2_player_game():
    # Payoff matrix
    V = 2
    D = 3
    T = 1
    A = np.array([
        [(V - D) / 2, V],
        [0, (V / 2) - T],
    ])
    game = egt.games.Matrix2PlayerGameHolder(2, payoff_matrix=A)
    evolver = egt.analytical.PairwiseComparison(100, game)
    result = evolver.calculate_fixation_probability(0, 1, 1)

    assert np.isclose(0.8641155742462664, result)


def test_analytical_pairwise_comparison_n_player_game():
    # Payoff matrix
    V = 2
    D = 3
    T = 1
    A = np.array([
        [(V - D) / 2, V, V/2, D],
        [0, (V / 2) - T, T, V],
    ])
    game = egt.games.MatrixNPlayerGameHolder(nb_strategies=2, group_size=3, payoff_matrix=A)
    evolver = egt.analytical.PairwiseComparison(100, game)
    result = evolver.calculate_fixation_probability(1, 0, 1)

    assert np.isclose(0.020104159364630978, result)

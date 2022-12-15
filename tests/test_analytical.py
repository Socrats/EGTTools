import numpy as np
import egttools as egt


def test_analytical_fitness_calculation_stoch_dynamics():
    group_size = 6
    min_nb_cooperators = 3
    c = 0.1
    b = 1
    population_size = 50
    risk = 1.
    game = egt.games.OneShotCRD(b, c, risk, group_size, min_nb_cooperators)
    payoffs = egt.utils.transform_payoffs_to_pairwise(game.nb_strategies(), game)
    evolver = egt.analytical.StochDynamics(game.nb_strategies(), payoffs, population_size, group_size)
    result = evolver.fitness_group(population_size - 1, 0, 1)

    assert result > 0


def test_analytical_pairwise_comparison_generic_game():
    group_size = 6
    min_nb_cooperators = 3
    c = 0.1
    b = 1
    population_size = 50
    risk = 1.
    game = egt.games.OneShotCRD(b, c, risk, group_size, min_nb_cooperators)
    evolver = egt.analytical.PairwiseComparison(population_size, game)
    result = evolver.calculate_fixation_probability(0, 1, 1)

    assert np.isclose(0.003682100470889826, result)


def test_analytical_pairwise_comparison_2_player_game():
    # Payoff matrix
    v = 2
    d = 3
    t = 1
    payoff_matrix = np.array([
        [(v - d) / 2, v],
        [0, (v / 2) - t],
    ])
    game = egt.games.Matrix2PlayerGameHolder(2, payoff_matrix=payoff_matrix)
    evolver = egt.analytical.PairwiseComparison(100, game)
    result = evolver.calculate_fixation_probability(0, 1, 1)

    assert np.isclose(0.8641155742462664, result)


def test_analytical_pairwise_comparison_n_player_game():
    # Payoff matrix
    v = 2
    d = 3
    t = 1
    payoff_matrix = np.array([
        [(v - d) / 2, v, v / 2, d],
        [0, (v / 2) - t, t, v],
    ])
    game = egt.games.MatrixNPlayerGameHolder(nb_strategies=2, group_size=3, payoff_matrix=payoff_matrix)
    evolver = egt.analytical.PairwiseComparison(100, game)
    result = evolver.calculate_fixation_probability(1, 0, 1)

    assert np.isclose(0.020104159364630978, result)


def test_if_stoch_dynamics_matches_with_pairwise_comparison():
    # Payoff matrix
    v = 2
    d = 3
    t = 1
    payoff_matrix = np.array([
        [(v - d) / 2, v],
        [0, (v / 2) - t],
    ])
    game = egt.games.Matrix2PlayerGameHolder(2, payoff_matrix=payoff_matrix)
    evolver1 = egt.analytical.PairwiseComparison(100, game)
    evolver2 = egt.analytical.StochDynamics(nb_strategies=2, payoffs=payoff_matrix, pop_size=100, group_size=2, mu=0)

    assert np.isclose(evolver1.calculate_fixation_probability(0, 1, 1), evolver2.fixation_probability(0, 1, 1))
    assert np.isclose(evolver1.calculate_gradient_of_selection(1, np.array([30
                                                                               , 70]))[0],
                      evolver2.gradient_selection(30, 0, 1, 1))
    transition1, fixation1 = evolver1.calculate_transition_and_fixation_matrix_sml(1)
    transition2, fixation2 = evolver2.transition_and_fixation_matrix(1)

    assert np.allclose(transition1.transpose(), transition2)
    assert np.allclose(fixation1, fixation2)

    full_transitions1 = evolver1.calculate_transition_matrix(1, 0.01)
    evolver2.mu = 0.01
    full_transitions2 = evolver2.calculate_full_transition_matrix(1)

    assert np.allclose(full_transitions1.toarray().transpose(), full_transitions2.toarray())

    # Now the same but for N-player games
    payoff_matrix = np.array([
        [(v - d) / 2, v, v / 2, d],
        [0, (v / 2) - t, t, v],
    ])
    game = egt.games.MatrixNPlayerGameHolder(nb_strategies=2, group_size=3, payoff_matrix=payoff_matrix)
    pairwise_payoffs = egt.utils.transform_payoffs_to_pairwise(2, game)
    evolver1 = egt.analytical.PairwiseComparison(100, game)
    evolver2 = egt.analytical.StochDynamics(nb_strategies=2, payoffs=pairwise_payoffs, pop_size=100, group_size=3, mu=0)

    assert np.isclose(evolver1.calculate_fixation_probability(1, 0, 1), evolver2.fixation_probability(1, 0, 1))
    assert np.isclose(evolver1.calculate_gradient_of_selection(1, np.array([30, 70]))[0],
                      evolver2.gradient_selection(30, 0, 1, 1))

    transition1, fixation1 = evolver1.calculate_transition_and_fixation_matrix_sml(1)
    transition2, fixation2 = evolver2.transition_and_fixation_matrix(1)

    assert np.allclose(transition1.transpose(), transition2)
    assert np.allclose(fixation1, fixation2)

    full_transitions1 = evolver1.calculate_transition_matrix(1, 0.01)
    evolver2 = egt.analytical.StochDynamics(nb_strategies=2, payoffs=payoff_matrix, pop_size=100, group_size=3, mu=0.01)
    full_transitions2 = evolver2.calculate_full_transition_matrix(1)

    assert np.allclose(full_transitions1.toarray().transpose(), full_transitions2.toarray())


def test_replicator_dynamics_2_player():
    # Payoff matrix
    v = 2
    d = 3
    t = 1
    payoff_matrix = np.array([
        [(v - d) / 2, v],
        [0, (v / 2) - t],
    ])
    x = np.array([0, 1])
    result = egt.analytical.replicator_equation(x, payoff_matrix)

    # Now the same but for N-player games
    payoff_matrix2 = np.array([
        [(v - d) / 2, v, v / 2, d],
        [0, (v / 2) - t, t, v],
    ])

    result = egt.analytical.replicator_equation_n_player(x, payoff_matrix2, group_size=3)


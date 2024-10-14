import numpy as np, egttools as egt


def test_matrix_2_player_game_holder():
    # Payoff matrix
    r = 1
    p = 0
    t = 2
    s = -1
    delta = 4
    eps = 0.25

    payoff_matrix = np.array([
        [r - (eps / 2), r - eps, 0, s + delta - eps, r - eps],
        [r, r, s, s, s],
        [0, t, p, p, p],
        [t - delta, t, p, p, p],
        [r, t, p, p, p],
    ])

    nb_strategies = payoff_matrix.shape[0]

    # instantiate game
    game = egt.games.Matrix2PlayerGameHolder(nb_strategies, payoff_matrix)

    assert np.allclose(game.payoffs(), payoff_matrix)
    assert game.nb_strategies() == nb_strategies


def pgg(group_size, multiplying_factor, cost):
    nb_strategies = 2
    nb_group_configurations = egt.calculate_nb_states(group_size, nb_strategies)
    payoff_matrix = np.zeros(shape=(nb_strategies, nb_group_configurations))

    for index in range(nb_group_configurations):
        group_configurations = egt.sample_simplex(index, group_size, nb_strategies)
        # group_configurations[1] gives the number of cooperators in the group
        if group_configurations[0] > 0:
            payoff_matrix[0, index] = (multiplying_factor * group_configurations[1] * cost) / group_size
        if group_configurations[1] > 0:
            payoff_matrix[1, index] = ((multiplying_factor * group_configurations[1] * cost) / group_size) - cost

    return payoff_matrix


def test_matrix_n_player_game_holder():
    # Payoff matrix
    group_size = 4
    multiplying_factor = 1
    cost = 1
    payoff_matrix = pgg(group_size=group_size, cost=cost, multiplying_factor=multiplying_factor)

    nb_strategies = payoff_matrix.shape[0]

    # instantiate game
    game = egt.games.MatrixNPlayerGameHolder(nb_strategies=nb_strategies, group_size=group_size,
                                             payoff_matrix=payoff_matrix)

    assert np.allclose(game.payoffs(), payoff_matrix)
    assert game.nb_strategies() == nb_strategies
    assert game.group_size() == group_size

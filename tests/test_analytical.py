import numpy as np
import egttools as egt


def test_analytical_fitness_calculation():
    N = 6
    M = 3
    c = 0.1
    b = 1
    Z = 50
    beta = 2
    pop_states = np.arange(0, Z + 1, 1)
    risk = 1.
    game = egt.games.OneShotCRD(b, c, risk, N, M)
    payoffs = egt.utils.transform_payoffs_to_pairwise(game.nb_strategies(), game)
    evolver = egt.analytical.StochDynamics(game.nb_strategies(), payoffs, Z, N)
    result = evolver.fitness_group(Z - 1, 0, 1)

    assert result > 0

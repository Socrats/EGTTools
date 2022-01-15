import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


# os.environ['KMP_STACKSIZE'] = '4m'
# os.environ['KMP_VERSION'] = '.TRUE.'

def test_openmp_simulations():
    import numpy as np
    import egttools as egt

    # Payoff matrix
    V = 2
    D = 3
    T = 1
    Z = 100
    A = np.array([
        [(V - D) / 2, V],
        [0, (V / 2) - T],
    ])

    game = egt.games.NormalFormGame(1, A)
    assert game.type() == 'NormalFormGame'

    evolver = egt.numerical.PairwiseMoran(Z, game, 1000)
    dist = evolver.estimate_stationary_distribution_sparse(1, 100000, 1000, 1, 1e-3)

    assert dist.toarray().shape == (1, Z+1)

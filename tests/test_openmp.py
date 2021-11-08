import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
if __name__ == "__main__":
    import egttools as egt
    import numpy as np

    print(egt.__version__)
    print(np.__version__)

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
    print(game.type())
    evolver = egt.numerical.PairwiseMoran(Z, game, 1000)
    dist = evolver.stationary_distribution(1, 10000000, 1000, 1, 1e-3)
    print(dist)

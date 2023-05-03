import pytest

import os
from sys import platform

import numpy as np
egt = pytest.importorskip("egttools")

Random = egt.Random
PairwiseComparisonNumerical = egt.numerical.PairwiseComparisonNumerical
NormalFormGame = egt.games.NormalFormGame


# os.environ['KMP_STACKSIZE'] = '4m'
# os.environ['KMP_VERSION'] = '.TRUE.'

def test_openmp_simulations():
    # Payoff matrix
    V = 2
    D = 3
    T = 1
    Z = 100
    A = np.array([
        [(V - D) / 2, V],
        [0, (V / 2) - T],
    ])

    # Necessary to avoid issues with Anaconda on MacOSX
    # if platform == "darwin":
    #     os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

    Random.init(3610063510)

    assert Random._seed == 3610063510

    game = egt.games.NormalFormGame(1, A)
    assert game.type() == 'NormalFormGame'

    evolver = PairwiseComparisonNumerical(Z, game, 10000)
    dist = evolver.estimate_stationary_distribution_sparse(10, 100000, 1000, 1, 1e-3)

    assert dist.toarray().shape == (1, Z + 1)


if __name__ == "__main__":
    test_openmp_simulations()

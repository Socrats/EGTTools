import pytest

import numpy as np

egttools = pytest.importorskip("egttools")

PGG = egttools.games.PGG
player_factory = egttools.behaviors.pgg_behaviors.player_factory


def test_extend_abstract_game() -> None:
    """
    This test checks that a Normal form game runs as expected
    """
    strategies = player_factory([0, 1])

    game = PGG(group_size=2, cost=np.float64(1.0), multiplying_factor=np.float64(3.0), strategies=strategies)

    payoffs = np.array([
        [0., 1.5, 0.],
        [0., 0.5, 2.]
    ])

    assert game.nb_strategies() == 2
    assert game.type() == "PGG"
    np.testing.assert_array_equal(game.payoffs(), payoffs)

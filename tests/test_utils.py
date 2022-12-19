from typing import Tuple

import numpy as np
import pytest

egttools = pytest.importorskip("egttools")

replicator_equation = egttools.analytical.replicator_equation
find_saddle_type_and_gradient_direction = egttools.utils.find_saddle_type_and_gradient_direction


@pytest.fixture
def setup_hawk_dove_gradients_and_saddle_points() -> Tuple[np.array, np.array]:
    nb_points = 101
    strategy_i = np.linspace(0, 1, num=nb_points, dtype=np.float64)
    strategy_j = 1 - strategy_i
    states = np.array((strategy_i, strategy_j)).T

    # Payoff matrix
    v, d, t = 2, 3, 1
    payoffs = np.array([
        [(v - d) / 2, v],
        [0, (v / 2) - t],
    ])

    # Calculate gradient
    gradients = np.array([replicator_equation(states[i], payoffs)[0] for i in range(len(states))])
    epsilon = 1e-3
    saddle_points_idx = np.where((gradients <= epsilon) & (gradients >= -epsilon))[0]

    return gradients, saddle_points_idx


def test_find_saddle_type_and_gradient_direction(setup_hawk_dove_gradients_and_saddle_points) -> None:
    """
    This test is used to check if the types of saddle points are correctly
    identified and if the arrow directions are correct.
    """

    saddle_type, gradient_direction = find_saddle_type_and_gradient_direction(
        *setup_hawk_dove_gradients_and_saddle_points)

    assert not saddle_type[0]  # First
    assert not saddle_type[2]  # Last saddle points are not stable
    assert saddle_type[1]  # Middle saddle point is stable

    # check that there are only 2 arrows
    assert gradient_direction.shape == (2, 2)
    # check tha the first arrows goes from 0 to the stable point
    assert tuple(gradient_direction[0]) == (0., 0.79)
    # check tha the second goes from 1 to the stable point
    assert tuple(gradient_direction[1]) == (1., 0.81)

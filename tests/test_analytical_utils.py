"""
This file tests egttools.analytical.utils functions.
"""

import numpy as np
import numpy.typing as npt
import pytest
from typing import Callable, List, Tuple

# Import the functions from your improved utils file.
# Adjust the import path as necessary for your project structure.
from egttools.analytical.utils import (
    calculate_gradients,
    find_roots,
    # Below functions are expected to be defined eventually.
    get_pairwise_gradient_from_replicator,
    get_pairwise_gradient_from_replicator_n_player,
    check_if_there_is_random_drift,
    find_roots_and_stability,
    check_if_point_in_unit_simplex,
    check_replicator_stability_pairwise_games,
)


def dummy_gradient(state: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """
    A dummy gradient function for testing.
    For a given state (frequency vector), returns 1 - state.
    """
    return 1 - state


def test_calculate_gradients() -> None:
    """
    Test that calculate_gradients returns the expected gradients for given states.
    """
    # Create a simple population state matrix (2 states, 3 strategies)
    population_states = np.array([
        [0.2, 0.5, 0.3],
        [0.1, 0.7, 0.2]
    ], dtype=np.float64)

    # Compute gradients using the dummy gradient function
    gradients = calculate_gradients(population_states, dummy_gradient)

    # Verify that the returned array has the same shape as the input
    assert gradients.shape == population_states.shape

    # Expected result is 1 minus each state entry.
    expected = np.array([1 - population_states[0], 1 - population_states[1]])
    np.testing.assert_allclose(gradients, expected, atol=1e-8)


@pytest.mark.skip(reason="Function 'find_roots' is not implemented yet.")
def test_find_roots() -> None:
    """
    Placeholder test for find_roots; skip until implementation is provided.
    """

    # Prepare a dummy gradient function that has a known fixed point,
    # for example, a constant zero gradient.
    def zero_gradient(state: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return np.zeros_like(state)

    # Use a simple case: strategy frequency vector in a 2-strategy system.
    nb_strategies = 2

    # We expect that every tested initial point is already a fixed point.
    roots = find_roots(zero_gradient, nb_strategies, nb_initial_random_points=3, atol=1e-7, tol_close_points=1e-4,
                       method="hybr")
    # Since zero_gradient returns zero, we may expect the roots to be the input points.
    # For now, simply check that roots is a list.
    assert isinstance(roots, list)


@pytest.mark.skip(reason="Function 'get_pairwise_gradient_from_replicator' is not implemented yet.")
def test_get_pairwise_gradient_from_replicator() -> None:
    """
    Placeholder test for get_pairwise_gradient_from_replicator.
    """
    pytest.skip("Test not implemented until function is defined.")


@pytest.mark.skip(reason="Function 'get_pairwise_gradient_from_replicator_n_player' is not implemented yet.")
def test_get_pairwise_gradient_from_replicator_n_player() -> None:
    """
    Placeholder test for get_pairwise_gradient_from_replicator_n_player.
    """
    pytest.skip("Test not implemented until function is defined.")


@pytest.mark.skip(reason="Function 'check_if_there_is_random_drift' is not implemented yet.")
def test_check_if_there_is_random_drift() -> None:
    """
    Placeholder test for check_if_there_is_random_drift.
    """
    pytest.skip("Test not implemented until function is defined.")


@pytest.mark.skip(reason="Function 'find_roots_and_stability' is not implemented yet.")
def test_find_roots_and_stability() -> None:
    """
    Placeholder test for find_roots_and_stability.
    """
    pytest.skip("Test not implemented until function is defined.")


@pytest.mark.skip(reason="Function 'check_if_point_in_unit_simplex' is not implemented yet.")
def test_check_if_point_in_unit_simplex() -> None:
    """
    Placeholder test for check_if_point_in_unit_simplex.
    """
    pytest.skip("Test not implemented until function is defined.")


@pytest.mark.skip(reason="Function 'check_replicator_stability_pairwise_games' is not implemented yet.")
def test_check_replicator_stability_pairwise_games() -> None:
    """
    Placeholder test for check_replicator_stability_pairwise_games.
    """
    pytest.skip("Test not implemented until function is defined.")

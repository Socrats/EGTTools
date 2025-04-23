# Copyright (c) 2019-2021  Elias Fernandez
#
# This file is part of EGTtools.
#
# EGTtools is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# EGTtools is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with EGTtools.  If not, see <http://www.gnu.org/licenses/>
from typing import Tuple, List, Callable

import numpy
import numpy as np
import numpy.typing as npt
from scipy.linalg import eigvals
from scipy.optimize import root

from . import replicator_equation, replicator_equation_n_player
from .sed_analytical import StochDynamics
from .. import sample_unit_simplex


def get_pairwise_gradient_from_replicator(
        i: int,
        j: int,
        x: int,
        nb_strategies: int,
        payoffs: npt.NDArray[np.float64],
        freq_array: npt.NDArray[np.float64]
) -> float:
    """
    Compute the gradient of selection between two strategies i and j using the replicator equation.

    Parameters
    ----------
    i : int
        Index of the invader strategy.
    j : int
        Index of the resident strategy.
    x : int
        Number of invader individuals in the population.
    nb_strategies : int
        Total number of strategies.
    payoffs : ndarray
        Payoff matrix (assumed shape: [nb_strategies, nb_strategies]).
    freq_array : ndarray
        Current frequency of each strategy.

    Returns
    -------
    float
        Gradient of selection between strategies i and j.
    """
    if freq_array is None:
        freq_array = np.zeros(shape=(nb_strategies,))
    else:
        freq_array[:] = 0

    freq_array[i] = x
    freq_array[j] = 1. - x

    return replicator_equation(freq_array, payoffs)[i]


def get_pairwise_gradient_from_replicator_n_player(
        i: int,
        j: int,
        x: int,
        nb_strategies: int,
        group_size: int,
        payoffs: npt.NDArray[np.float64],
        freq_array: npt.NDArray[np.float64]
) -> float:
    """
    Compute the gradient of selection for an n-player game using the replicator equation.

    Parameters
    ----------
    i : int
        Index of the invader strategy.
    j : int
        Index of the resident strategy.
    x : int
        Number of invader individuals.
    nb_strategies : int
        Number of strategies.
    group_size : int
        Group size in the game.
    payoffs : ndarray
        Payoff matrix shaped (nb_strategies, nb_group_configurations).
    freq_array : ndarray
        Frequency vector for the population.

    Returns
    -------
    float
        Gradient of selection between the two strategies.
    """
    if freq_array is None:
        freq_array = np.zeros(shape=(nb_strategies,))
    else:
        freq_array[:] = 0

    freq_array[i] = x
    freq_array[j] = 1. - x

    return replicator_equation_n_player(freq_array, payoffs, group_size)[i]


def check_if_there_is_random_drift(
        payoff_matrix: npt.NDArray[np.float64],
        group_size: int,
        population_size: int = None,
        beta: float = None,
        nb_points: int = 100,
        atol: float = 1e-8
) -> List[Tuple[int, int]]:
    """
    Check for pairs of strategies that exhibit random drift based on replicator gradients.

    Parameters
    ----------
    payoff_matrix : ndarray
        Payoff matrix of shape (nb_strategies, nb_strategies) or appropriate for n-player games.
    group_size : int
        Size of groups in the game.
    population_size : int, optional default=None
        Total number of individuals in the population.
    beta : float, optional default=None
        Intensity of selection.
    nb_points : int, optional
        Number of discrete invader values to evaluate, by default 100.
    atol : float, optional
        Absolute tolerance to consider the gradient as zero, by default 1e-8.

    Returns
    -------
    List[Tuple[int, int]]
        List of strategy index pairs (i, j) exhibiting drift (i.e. zero gradient across evaluated points).
    """
    # To check if there is random drift, the transition probabilities should be zero

    points = np.linspace(0, 1, nb_points)
    f = None

    if population_size is None:
        freq_array = np.zeros(shape=(payoff_matrix.shape[0]))

        if group_size == 2:
            def gradient_function(i, j, x):
                return get_pairwise_gradient_from_replicator(i, j, x, payoff_matrix.shape[0],
                                                             payoff_matrix,
                                                             freq_array)
        else:
            def gradient_function(i, j, x):
                return get_pairwise_gradient_from_replicator_n_player(i, j, x, payoff_matrix.shape[0],
                                                                      group_size,
                                                                      payoff_matrix,
                                                                      freq_array)

        f = gradient_function
    else:
        if beta is None:
            raise Exception("the beta parameter must be specified!")
        evolver = StochDynamics(payoff_matrix.shape[0], payoff_matrix, population_size, group_size)

        def gradient_function(i, j, x):
            return evolver.gradient_selection(np.floor(x * population_size).astype(np.int64),
                                              i, j,
                                              beta)

        f = gradient_function

    solutions = []
    for row_strategy in range(payoff_matrix.shape[0]):
        # we don't want to look at the case where j==i
        for col_strategy in range(row_strategy + 1, payoff_matrix.shape[0]):
            gradients = []
            for point in points:
                res = f(row_strategy, col_strategy, point)
                gradients.append(res)

            if np.allclose(gradients, 0., atol=atol):
                solutions.append((row_strategy, col_strategy))

    return solutions


def find_roots_and_stability(
        gradient_function: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]],
        nb_strategies: int,
        nb_initial_random_points: int = 100,
        atol: float = 1e-8,
        atol_neg: float = 1e-8,
        atol_pos: float = 1e-8,
        atol_zero: float = 1e-8,
        tol_close_points: float = 1e-3,
        method: str = "hybr"
) -> Tuple[List[npt.NDArray[np.float64]], List[int]]:
    """
    Find fixed points of the gradient function and determine their stability.

    Parameters
    ----------
    gradient_function : Callable
        Function computing the gradient vector for given frequencies.
    nb_strategies : int
        Number of strategies in the game.
    nb_initial_random_points : int, optional
        Number of initial random points sampled from the simplex, by default 100.
    atol : float, optional
        Tolerance for convergence of root finding.
    atol_neg : float
        Threshold for negative eigenvalues to consider stable direction.
    atol_pos : float
        Threshold for positive eigenvalues to consider unstable direction.
    atol_zero : float
        Threshold for eigenvalues close to zero.
    tol_close_points : float
        Tolerance to merge close fixed points.
    method : str
        Root-finding method to use (e.g. 'hybr', 'lm').

    Returns
    -------
    Tuple[List[numpy.ndarray], List[int]]
        A list of fixed points and their associated stability categories.
    """
    # we test all the vertex of the simplex and some random initial points
    initial_states = [[0 if i != j else 1 for i in range(nb_strategies)] for j in range(nb_strategies)]
    for i in range(nb_initial_random_points):
        initial_states.append(sample_unit_simplex(nb_strategies))

    initial_states = np.asarray(initial_states)

    roots = []
    stability = []

    for initial_state in initial_states:
        sol = root(gradient_function, initial_state, method=method, jac=False)

        if sol.success:
            v = sol.x
            if check_if_point_in_unit_simplex(v, atol):
                # only add new fixed points to list
                if not np.array([np.allclose(v, x, atol=tol_close_points) for x in roots]).any():
                    roots.append(v)

                    # now we check the stability of the roots using the jacobian
                    eigenvalues = eigvals(sol.fjac)
                    print(eigenvalues)
                    # If all eigenvalues are negatives or zero it's stable
                    if (eigenvalues.real < -atol_neg).all() or np.array(
                            [np.isclose(el, 0., atol=atol_zero) for el in
                             eigenvalues.real[eigenvalues.real > -atol_neg]]).all():
                        stability.append(1)
                    # If all eigenvalues are positive or zero it's unstable
                    elif (eigenvalues.real > atol_pos).all() or np.array(
                            [np.isclose(el, 0., atol=atol_zero) for el in
                             eigenvalues.real[eigenvalues.real < atol_pos]]).all():
                        stability.append(-1)
                    else:  # saddle point
                        # This is probably wrong, but let's first assume that if we reach here, the point is a saddle
                        stability.append(0)
                        # # we need to check the hessian matrix to find out if the point is a saddle
                        # eigenvalues, _ = np.linalg.eig(sol.hess)
                        # if (eigenvalues > 0).any() and (eigenvalues < 0).any():
                        #     stability.append(0)

    return roots, stability


def check_if_point_in_unit_simplex(
        point: npt.NDArray[np.float64],
        delta: float = 1e-12
) -> bool:
    """
    Check whether a point (in barycentric coordinates) lies inside the unit simplex.

    A point is considered inside the simplex if the sum of its coordinates is approximately 1
    and each coordinate is between 0 and 1 within a specified tolerance.

    Parameters
    ----------
    point : ndarray of shape (n,)
        Barycentric coordinates of the point (i.e., a probability distribution over `n` strategies).
    delta : float, optional
        Tolerance used to determine if the point lies within bounds [0 - delta, 1 + delta].
        Default is 1e-12.

    Returns
    -------
    bool
        True if the point is inside the unit simplex, False otherwise.
    """
    if not np.isclose(np.sum(point), 1.0, atol=1e-2):
        return False

    if not np.all((point > -delta) & (point < 1 + delta)):
        return False

    return True


def calculate_gradients(
        population_states: npt.NDArray[np.float64],
        gradient_function: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]
) -> npt.NDArray[np.float64]:
    """
    Calculate the selection gradients for a list of population states.

    Parameters
    ----------
    population_states : ndarray of shape (m, n)
        A 2D NumPy array where each row corresponds to a population state,
        with `n` strategies and `m` total states.
    gradient_function : Callable[[ndarray], ndarray]
        A function that takes a 1D NumPy array of strategy frequencies (length n)
        and returns a 1D array representing the gradient for each strategy.

    Returns
    -------
    ndarray of shape (m, n)
        A NumPy array where each row contains the gradient of selection
        for the corresponding input population state.
    """
    return np.array([gradient_function(population_states[i]) for i in range(population_states.shape[0])])


def find_roots(
        gradient_function: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]],
        nb_strategies: int,
        nb_initial_random_points: int = 3,
        atol: float = 1e-7,
        tol_close_points: float = 1e-4,
        method: str = "hybr"
) -> List[npt.NDArray[np.float64]]:
    """
    Search for the roots (stationary points) of the given gradient function on the unit simplex.

    Parameters
    ----------
    gradient_function : Callable[[ndarray], ndarray]
        Function returning the gradient of selection given a population state.
    nb_strategies : int
        Number of strategies/types present in the population.
    nb_initial_random_points : int, optional
        Number of additional random starting points on the simplex (besides the simplex vertices), by default 3.
    atol : float, optional
        Absolute tolerance for determining if a point lies in the unit simplex, by default 1e-7.
    tol_close_points : float, optional
        Tolerance for considering two stationary points as identical, by default 1e-4.
    method : str, optional
        Root-finding method to use (passed to `scipy.optimize.root`), by default "hybr".

    Returns
    -------
    List[npt.NDArray[np.float64]]
        A list of stationary points (roots) found, where each point is a 1D NumPy array of shape (nb_strategies,).
    """
    # we test all the vertex of the simplex and some random initial points
    initial_states = [[0 if i != j else 1 for i in range(nb_strategies)] for j in range(nb_strategies)]
    for i in range(nb_initial_random_points):
        initial_states.append(sample_unit_simplex(nb_strategies))

    initial_states = np.asarray(initial_states)

    roots = []

    for initial_state in initial_states:
        sol = root(gradient_function, initial_state, method=method, jac=False)

        if sol.success:
            v = sol.x
            if check_if_point_in_unit_simplex(v, atol):
                # only add new fixed points to list
                if not np.array([np.allclose(v, x, atol=tol_close_points) for x in roots]).any():
                    roots.append(v)

    return roots


def check_replicator_stability_pairwise_games(
        stationary_points: List[npt.NDArray[np.float64]],
        payoff_matrix: npt.NDArray[np.float64],
        atol_neg: float = 1e-4,
        atol_pos: float = 1e-4,
        atol_zero: float = 1e-4
) -> List[int]:
    """
    Determine the stability of stationary points for the replicator equation in pairwise games.

    This function uses the Jacobian of the replicator dynamics to classify each stationary point
    as stable, unstable, or a saddle point based on the signs of the eigenvalues of the Jacobian.

    Parameters
    ----------
    stationary_points : list of ndarray
        A list of stationary points (strategy frequency vectors), each of shape (n,).
    payoff_matrix : ndarray of shape (n, n)
        Payoff matrix representing the interactions between `n` strategies.
    atol_neg : float, optional
        Tolerance for determining if an eigenvalue is considered significantly negative.
    atol_pos : float, optional
        Tolerance for determining if an eigenvalue is considered significantly positive.
    atol_zero : float, optional
        Tolerance for determining if an eigenvalue is effectively zero.

    Returns
    -------
    list of int
        A list where each entry corresponds to the stability classification of a stationary point:

        - `1` for stable (all eigenvalues ≤ 0)
        - `-1` for unstable (all eigenvalues ≥ 0)
        - `0` for saddle (mixed signs among eigenvalues)
    """

    def fitness(i: int, x: npt.NDArray[np.float64]) -> float:
        return float(np.dot(payoff_matrix, x)[i])

    def jacobian(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        ax = payoff_matrix @ x
        avg_fitness = float(x @ ax)
        n = len(x)
        jac = np.empty((n, n), dtype=np.float64)
        for j in range(n):
            for i in range(n):
                if i != j:
                    jac[j, i] = x[i] * (payoff_matrix[i, j] - np.dot(x, payoff_matrix[:, j]))
                else:
                    jac[j, i] = (
                            fitness(i, x)
                            - avg_fitness
                            + x[i] * (payoff_matrix[i, i] - np.dot(x, payoff_matrix[:, i]))
                    )
        return jac

    stability: List[int] = []

    for point in stationary_points:
        eigs = eigvals(jacobian(point)).real

        # Stable: all eigenvalues ≤ 0 (or zero within tolerance)
        if (eigs < -atol_neg).all() or np.all(np.isclose(eigs[eigs > -atol_neg], 0.0, atol=atol_zero)):
            stability.append(1)
        # Unstable: all eigenvalues ≥ 0 (or zero within tolerance)
        elif (eigs > atol_pos).all() or np.all(np.isclose(eigs[eigs < atol_pos], 0.0, atol=atol_zero)):
            stability.append(-1)
        # Mixed signs: saddle point
        else:
            stability.append(0)

    return stability

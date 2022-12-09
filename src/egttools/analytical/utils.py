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
import numpy
import numpy as np
from scipy.optimize import root
from scipy.linalg import eigvals
from typing import Tuple, List, Optional, Callable
from .sed_analytical import StochDynamics, replicator_equation, replicator_equation_n_player
from .. import sample_unit_simplex


def get_pairwise_gradient_from_replicator(i: int, j: int, x: float, nb_strategies: int, payoffs: np.ndarray,
                                          freq_array: Optional[np.ndarray] = None) -> float:
    """
    Calculate the gradient for strategy/type `i` at the edges of the simplex (when there are
    only two strategies in the population `i` and `j`).

    Parameters
    ----------
    i: int
        index of the strategy whose gradient we wish to calculate
    j: int
        index of the other strategy present in the population
    x: float
        frequency of i type
    nb_strategies: int
        total number of strategies in the population
    payoffs: numpy.ndarray
        payoff matrix that defines the expected  payoff of any give strategy against each other
    freq_array: Optional[numpy.ndarray]
        optional vector to store the frequencies of each strategy in the population

    Returns
    -------
    float
        The gradient of strategy i.

    """
    if freq_array is None:
        freq_array = np.zeros(shape=(nb_strategies,))
    else:
        freq_array[:] = 0

    freq_array[i] = x
    freq_array[j] = 1. - x

    return replicator_equation(freq_array, payoffs)[i]


def get_pairwise_gradient_from_replicator_n_player(i: int, j: int, x: float, nb_strategies: int, group_size: int,
                                                   payoffs: np.ndarray,
                                                   freq_array: Optional[np.ndarray] = None) -> float:
    """
    Calculate the gradient for strategy/type `i` at the edges of the simplex (when there are
    only two strategies in the population `i` and `j`).

    Parameters
    ----------
    i: int
        index of the strategy whose gradient we wish to calculate
    j: int
        index of the other strategy present in the population
    x: float
        frequency of i type
    nb_strategies: int
        total number of strategies in the population
    group_size: int
        size of the group
    payoffs: numpy.ndarray
        payoff matrix that defines the expected  payoff of any give strategy against each other
    freq_array: Optional[numpy.ndarray]
        optional vector to store the frequencies of each strategy in the population

    Returns
    -------
    float
        The gradient of strategy i.

    """
    if freq_array is None:
        freq_array = np.zeros(shape=(nb_strategies,))
    else:
        freq_array[:] = 0

    freq_array[i] = x
    freq_array[j] = 1. - x

    return replicator_equation_n_player(freq_array, payoffs, group_size)[i]


def check_if_there_is_random_drift(payoff_matrix: np.ndarray,
                                   population_size: Optional[int] = None,
                                   group_size: int = 2,
                                   beta: Optional[float] = None,
                                   nb_points: int = 10,
                                   atol: float = 1e-7
                                   ) -> List[Tuple[int, int]]:
    """
    Checks if there is random drift along the edge between two strategies in the simplex.

    Parameters
    ----------
    payoff_matrix: numpy.ndarray
        The square matrix of payoffs. If the game is pairwise (group_size = 2) then each entry
        represents the payoff of the row strategy vs the column strategy. If the group_size > 2, then
        each entry should be a function that will return the payoff of the row strategy in a group of size N
        with N-k members of the column strategy. If you only have a matrix where the columns
        represent all possible game states, then you can use the function `egttools.utils.transform_payoffs_to_pairwise`
        to get a matrix in the correct form.

    population_size: Optional[int]
        The size of the population. If this value is not given, we assume that
        we calculate the dynamics in infinite populations using the replicator_equation.

    group_size: int
        The size of the group. If you specify population size, you should also specify this value. By default we assume
        that the game is pairwise.

    beta: Optional[float]
        The intensity of selection.If you specify population size, you should also specify this value.

    nb_points: int
        Number of points for which to check the gradient. It is 10 by default.

    atol: float
        Tolerance to consider a value zero

    Returns
    -------
    List[Tuple[int, int]]
        A list of tuples indicating the undirected edged where there should be random drift.

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


def find_roots_and_stability(gradient_function: Callable[[np.ndarray], np.ndarray], nb_strategies: int,
                             nb_initial_random_points: int = 3,
                             atol: float = 1e-7,
                             atol_neg: float = 1e-4, atol_pos: float = 1e-4,
                             atol_zero: float = 1e-4,
                             tol_close_points: float = 1e-4,
                             method: str = 'hybr') -> Tuple[List[np.array], List[int]]:
    """
    Searches for the roots of the differential equation `gradient_function` and calculates the stability based
    on an estimate of the Jacobian. This estimate is often imprecise which leads to wrong results.

    Parameters
    ----------
    gradient_function: Callable[[np.ndarray], np.ndarray]
        function that returns a numpy.ndarray with the gradient of every strategy/type given a
        current population state.
    nb_strategies: int
        number of strategies/types present in the population.
    nb_initial_random_points: int
        number of random points to use as initial states for the root function. These are
        additional to the vertex of the simplex.
    atol: float
        tolerance for considering that a point is in the simplex.
    atol_neg: float
        tolerance to consider a value negative.
    atol_pos: float
        tolerance to consider a value positive.
    atol_zero: float
        tolerance to determine if a value is zero.
    tol_close_points: float
        tolerance for considering that two points are equal.
    method: str
        one of the options described in `scipy.optimize.root`
        (see https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.root.html)

    Returns
    -------
    Tuple[List[np.array], List[int]]
        A tuple containing the list of roots and a list with 1 indicating stable points, 0 saddle points
        and -1 unstable points.

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


def check_if_point_in_unit_simplex(point: np.ndarray, delta: float = 1e-12) -> bool:
    """
    Checks if a point (in barycentric coordinates) is inside the unit simplex.

    Parameters
    ----------
    point: numpy.ndarray
        The barycentric coordinates of the point.
    delta: float
        Tolerance to consider a point outside the unit simplex.

    Returns
    -------
    bool
        Whether the point is inside the unit simplex.

    """
    if not np.isclose(np.sum(point), 1., atol=1.e-2):
        return False

    if not np.all((point > -delta) & (point < 1 + delta)):  # only if fp in simplex
        return False

    return True


def calculate_gradients(population_states: np.ndarray,
                        gradient_function: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
    """
    Calculates the gradients of selection of each of the states given in `population_states`.

    Parameters
    ----------
    population_states: numpy.ndarray
        A numpy array of shape (m,n) where n is the number of strategies in the population and
        m the number of states for which the gradient should be calculated.
    gradient_function: Callable[[np.ndarray], np.ndarray]
        A function which accepts a vector of shape (n,) containing the frequencies of each
        strategy/type in the population, and returns another vector of shape (n,) containing
        the gradient for each strategy.

    Returns
    -------
    numpy.ndarray
        A numpy array of shape (m,n) containing the gradients for
        each of the input states given in `population_states`.

    """
    return np.array([gradient_function(population_states[i]) for i in range(population_states.shape[0])])


def find_roots(gradient_function: Callable[[np.ndarray], np.ndarray],
               nb_strategies: int, nb_initial_random_points: int = 3,
               atol: float = 1e-7, tol_close_points: float = 1e-4,
               method: str = 'hybr') -> List[np.ndarray]:
    """
    Searches for the roots of the given differential equation.

    Parameters
    ----------
    gradient_function: Callable[[np.ndarray], np.ndarray]
        function that returns a numpy.ndarray with the gradient of every strategy/type given a
        current population state.
    nb_strategies: int
        number of strategies/types present in the population.
    nb_initial_random_points: int
        number of random points to use as initial states for the root function. These are
        additional to the vertex of the simplex.
    atol: float
        tolerance for considering that a point is in the simplex.
    tol_close_points: float
        tolerance for considering that two points are equal.
    method: str
        one of the options described in `scipy.optimize.root`
        (see https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.root.html)

    Returns
    -------
    List[numpy.ndarray]
        A list of tuples with the identified roots/stationary points.

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


def check_replicator_stability_pairwise_games(stationary_points: List[numpy.ndarray], payoff_matrix: numpy.ndarray,
                                              atol_neg: float = 1e-4, atol_pos: float = 1e-4,
                                              atol_zero: float = 1e-4) -> List[int]:
    """
    Calculates the stability of the roots assuming that they are from a system governed by the replicator
    equation (this function uses the Jacobian of the replicator equation in pairwise games to calculate the
    stability).

    Parameters
    ----------
    stationary_points: List[numpy.ndarray]
        a list of stationary points (represented as numpy.ndarray).
    payoff_matrix: numpy.ndarray
        a payoff matrix represented as a numpy.ndarray.
    atol_neg: float
        tolerance to consider a value negative.
    atol_pos: float
        tolerance to consider a value positive.
    atol_zero: float
        tolerance to determine if a value is zero.

    Returns
    -------
    List[int]
        A list of integers indicating the stability of the stationary points for the replicator equation:
        1 - stable
        -1 - unstable
        0 - saddle

    """

    def fitness(i: int, x: np.ndarray):
        return np.dot(payoff_matrix, x)[i]

    # First we build a Jacobian matrix
    def jacobian(x: numpy.ndarray):
        ax = np.dot(payoff_matrix, x)
        avg_fitness = np.dot(x, ax)
        jac = [[x[i] * (payoff_matrix[i, j] - np.dot(x, payoff_matrix[:, j])) if i != j else (
                fitness(i, x) - avg_fitness + x[i] * (payoff_matrix[i, i] - np.dot(x, payoff_matrix[:, i]))) for i in
                range(len(x))] for j in range(len(x))]
        return np.asarray(jac)

    stability = []

    for point in stationary_points:
        # now we check the stability of the roots using the jacobian
        eigenvalues = eigvals(jacobian(point))
        # If all eigenvalues are negatives or zero it's stable
        if (eigenvalues.real < -atol_neg).all() or np.array(
                [np.isclose(el, 0., atol=atol_zero) for el in eigenvalues.real[eigenvalues.real > -atol_neg]]).all():
            stability.append(1)
        # If all eigenvalues are positive or zero it's unstable
        elif (eigenvalues.real > atol_pos).all() or np.array(
                [np.isclose(el, 0., atol=atol_zero) for el in eigenvalues.real[eigenvalues.real < atol_pos]]).all():
            stability.append(-1)
        else:  # saddle point
            # This is probably wrong, but let's first assume that if we reach here, the point is a saddle
            stability.append(0)
            # # we need to check the hessian matrix to find out if the point is a saddle
            # eigenvalues, _ = np.linalg.eig(sol.hess)
            # if (eigenvalues > 0).any() and (eigenvalues < 0).any():
            #     stability.append(0)

    return stability

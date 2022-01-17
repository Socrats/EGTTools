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

import numpy as np
from typing import Tuple, List, Optional
from .sed_analytical import StochDynamics, replicator_equation


def get_pairwise_gradient_from_replicator(i: int, j: int, x: float, nb_strategies: int, payoffs: np.ndarray,
                                          freq_array: Optional[np.ndarray] = None) -> float:
    if freq_array is None:
        freq_array = np.zeros(shape=(nb_strategies,))
    else:
        freq_array[:] = 0

    freq_array[i] = x
    freq_array[j] = 1. - x

    return replicator_equation(freq_array, payoffs)[i]


def check_if_there_is_random_drift(payoff_matrix: np.ndarray,
                                   population_size: Optional[int] = None,
                                   group_size: Optional[int] = 2,
                                   beta: Optional[float] = None,
                                   nb_points: Optional[int] = 10,
                                   atol: Optional[float] = 1e-7
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

    group_size: Optional[int]
        The size of the group. If you specify population size, you should also specify this value. By default we assume
        that the game is pairwise.

    beta: Optional[float]
        The intensity of selection.If you specify population size, you should also specify this value.

    nb_points: Optional[int]
        Number of points for which to check the gradient. It is 10 by default.

    atol: Optional[float]
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

        def gradient_functionn(i, j, x):
            return get_pairwise_gradient_from_replicator(i, j, x, payoff_matrix.shape[0],
                                                         payoff_matrix,
                                                         freq_array)

        f = gradient_functionn
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

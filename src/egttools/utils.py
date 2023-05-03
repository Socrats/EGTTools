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

"""
This python module contains some utility functions
to find saddle points and plot gradients in 2 player, 2 strategy games.
"""
from warnings import warn

import numpy as np
from scipy.linalg import schur, eigvals
from scipy.sparse import csr_matrix, csc_matrix
from typing import Optional, List, Generator, Union, Callable, Dict
from networkx import Graph
from egttools.games import AbstractGame


def find_saddle_type_and_gradient_direction(gradient: Union[List[float], np.ndarray],
                                            saddle_points_idx: Union[List[int], np.ndarray],
                                            offset: float = 0.01):
    """
    Finds whether a saddle point is stable or not. And defines the direction of the
    gradient among stable and unstable points.

    Parameters
    ----------
    gradient : Union[List[float], numpy.ndarray[numpy.float64[1,m]]]
        array containing the gradient of selection for all states of the population
    saddle_points_idx : Union[List[int], numpy.ndarray[numpy.int64[1,m]]]
        array containing the saddle points indices
    offset : float
        offset for the gradient_directions, so that arrows don't overlap with point

    Returns
    -------
    Tuple[List[bool], List[float]]
        Tuple containing an array that indicates the type of saddle points and another array indicating
        the direction of the gradient between unstable and stable points
    """
    saddle_type = []
    gradient_direction = []
    nb_points = len(gradient)
    real_offset = offset * (nb_points - 1)
    for i, point in enumerate(saddle_points_idx):
        if point < nb_points - 1:
            if point > 0:
                if gradient[point + 1] > 0:
                    saddle_type.append(False)
                    gradient_direction.append((point, saddle_points_idx[i + 1] - real_offset))
                elif gradient[point - 1] < 0:
                    saddle_type.append(False)
                    gradient_direction.append((point, saddle_points_idx[i - 1] + real_offset))
                else:
                    saddle_type.append(True)
            else:
                if gradient[point + 1] > 0:
                    saddle_type.append(False)
                    gradient_direction.append((point, saddle_points_idx[i + 1] - real_offset))
                else:
                    saddle_type.append(True)
        else:
            if gradient[point - 1] < 0:
                saddle_type.append(False)
                gradient_direction.append((point, saddle_points_idx[i - 1] + real_offset))
            else:
                saddle_type.append(True)
    saddle_type = np.asarray(saddle_type)
    gradient_direction = np.asarray(gradient_direction) / (nb_points - 1)
    return saddle_type, gradient_direction


def get_payoff_function(strategy_i: int,
                        strategy_j: int,
                        nb_strategies: int,
                        game: AbstractGame) -> Callable[[int, int, Optional[List]], float]:
    """
    Returns a function which gives the payoff of strategy i against strategy j.

    The returned function will return the payoff of strategy i
    given k individuals of strategy i and group_size - k j strategists.

    Parameters
    ----------
    strategy_i : int
        index of strategy i
    strategy_j : int
        index of strategy j
    nb_strategies : int
        Total number of strategies in the population.
    game: egttools.games.AbstractGame
        A game object which contains the method `egttools.games.AbstractGame.payoff` which returns the payoff of
        a strategy given a group composition.

    Returns
    -------
    Callable[[int, int, Optional[List]], float]
        A function which will return the payoff of strategy i,
        given k individuals of strategy i and group_size - k j strategists.
    """

    def get_payoff(k: int, group_size: int, *args: Optional[List]) -> float:
        """
        Returns the payoff given k individuals of strategy i and group_size - k j strategists.

        This function is useful when we can assume that there will only be two
        strategies in a group at any moment in time.

        Parameters
        ----------
        k : int
            Number of players adopting strategy i in the group.
        group_size: int
            Total size of the group.
        args: Optional[List]
            Extra arguments which may be required ot calculate the payoff

        Returns
        -------
        float
            The payoff of strategy i in a group with k i strategists and group_size - k
            j strategists.
        """
        if k > group_size:
            raise Exception("You have indicated a wrong group composition. k must be smaller or equal to group_size.")
        group_composition = np.zeros(shape=(nb_strategies,), dtype=int)
        if strategy_i != strategy_j:
            group_composition[strategy_i] = k
            group_composition[strategy_j] = group_size - k
        else:
            group_composition[strategy_i] = group_size
        return game.payoff(strategy_i, group_composition.tolist())

    return get_payoff


def transform_payoffs_to_pairwise(nb_strategies: int,
                                  game: AbstractGame) -> np.ndarray:
    """
    This function transform a payoff matrix in full format to a pairwise format.

    The transformation should only be done if it is possible to assume that there will always be
    at most 2 strategies in a group at a given time. An example of this would be when calculating
    the Small Mutation Limit (SML) of the Pairwise Moran Process. In this case, we do not need to
    know the payoffs for each strategy for any group composition (i.e., when there are more than
    2 strategies in the group), but only for all possible combinations of each 2 strategies.

    To be able to represent this in a nb_strategies x nb_strategies square matrix, we make each
    entry of the matrix a function, which will return the payoff of the strategy given k players
    adopting strategy i and N - k players adopting strategy j.

    Parameters
    ----------
    nb_strategies : int
        Number of strategies in the population
    game : egttools.games.AbstractGame
        A game object which implements the method `egttools.games.AbstractGame.payoff` which returns the payoff of
        a strategy given a group composition.

    Returns
    -------
    numpy.ndarray[numpy.float64[m,m]]
        Returns the payoff matrix in shape nb_strategies x nb_strategies, and each entry of the payoff
        matrix is a function which will return the payoff of a strategy i against strategy j given
        a group composition with k members of strategy i and N - k members of strategy j.
    """
    return np.asarray(
        [[get_payoff_function(i, j, nb_strategies, game) for j in range(nb_strategies)] for i in range(nb_strategies)])


def calculate_stationary_distribution(transition_matrix: Union[np.ndarray, csr_matrix, csc_matrix]) -> np.ndarray:
    """
    Calculates stationary distribution from a transition matrix of Markov chain.

    The use of this function is not recommended if the matrix is non-Hermitian. Please use
    calculate_stationary_distribution_non_hermitian instead in this case.

    The stationary distribution is the normalized eigenvector associated with the eigenvalue 1

    Parameters
    ----------
    transition_matrix : Union[numpy.ndarray, scipy.sparse.csr_matrix, scipy.sparse.csc_matrix]
        A 2 dimensional transition matrix

    Returns
    -------
    numpy.ndarray
        A 1-dimensional vector containing the stationary distribution

    See Also
    --------
    egttools.utils.calculate_stationary_distribution_non_hermitian

    """
    if (type(transition_matrix) == csr_matrix) or (type(transition_matrix) == csc_matrix):
        tmp = transition_matrix.toarray()
    else:
        tmp = transition_matrix

    # Check if there is any transition with value 1 - this would mean that the game is degenerate
    if np.isclose(tmp, 1., atol=1e-11).any():
        warn(
            "Some of the entries in the transition matrix are close to 1 (with a tolerance of 1e-11). "
            "This could result in more than one eigenvalue of magnitute 1 "
            "(the Markov Chain is degenerate), so please be careful when analysing the results.", RuntimeWarning)

    # calculate stationary distributions using eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(tmp)
    index_stationary = np.argmin(abs(eigenvalues - 1.0))  # look for the element closest to 1 in the list of eigenvalues
    sd = abs(eigenvectors[:, index_stationary].T.real)  # it is essential to access the matrix by column
    return sd / sd.sum()  # normalize


def calculate_stationary_distribution_non_hermitian(
        transition_matrix: Union[np.ndarray, csr_matrix, csc_matrix]) -> np.ndarray:
    """
    Calculates stationary distribution from a transition matrix of Markov chain which is not hermitian.

    The stationary distribution is the normalized eigenvector associated with the eigenvalue 1

    Parameters
    ----------
    transition_matrix : Union[numpy.ndarray, scipy.sparse.csr_matrix, scipy.sparse.csc_matrix]
        A 2 dimensional transition matrix

    Returns
    -------
    numpy.ndarray
        A 1-dimensional vector containing the stationary distribution

    See Also
    --------
    egttools.utils.calculate_stationary_distribution

    """
    if (type(transition_matrix) == csr_matrix) or (type(transition_matrix) == csc_matrix):
        tmp = transition_matrix.toarray()
    else:
        tmp = transition_matrix

    # Check if there is any transition with value 1 - this would mean that the game is degenerate
    if np.isclose(tmp, 1., atol=1e-11).any():
        warn(
            "Some of the entries in the transition matrix are close to 1 (with a tolerance of 1e-11). "
            "This could result in more than one eigenvalue of magnitute 1 "
            "(the Markov Chain is degenerate), so please be careful when analysing the results.", RuntimeWarning)

    # calculate stationary distributions using eigenvalues and eigenvectors
    # noinspection PyTupleAssignmentBalance
    schur_form, eigenvectors = schur(tmp)
    eigenvalues = eigvals(schur_form)
    index_stationary = np.argmin(abs(eigenvalues - 1.0))  # look for the element closest to 1 in the list of eigenvalues
    sd = abs(eigenvectors[:, index_stationary].T.real)  # it is essential to access the matrix by column
    return sd / sd.sum()  # normalize


def combine(values: List[Union[int, str]], length: int) -> Generator:
    """
    Outputs a generator that will generate an ordered list
    with the possible combinations of values with length.

    Each time the generator is called it will output a list
    of length :param length which contains a combinations of the elements in
    the list of values.

    Parameters
    ----------
    values : List
        elements to combine
    length : int
        size of the output

    Returns
    -------
    Generator
        A generator which outputs ordered combinations of value as a list of size
        length


    Examples
    --------
    >>> for value in combine([1, 2], 2):
    ...     print(value)
    [1, 1]
    [2, 1]
    [1, 2]
    [2, 2]
    """
    output = [values[0]] * length
    max_combinations = len(values) ** length
    yield output
    for i in range(1, max_combinations):
        for j in range(length):
            output[j] = values[(i // len(values) ** j) % len(values)]
        yield output


def calculate_nb_unique_combinations(slots_per_bin: Union[List[int], np.ndarray]) -> int:
    """
    Calculates the number of unique combinations given the required number
    of elements of each group, which should be given in List format in the
    slots_per_bin parameter.

    Parameters
    ----------
    slots_per_bin: Union[List[int], numpy.ndarray]
        The list should contain the required number of elements of each group
        that should be combined in a tuple of length sum(slots_per_bin).

    Returns
    -------
    int
        The total number of unique combinations of the groups in the available slots.

    """
    total_slots = np.sum(slots_per_bin)

    divider = np.prod([np.math.factorial(x) for x in slots_per_bin])

    return np.math.factorial(total_slots) // divider


def transform_graph_to_dict_of_lists(graph: Graph) -> Dict[int, List[int]]:
    graph_dictionary = {}
    for node in graph.nodes:
        graph_dictionary[node] = list(graph[node].keys())

    return graph_dictionary

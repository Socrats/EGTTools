# Copyright (c) 2019-2022  Elias Fernandez
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

"""Simplified plotting functions"""
import matplotlib.pyplot as plt
import numpy as np

from typing import Optional, Tuple, Callable, List
from ..games import (AbstractGame, Matrix2PlayerGameHolder, MatrixNPlayerGameHolder, )
from .. import (calculate_nb_states, )
from .helpers import (barycentric_to_xy_coordinates,
                      xy_to_barycentric_coordinates, calculate_stability,
                      find_roots_in_discrete_barycentric_coordinates)
from ..analytical import (replicator_equation, PairwiseComparison, )
from ..analytical import replicator_equation_n_player
from ..analytical.utils import check_if_there_is_random_drift, check_replicator_stability_pairwise_games, find_roots
from ..helpers.vectorized import (vectorized_replicator_equation, vectorized_replicator_equation_n_player,
                                  vectorized_barycentric_to_xy_coordinates)
from . import Simplex2D


def plot_replicator_dynamics_in_simplex(payoff_matrix: np.ndarray,
                                        group_size: int = 2,
                                        nb_points_simplex: int = 100,
                                        nb_of_initial_points_for_root_search: int = 10,
                                        atol: float = 1e-7,
                                        atol_equal: float = 1e-12,
                                        method_find_roots: str = 'hybr',
                                        atol_stability_pos: float = 1e-4, atol_stability_neg: float = 1e-4,
                                        atol_stability_zero: float = 1e-4,
                                        figsize: Tuple[int, int] = (10, 8),
                                        ax: Optional[plt.axis] = None) -> Tuple[Simplex2D,
                                                                                Callable[[np.ndarray, int], np.ndarray],
                                                                                List[np.ndarray],
                                                                                List[np.ndarray],
                                                                                List[int]]:
    """
    Helper function to simplify the plotting of the replicator dynamics in a 2 Simplex.

    Parameters
    ----------
    payoff_matrix: numpy.ndarray
        The square payoff matrix. Group games are still unsupported in the replicator dynamics. This feature will
        soon be added.
    group_size: int
        size of the group
    nb_points_simplex: int
        Number of initial points to draw in the simplex
    nb_of_initial_points_for_root_search: int
        Number of initial points used in the method that searches for the roots of the replicator equation
    atol: float
        Tolerance to consider a value equal to zero. This is used to check if an edge has random drift. By default,
        the tolerance is 1e-7.
    atol_equal: float
        Tolerance to consider two arrays equal.
    method_find_roots: str
        Method used in scipy.optimize.root
    atol_stability_neg: float
        Tolerance used to determine the stability of the roots. This is used to determine whether an
        eigenvalue is negative.
    atol_stability_pos: float
        Tolerance used to determine the stability of the roots. This is used to determine whether an
        eigenvalue is positive.
    atol_stability_zero: float
        Tolerance used to determine the stability of the roots. This is used to determine whether an
        eigenvalue is zero.
    figsize: Tuple[int, int]
        Size of the figure. This parameter is only used if the ax parameter is not defined.
    ax: Optional[matplotlib.pyplot.axis]
        A matplotlib figure axis.

    Returns
    -------
    Tuple[Simplex2D, Callable[[numpy.ndarray, int], numpy.ndarray], List[numpy.ndarray], List[numpy.ndarray], List[int]]

    A tuple with the simplex object which can be used to add more features to the plot, the function that
    can be used to calculate gradients and should be passed to `Simplex2D.draw_trajectory_from_roots` and
    `Simplex2D.draw_scatter_shadow`, a list of the roots in barycentric coordinates, a list of the roots in
    cartesian coordinates and a list of booleans or integers indicating whether the roots are stable.

    """
    if (group_size > 2) and (payoff_matrix.shape[1] == payoff_matrix.shape[0]):
        nb_group_configurations = calculate_nb_states(group_size, payoff_matrix.shape[0])
        if payoff_matrix.shape[1] != nb_group_configurations:
            raise ValueError("The number of columns of the payoff matrix must be equal "
                             "the the number of possible group configurations, when group_size > 2.")

    simplex = Simplex2D(nb_points=nb_points_simplex)
    simplex.add_axis(figsize, ax)

    random_drift = check_if_there_is_random_drift(payoff_matrix, group_size=group_size, atol=atol)
    simplex.add_edges_with_random_drift(random_drift)
    v = np.asarray(xy_to_barycentric_coordinates(simplex.X, simplex.Y, simplex.corners))
    if group_size > 2:
        results = vectorized_replicator_equation_n_player(v, payoff_matrix, group_size)

        def gradient_function(u):
            return replicator_equation_n_player(u, payoff_matrix, group_size)
    else:
        results = vectorized_replicator_equation(v, payoff_matrix)

        def gradient_function(u):
            return replicator_equation(u, payoff_matrix)
    xy_results = vectorized_barycentric_to_xy_coordinates(results, simplex.corners)

    ux = xy_results[:, :, 0].astype(np.float64)
    uy = xy_results[:, :, 1].astype(np.float64)

    simplex.apply_simplex_boundaries_to_gradients(ux, uy)

    roots = find_roots(gradient_function=gradient_function,
                       nb_strategies=payoff_matrix.shape[0],
                       nb_initial_random_points=nb_of_initial_points_for_root_search,
                       atol=atol_equal, tol_close_points=atol_equal, method=method_find_roots)

    roots_xy = [barycentric_to_xy_coordinates(root, corners=simplex.corners) for root in roots]

    if group_size > 2:
        stability = calculate_stability(roots, gradient_function)
        stability = [1 if x is True else -1 for x in stability]
    else:
        stability = check_replicator_stability_pairwise_games(roots, payoff_matrix, atol_neg=atol_stability_neg,
                                                              atol_pos=atol_stability_pos,
                                                              atol_zero=atol_stability_zero)

    return simplex, lambda u, t: gradient_function(u), roots, roots_xy, stability


def plot_pairwise_comparison_rule_dynamics_in_simplex(population_size: int,
                                                      beta: float,
                                                      payoff_matrix: np.ndarray = None,
                                                      game: AbstractGame = None,
                                                      group_size: Optional[int] = 2,
                                                      atol: Optional[float] = 1e-7,
                                                      figsize: Optional[Tuple[int, int]] = (10, 8),
                                                      ax: Optional[plt.axis] = None) -> \
        Tuple[Simplex2D, Callable[[np.ndarray, int], np.ndarray], List[np.ndarray], List[np.ndarray], List[
            bool], AbstractGame, PairwiseComparison]:
    """
    Helper function to simplify the plotting of the moran dynamics in a 2 Simplex.

    Parameters
    ----------
    population_size:
        Size of the finite population.
    beta:
        Intensity of selection.
    payoff_matrix:
        The square payoff matrix. Group games are still unsupported in the replicator dynamics. This feature will
        soon be added.
    game:
        Game that should contain a set of payoff matrices
    group_size:
        Size of the group. By default, we assume that interactions are pairwise (the group size is 2).
    atol:
        Tolerance to consider a value equal to zero. This is used to check if an edge has random drift. By default
        the tolerance is 1e-7.
    figsize:
        Size of the figure. This parameter is only used if the ax parameter is not defined.
    ax:
        A matplotlib figure axis.

    Returns
    -------
    A tuple with the simplex object which can be used to add more features to the plot, the function that
    can be used to calculate gradients and should be passed to `Simplex2D.draw_trajectory_from_roots` and
    `Simplex2D.draw_scatter_shadow`, a list of the roots in barycentric coordinates, a list of the roots in
    cartesian coordinates and a list of booleans indicating whether the roots are stable. It also returns the
    game class (this is important, since a new game is created when passing a payoff matrix, and if not returned,
    a reference to the game instance will disappear, and it will produce a segmentation fault). Finally, it also returns
    a reference to the evolver object.
    """
    if (payoff_matrix is None) and (game is None):
        raise Exception("You need to define either a payoff matrix or a game.")
    elif game is None:
        if (group_size is None) or (group_size < 2):
            raise Exception("group_size not be None and must be >= 2")

        if group_size == 2:
            game = Matrix2PlayerGameHolder(payoff_matrix.shape[0], payoff_matrix)
        else:
            game = MatrixNPlayerGameHolder(payoff_matrix.shape[0], group_size, payoff_matrix)

    simplex = Simplex2D(discrete=True, size=population_size, nb_points=population_size + 1)
    simplex.add_axis(figsize, ax)

    random_drift = check_if_there_is_random_drift(payoff_matrix=game.payoffs(), population_size=population_size,
                                                  group_size=group_size, beta=beta, atol=atol)
    simplex.add_edges_with_random_drift(random_drift)

    v = np.asarray(xy_to_barycentric_coordinates(simplex.X, simplex.Y, simplex.corners))
    v_int = np.floor(v * population_size).astype(np.int64)

    evolver = PairwiseComparison(population_size=population_size, game=game)
    # evolver = StochDynamics(nb_strategies=3, payoffs=payoff_matrix, pop_size=population_size, group_size=group_size)
    result = np.zeros(shape=(v_int.shape[1], v_int.shape[2], 3))
    for i in range(v_int.shape[1]):
        for j in range(v_int.shape[2]):
            result[i, j, :] = evolver.calculate_gradient_of_selection(beta, v_int[:, i, j])

    result = result.swapaxes(0, 1).swapaxes(0, 2)
    xy_results = vectorized_barycentric_to_xy_coordinates(result, simplex.corners)

    ux = xy_results[:, :, 0].astype(np.float64)
    uy = xy_results[:, :, 1].astype(np.float64)

    simplex.apply_simplex_boundaries_to_gradients(ux, uy)

    roots = find_roots_in_discrete_barycentric_coordinates(
        lambda u: population_size * evolver.calculate_gradient_of_selection(beta, u), population_size,
        nb_interior_points=calculate_nb_states(population_size,
                                               3),
        atol=1e-1)
    roots_xy = [barycentric_to_xy_coordinates(x, simplex.corners) for x in roots]
    stability = calculate_stability(roots, lambda u: population_size * evolver.calculate_gradient_of_selection(beta, u))

    return (simplex,
            lambda u, t: population_size * evolver.calculate_gradient_of_selection(beta, u),
            roots, roots_xy, stability, game, evolver)


def plot_pairwise_comparison_rule_dynamics_in_simplex_without_roots(population_size: int,
                                                                    beta: float,
                                                                    payoff_matrix: np.ndarray = None,
                                                                    game: AbstractGame = None,
                                                                    group_size: Optional[int] = 2,
                                                                    figsize: Optional[Tuple[int, int]] = (10, 8),
                                                                    ax: Optional[plt.axis] = None) -> \
        Tuple[Simplex2D, Callable[[np.ndarray, int], np.ndarray], AbstractGame, PairwiseComparison]:
    """
    Helper function to simplify the plotting of the moran dynamics in a 2 Simplex.

    Parameters
    ----------
    population_size:
        Size of the finite population.
    beta:
        Intensity of selection.
    payoff_matrix:
        The square payoff matrix.
    game:
        Game that should contain a set of payoff matrices
    group_size:
        Size of the group. By default, we assume that interactions are pairwise (the group size is 2).
    figsize:
        Size of the figure. This parameter is only used if the ax parameter is not defined.
    ax:
        A matplotlib figure axis.

    Returns
    -------
    A tuple with the simplex object which can be used to add more features to the plot, the function that
    can be used to calculate gradients and should be passed to `Simplex2D.draw_trajectory_from_roots` and
    `Simplex2D.draw_scatter_shadow`, a list of the roots in barycentric coordinates, a list of the roots in
    cartesian coordinates and a list of booleans indicating whether the roots are stable. It also returns the
    game class (this is important, since a new game is created when passing a payoff matrix, and if not returned,
    a reference to the game instance will disappear, and it will produce a segmentation fault). Finally, it also returns
    a reference to the evolver object.
    """
    if (payoff_matrix is None) and (game is None):
        raise Exception("You need to define either a payoff matrix or a game.")
    elif game is None:
        if (group_size is None) or (group_size < 2):
            raise Exception("group_size not be None and must be >= 2")

        if group_size == 2:
            game = Matrix2PlayerGameHolder(payoff_matrix.shape[0], payoff_matrix)
        else:
            game = MatrixNPlayerGameHolder(payoff_matrix.shape[0], group_size, payoff_matrix)

    simplex = Simplex2D(discrete=True, size=population_size, nb_points=population_size + 1)
    simplex.add_axis(figsize, ax)

    v = np.asarray(xy_to_barycentric_coordinates(simplex.X, simplex.Y, simplex.corners))
    v_int = np.floor(v * population_size).astype(np.int64)

    evolver = PairwiseComparison(population_size=population_size, game=game)
    result = np.zeros(shape=(v_int.shape[1], v_int.shape[2], 3))
    for i in range(v_int.shape[1]):
        for j in range(v_int.shape[2]):
            if v_int[:, i, j].sum() <= population_size:
                result[i, j, :] = evolver.calculate_gradient_of_selection(beta, v_int[:, i, j])

    result = result.swapaxes(0, 1).swapaxes(0, 2)
    xy_results = vectorized_barycentric_to_xy_coordinates(result, simplex.corners)

    ux = xy_results[:, :, 0].astype(np.float64)
    uy = xy_results[:, :, 1].astype(np.float64)

    simplex.apply_simplex_boundaries_to_gradients(ux, uy)

    return simplex, lambda u, t: population_size * evolver.calculate_gradient_of_selection(beta, u), game, evolver

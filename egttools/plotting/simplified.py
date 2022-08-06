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
import egttools.games
import matplotlib.pyplot as plt
import numpy as np

from typing import Optional, Tuple, Callable, List
from egttools.numerical import (calculate_nb_states, )
from egttools.plotting.helpers import (barycentric_to_xy_coordinates,
                                       xy_to_barycentric_coordinates, calculate_stationary_points, calculate_stability,
                                       find_roots_in_discrete_barycentric_coordinates)
from egttools.analytical import (replicator_equation, StochDynamics)
from egttools.analytical.utils import check_if_there_is_random_drift, check_replicator_stability_pairwise_games
from egttools.helpers.vectorized import (vectorized_replicator_equation, vectorized_barycentric_to_xy_coordinates)
from egttools.plotting import Simplex2D
from egttools.utils import transform_payoffs_to_pairwise


def plot_replicator_dynamics_in_simplex(payoff_matrix: np.ndarray, atol: float = 1e-7, atol_equal: float = 1e-12,
                                        atol_stability_pos: float = 1e-4, atol_stability_neg: float = 1e-4,
                                        atol_stability_zero: float = 1e-4,
                                        figsize: Tuple[int, int] = (10, 8),
                                        ax: Optional[plt.axis] = None) -> Tuple[Simplex2D,
                                                                                Callable[[np.ndarray, int], np.ndarray],
                                                                                List[np.ndarray],
                                                                                List[np.ndarray],
                                                                                List[int]]:
    """
    Helper function to simplified the plotting of the replicator dynamics in a 2 Simplex.

    Parameters
    ----------
    payoff_matrix: numpy.ndarray
        The square payoff matrix. Group games are still unsupported in the replicator dynamics. This feature will
        soon be added.
    atol: float
        Tolerance to consider a value equal to zero. This is used to check if an edge has random drift. By default
        the tolerance is 1e-7.
    atol_equal: float
        Tolerance to consider two arrays equal.
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
    simplex = Simplex2D()
    simplex.add_axis(figsize, ax)
    random_drift = check_if_there_is_random_drift(payoff_matrix, atol=atol)
    simplex.add_edges_with_random_drift(random_drift)
    v = np.asarray(xy_to_barycentric_coordinates(simplex.X, simplex.Y, simplex.corners))
    results = vectorized_replicator_equation(v, payoff_matrix)
    xy_results = vectorized_barycentric_to_xy_coordinates(results, simplex.corners)

    Ux = xy_results[:, :, 0].astype(np.float64)
    Uy = xy_results[:, :, 1].astype(np.float64)

    simplex.apply_simplex_boundaries_to_gradients(Ux, Uy)
    roots, roots_xy = calculate_stationary_points(simplex.trimesh.x, simplex.trimesh.y, simplex.corners,
                                                  lambda u: replicator_equation(u, payoff_matrix), atol=atol_equal)
    # stability = calculate_stability(roots, lambda u: replicator_equation(u, payoff_matrix))

    stability = check_replicator_stability_pairwise_games(roots, payoff_matrix, atol_neg=atol_stability_neg,
                                                          atol_pos=atol_stability_pos, atol_zero=atol_stability_zero)

    return simplex, lambda u, t: replicator_equation(u, payoff_matrix), roots, roots_xy, stability


def plot_moran_dynamics_in_simplex(population_size: int,
                                   beta: float,
                                   payoff_matrix: np.ndarray = None,
                                   game: egttools.games.AbstractGame = None,
                                   group_size: Optional[int] = 2,
                                   atol: Optional[float] = 1e-7,
                                   figsize: Optional[Tuple[int, int]] = (10, 8),
                                   ax: Optional[plt.axis] = None) -> \
        Tuple[Simplex2D, Callable[[np.ndarray, int], np.ndarray], List[np.ndarray], List[np.ndarray], List[
            bool], StochDynamics]:
    """
    Helper function to simplified the plotting of the moran dynamics in a 2 Simplex.

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
        Size of the group. By default we assume that interactions are pairwise (the group size is 2).
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
    cartesian coordinates and a list of booleans indicating whether the roots are stable.
    """
    if (group_size == 2) and payoff_matrix is None:
        raise Exception("You need to define a payoff matrix for pairwise games.")

    simplex = Simplex2D(discrete=True, size=population_size, nb_points=population_size + 1)
    simplex.add_axis(figsize, ax)

    if group_size > 2:
        payoffs = transform_payoffs_to_pairwise(game.payoffs().shape[0], game)
        payoff_matrix = game.payoffs()
    else:
        payoffs = payoff_matrix

    random_drift = check_if_there_is_random_drift(payoff_matrix=payoffs, population_size=population_size,
                                                  group_size=group_size, beta=beta, atol=atol)
    simplex.add_edges_with_random_drift(random_drift)

    v = np.asarray(xy_to_barycentric_coordinates(simplex.X, simplex.Y, simplex.corners))
    v_int = np.floor(v * population_size).astype(np.int64)

    evolver = StochDynamics(nb_strategies=3, payoffs=payoff_matrix, pop_size=population_size, group_size=group_size)
    result = np.zeros(shape=(v_int.shape[1], v_int.shape[2], 3))
    for i in range(v_int.shape[1]):
        for j in range(v_int.shape[2]):
            result[i, j, :] = evolver.full_gradient_selection_without_mutation(v_int[:, i, j], beta)

    result = result.swapaxes(0, 1).swapaxes(0, 2)
    xy_results = vectorized_barycentric_to_xy_coordinates(result, simplex.corners)

    Ux = xy_results[:, :, 0].astype(np.float64)
    Uy = xy_results[:, :, 1].astype(np.float64)

    simplex.apply_simplex_boundaries_to_gradients(Ux, Uy)

    roots = find_roots_in_discrete_barycentric_coordinates(
        lambda u: population_size * evolver.full_gradient_selection_without_mutation(u, beta), population_size,
        nb_interior_points=calculate_nb_states(population_size,
                                               3),
        atol=1e-1)
    roots_xy = [barycentric_to_xy_coordinates(x, simplex.corners) for x in roots]
    stability = calculate_stability(roots, lambda u: population_size * evolver.full_gradient_selection(u, beta))

    return (simplex,
            lambda u, t: population_size * evolver.full_gradient_selection_without_mutation(u, beta),
            roots, roots_xy, stability, evolver)


def plot_moran_dynamics_in_simplex_without_roots(population_size: int,
                                                 beta: float,
                                                 payoff_matrix: np.ndarray = None,
                                                 game: egttools.games.AbstractGame = None,
                                                 group_size: Optional[int] = 2,
                                                 atol: Optional[float] = 1e-7,
                                                 figsize: Optional[Tuple[int, int]] = (10, 8),
                                                 ax: Optional[plt.axis] = None) -> \
        Tuple[Simplex2D, Callable[[np.ndarray, int], np.ndarray], StochDynamics]:
    """
    Helper function to simplified the plotting of the moran dynamics in a 2 Simplex.

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
        Size of the group. By default we assume that interactions are pairwise (the group size is 2).
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
    cartesian coordinates and a list of booleans indicating whether the roots are stable.
    """
    if (group_size == 2) and payoff_matrix is None:
        raise Exception("You need to define a payoff matrix for pairwise games.")

    simplex = Simplex2D(discrete=True, size=population_size, nb_points=population_size + 1)
    simplex.add_axis(figsize, ax)

    if group_size > 2:
        payoffs = transform_payoffs_to_pairwise(game.payoffs().shape[0], game)
        payoff_matrix = game.payoffs()
    else:
        payoffs = payoff_matrix

    random_drift = check_if_there_is_random_drift(payoff_matrix=payoffs, population_size=population_size,
                                                  group_size=group_size, beta=beta, atol=atol)
    simplex.add_edges_with_random_drift(random_drift)

    v = np.asarray(xy_to_barycentric_coordinates(simplex.X, simplex.Y, simplex.corners))
    v_int = np.floor(v * population_size).astype(np.int64)

    evolver = StochDynamics(nb_strategies=3, payoffs=payoff_matrix, pop_size=population_size, group_size=group_size)
    result = np.zeros(shape=(v_int.shape[1], v_int.shape[2], 3))
    for i in range(v_int.shape[1]):
        for j in range(v_int.shape[2]):
            if v_int[:, i, j].sum() <= population_size:
                result[i, j, :] = evolver.full_gradient_selection_without_mutation(v_int[:, i, j], beta)

    result = result.swapaxes(0, 1).swapaxes(0, 2)
    xy_results = vectorized_barycentric_to_xy_coordinates(result, simplex.corners)

    Ux = xy_results[:, :, 0].astype(np.float64)
    Uy = xy_results[:, :, 1].astype(np.float64)

    simplex.apply_simplex_boundaries_to_gradients(Ux, Uy)

    return simplex, lambda u, t: population_size * evolver.full_gradient_selection_without_mutation(u, beta), evolver

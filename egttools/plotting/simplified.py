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

import matplotlib.pyplot as plt
import numpy as np

from typing import Optional, Tuple, Callable, List
from egttools.numerical import (calculate_nb_states, )
from egttools.plotting.helpers import (barycentric_to_xy_coordinates,
                                       xy_to_barycentric_coordinates, calculate_stationary_points, calculate_stability,
                                       find_roots_in_discrete_barycentric_coordinates)
from egttools.analytical import (replicator_equation, StochDynamics)
from egttools.analytical.utils import check_if_there_is_random_drift
from egttools.helpers.vectorized import (vectorized_replicator_equation, vectorized_barycentric_to_xy_coordinates)
from egttools.plotting import Simplex2D


def plot_replicator_dynamics_in_simplex(payoff_matrix: np.ndarray, atol: Optional[float] = 1e-7,
                                        figsize: Optional[Tuple[int, int]] = (10, 8),
                                        ax: Optional[plt.axis] = None) -> \
        Tuple[Simplex2D, Callable[[np.ndarray, int], np.ndarray], List[np.ndarray], List[np.ndarray], List[bool]]:
    """
    Helper function to simplified the plotting of the replicator dynamics in a 2 Simplex.

    Parameters
    ----------
    payoff_matrix:
        The square payoff matrix. Group games are still unsupported in the replicator dynamics. This feature will
        soon be added.
    atol:
        Tolerance to consider a value equal to zero. This is used to check if an edge has random drift. By default
        the tolerance is 1e-7.
    figsize:
        Size of the figure. This parameter is only used if the ax parameter is not defined.
    ax:
        A matplotlib figure axis.

    Returns
    -------
    Simplex2D
        The simplex object which can be used to add more features to the plot.

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
                                                  lambda u: replicator_equation(u, payoff_matrix))
    stability = calculate_stability(roots, lambda u: replicator_equation(u, payoff_matrix))

    return simplex, lambda u, t: replicator_equation(u, payoff_matrix), roots, roots_xy, stability


def plot_moran_dynamics_in_simplex(payoff_matrix: np.ndarray,
                                   population_size: int,
                                   beta: float,
                                   group_size: Optional[int] = 2,
                                   atol: Optional[float] = 1e-7,
                                   figsize: Optional[Tuple[int, int]] = (10, 8),
                                   ax: Optional[plt.axis] = None) -> \
        Tuple[Simplex2D, Callable[[np.ndarray, int], np.ndarray], List[np.ndarray], List[np.ndarray], List[
            bool], StochDynamics]:
    """

    Parameters
    ----------
    payoff_matrix:
        The square payoff matrix. Group games are still unsupported in the replicator dynamics. This feature will
        soon be added.
    population_size:
        Size of the finite population.
    beta:
        Intensity of selection.
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
    Simplex2D
        The simplex object which can be used to add more features to the plot.
    """
    simplex = Simplex2D(discrete=True, size=population_size, nb_points=population_size + 1)
    simplex.add_axis(figsize, ax)

    random_drift = check_if_there_is_random_drift(payoff_matrix=payoff_matrix, population_size=population_size,
                                                  group_size=group_size, beta=beta, atol=atol)
    simplex.add_edges_with_random_drift(random_drift)

    v = np.asarray(xy_to_barycentric_coordinates(simplex.X, simplex.Y, simplex.corners))
    v_int = np.floor(v * population_size).astype(np.int64)

    evolver = StochDynamics(nb_strategies=3, payoffs=payoff_matrix, pop_size=population_size, group_size=group_size)
    result = np.asarray([[evolver.full_gradient_selection(v_int[:, i, j], beta) for j in range(v_int.shape[2])] for i in
                         range(v_int.shape[1])]).swapaxes(0, 1).swapaxes(0, 2)
    xy_results = vectorized_barycentric_to_xy_coordinates(result, simplex.corners)

    Ux = xy_results[:, :, 0].astype(np.float64)
    Uy = xy_results[:, :, 1].astype(np.float64)

    simplex.apply_simplex_boundaries_to_gradients(Ux, Uy)

    roots = find_roots_in_discrete_barycentric_coordinates(
        lambda u: population_size * evolver.full_gradient_selection(u, beta), population_size,
        nb_interior_points=calculate_nb_states(population_size,
                                               3),
        atol=1e-1)
    roots_xy = [barycentric_to_xy_coordinates(x, simplex.corners) for x in roots]
    stability = calculate_stability(roots, lambda u: population_size * evolver.full_gradient_selection(u, beta))

    return (simplex,
            lambda u, t: population_size * evolver.full_gradient_selection_without_mutation(u, beta),
            roots, roots_xy, stability, evolver)

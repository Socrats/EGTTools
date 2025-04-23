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
from typing import Optional, Tuple, Callable, List, Literal

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from . import Simplex2D
from .helpers import (barycentric_to_xy_coordinates,
                      xy_to_barycentric_coordinates, calculate_stability,
                      find_roots_in_discrete_barycentric_coordinates)
from .. import (calculate_nb_states, )
from ..analytical import (replicator_equation, PairwiseComparison, )
from ..analytical import replicator_equation_n_player
from ..analytical.utils import check_if_there_is_random_drift, check_replicator_stability_pairwise_games, find_roots
from ..games import (AbstractGame, Matrix2PlayerGameHolder, MatrixNPlayerGameHolder, )
from ..helpers.vectorized import (vectorized_replicator_equation, vectorized_replicator_equation_n_player,
                                  vectorized_barycentric_to_xy_coordinates)


def plot_replicator_dynamics_in_simplex(
        payoff_matrix: Optional[NDArray[np.float64]] = None,
        game: Optional[AbstractGame] = None,
        group_size: int = 2,
        nb_points_simplex: int = 100,
        nb_of_initial_points_for_root_search: int = 10,
        atol: float = 1e-7,
        atol_equal: float = 1e-12,
        method_find_roots: str = 'hybr',
        atol_stability_pos: float = 1e-4,
        atol_stability_neg: float = 1e-4,
        atol_stability_zero: float = 1e-4,
        figsize: Tuple[int, int] = (10, 8),
        ax: Optional[plt.Axes] = None,
        stability_mode: Literal["bool", "int"] = "int"
) -> Tuple[
    Simplex2D,
    Callable[[NDArray[np.float64], int], NDArray[np.float64]],
    List[NDArray[np.float64]],
    List[NDArray[np.float64]],
    List[int] | List[bool]
]:
    """
    Plot the replicator dynamics on a 2D simplex for 2- or N-player matrix games.

    Parameters
    ----------
    payoff_matrix : Optional[NDArray[np.float64]], default=None
        The payoff matrix of the game (used if `game` is not provided).
    game : Optional[AbstractGame], default=None
        A game object from which the payoff matrix can be retrieved via `.payoffs()`.
    group_size : int
        Group size. Use 2 for pairwise interactions.
    nb_points_simplex : int
        Sampling resolution for vector field.
    nb_of_initial_points_for_root_search : int
        Number of starting points for root search.
    atol : float
        Tolerance for edge drift detection.
    atol_equal : float
        Tolerance for root comparison and filtering.
    method_find_roots : str
        Root-finding algorithm (e.g. 'hybr').
    atol_stability_pos : float
        Threshold for instability detection (positive eigenvalues).
    atol_stability_neg : float
        Threshold for stability detection (negative eigenvalues).
    atol_stability_zero : float
        Threshold for zero eigenvalues.
    figsize : Tuple[int, int]
        Figure size.
    ax : Optional[plt.Axes]
        Optional existing axis.
    stability_mode : {'bool', 'int'}, default='int'
        Whether to return stability as booleans or integers:
        - 'bool': True for stable, False otherwise
        - 'int': 1 (stable), 0 (saddle), -1 (unstable)

    Returns
    -------
    Tuple[Simplex2D, Callable, List[NDArray], List[NDArray], List[int] or List[bool]]
        Simplex plot object, replicator function, roots in barycentric and cartesian,
        and root stability indicators.


    Notes
    -----
    The returned gradient function is wrapped to include a dummy t parameter, enabling direct
    use with ODE solvers such as scipy.integrate.odeint. It can also be passed to `egttools.plotting.Simplex2D`
    to draw trajectories over the simplex.
    """
    if payoff_matrix is None and game is None:
        raise ValueError("You must provide either a payoff matrix or a game object.")

    if game is not None:
        payoff_matrix = game.payoffs()
        group_size = game.group_size()

    if (group_size > 2) and (payoff_matrix.shape[1] == payoff_matrix.shape[0]):
        nb_group_configurations = calculate_nb_states(group_size, payoff_matrix.shape[0])
        if payoff_matrix.shape[1] != nb_group_configurations:
            raise ValueError("Mismatch between payoff matrix shape and number of group configurations.")

    simplex = Simplex2D(nb_points=nb_points_simplex)
    simplex.add_axis(figsize, ax)

    random_drift = check_if_there_is_random_drift(payoff_matrix, group_size=group_size, atol=atol)
    simplex.add_edges_with_random_drift(random_drift)

    v = np.asarray(xy_to_barycentric_coordinates(simplex.X, simplex.Y, simplex.corners))

    if group_size > 2:
        results = vectorized_replicator_equation_n_player(v, payoff_matrix, group_size)

        def gradient_function(u: NDArray[np.float64]) -> NDArray[np.float64]:
            return replicator_equation_n_player(u, payoff_matrix, group_size)
    else:
        results = vectorized_replicator_equation(v, payoff_matrix)

        def gradient_function(u: NDArray[np.float64]) -> NDArray[np.float64]:
            return replicator_equation(u, payoff_matrix)

    xy_results = vectorized_barycentric_to_xy_coordinates(results, simplex.corners)

    ux = xy_results[:, :, 0].astype(np.float64)
    uy = xy_results[:, :, 1].astype(np.float64)

    simplex.apply_simplex_boundaries_to_gradients(ux, uy)

    roots = find_roots(
        gradient_function=gradient_function,
        nb_strategies=payoff_matrix.shape[0],
        nb_initial_random_points=nb_of_initial_points_for_root_search,
        atol=atol_equal,
        tol_close_points=atol_equal,
        method=method_find_roots
    )

    roots_xy = [barycentric_to_xy_coordinates(root, corners=simplex.corners) for root in roots]

    if group_size > 2:
        stability = calculate_stability(roots, gradient_function, return_mode=stability_mode)
    else:
        stability = check_replicator_stability_pairwise_games(
            roots, payoff_matrix,
            atol_neg=atol_stability_neg,
            atol_pos=atol_stability_pos,
            atol_zero=atol_stability_zero
        )

    return simplex, lambda u, t: gradient_function(u), roots, roots_xy, stability


def plot_pairwise_comparison_rule_dynamics_in_simplex(
        population_size: int,
        beta: float,
        payoff_matrix: Optional[NDArray[np.float64]] = None,
        game: Optional[AbstractGame] = None,
        group_size: Optional[int] = 2,
        atol: float = 1e-7,
        figsize: Tuple[int, int] = (10, 8),
        ax: Optional[plt.Axes] = None,
        stability_mode: Literal["bool", "int"] = "int"
) -> Tuple[
    Simplex2D,
    Callable[[NDArray[np.float64], int], NDArray[np.float64]],
    List[NDArray[np.float64]],
    List[NDArray[np.float64]],
    List[bool] | List[int],
    AbstractGame,
    PairwiseComparison
]:
    """
    Plot dynamics of a finite population using the pairwise comparison rule on a 2D simplex.

    This method visualizes the direction of selection under a discrete dynamics model
    for finite populations using pairwise comparison.

    Parameters
    ----------
    population_size : int
        Number of individuals in the population.
    beta : float
        Selection strength parameter (0 = neutral drift).
    payoff_matrix : Optional[NDArray[np.float64]], default=None
        Matrix of payoffs. If not provided, a `game` must be supplied.
    game : Optional[AbstractGame], default=None
        A game object encoding payoff logic. Used if `payoff_matrix` is not given.
    group_size : Optional[int], default=2
        Number of interacting players.
    atol : float, default=1e-7
        Tolerance used to identify edges with random drift.
    figsize : Tuple[int, int], default=(10, 8)
        Size of the figure, only used if `ax` is not provided.
    ax : Optional[plt.Axes], default=None
        Matplotlib axis to plot on.
    stability_mode : {'bool', 'int'}, default='int'
        Whether to return boolean or ternary stability classification:
        - 'bool': True = stable, False = not stable
        - 'int': 1 = stable, 0 = saddle, -1 = unstable

    Returns
    -------
    Tuple[Simplex2D, Callable, List[NDArray], List[NDArray], List[bool] or List[int], AbstractGame, PairwiseComparison]
        - The `Simplex2D` plot object.
        - The selection gradient function for dynamics simulation.
        - List of equilibrium points in barycentric coordinates.
        - Same list in Cartesian coordinates.
        - List of stability indicators (bool or int).
        - The instantiated game object.
        - The evolver used for computing selection gradients.
    """

    if payoff_matrix is None and game is None:
        raise ValueError("You must define either a payoff matrix or a game.")
    elif game is None:
        if group_size is None or group_size < 2:
            raise ValueError("group_size must be >= 2 when constructing a game from a matrix.")

        if group_size == 2:
            game = Matrix2PlayerGameHolder(payoff_matrix.shape[0], payoff_matrix)
        else:
            game = MatrixNPlayerGameHolder(payoff_matrix.shape[0], group_size, payoff_matrix)

    payoff_matrix = game.payoffs()

    simplex = Simplex2D(discrete=True, size=population_size, nb_points=population_size + 1)
    simplex.add_axis(figsize, ax)

    random_drift = check_if_there_is_random_drift(
        payoff_matrix=payoff_matrix,
        population_size=population_size,
        group_size=group_size,
        beta=beta,
        atol=atol
    )
    simplex.add_edges_with_random_drift(random_drift)

    v = np.asarray(xy_to_barycentric_coordinates(simplex.X, simplex.Y, simplex.corners))
    v_int = np.floor(v * population_size).astype(np.int64)

    evolver = PairwiseComparison(population_size=population_size, game=game)
    result = np.zeros(shape=(v_int.shape[1], v_int.shape[2], 3))
    for i in range(v_int.shape[1]):
        for j in range(v_int.shape[2]):
            result[i, j, :] = evolver.calculate_gradient_of_selection(beta, v_int[:, i, j])

    result = result.swapaxes(0, 1).swapaxes(0, 2)
    xy_results = vectorized_barycentric_to_xy_coordinates(result, simplex.corners)

    ux = xy_results[:, :, 0].astype(np.float64)
    uy = xy_results[:, :, 1].astype(np.float64)

    simplex.apply_simplex_boundaries_to_gradients(ux, uy)

    gradient_fn = lambda u: population_size * evolver.calculate_gradient_of_selection(beta, u)

    roots = find_roots_in_discrete_barycentric_coordinates(
        gradient_fn,
        population_size,
        nb_interior_points=calculate_nb_states(population_size, 3),
        atol=1e-1
    )
    roots_xy = [barycentric_to_xy_coordinates(x, simplex.corners) for x in roots]

    stability = calculate_stability(roots, gradient_fn, return_mode=stability_mode)

    return (
        simplex,
        lambda u, t: gradient_fn(u),
        roots,
        roots_xy,
        stability,
        game,
        evolver
    )


def plot_pairwise_comparison_rule_dynamics_in_simplex_without_roots(
        population_size: int,
        beta: float,
        payoff_matrix: Optional[NDArray[np.float64]] = None,
        game: Optional[AbstractGame] = None,
        group_size: Optional[int] = 2,
        figsize: Tuple[int, int] = (10, 8),
        ax: Optional[plt.Axes] = None
) -> Tuple[
    Simplex2D,
    Callable[[NDArray[np.float64], int], NDArray[np.float64]],
    AbstractGame,
    PairwiseComparison
]:
    """
    Plot dynamics on the simplex under the pairwise comparison rule, without computing roots.

    This version is faster and suited for visualization-only use cases.
    It skips root-finding and stability classification.

    Parameters
    ----------
    population_size : int
        Number of individuals in the population.
    beta : float
        Strength of selection.
    payoff_matrix : Optional[NDArray[np.float64]], default=None
        The game payoff matrix (used if `game` is not provided).
    game : Optional[AbstractGame], default=None
        Game object. If not provided, one will be created from `payoff_matrix`.
    group_size : Optional[int], default=2
        Number of players interacting simultaneously (used only if constructing a game).
    figsize : Tuple[int, int], default=(10, 8)
        Size of the figure.
    ax : Optional[plt.Axes], default=None
        Optional matplotlib axis to draw on.

    Returns
    -------
    Tuple[Simplex2D, Callable, AbstractGame, PairwiseComparison]
        - The `Simplex2D` plot object.
        - The gradient function (for use in trajectory plotting).
        - The `AbstractGame` object used.
        - The `PairwiseComparison` evolver instance.
    """
    if payoff_matrix is None and game is None:
        raise ValueError("You must define either a payoff matrix or a game.")
    elif game is None:
        if group_size is None or group_size < 2:
            raise ValueError("group_size must be >= 2 when constructing a game from a matrix.")

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

    return (
        simplex,
        lambda u, t: population_size * evolver.calculate_gradient_of_selection(beta, u),
        game,
        evolver
    )

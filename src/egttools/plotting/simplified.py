# Copyright (c) 2019-2026  Elias Fernandez
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

"""Simplified plotting functions."""
from typing import Optional, Tuple, Callable, List, Literal, Union

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from . import Simplex2D
from .helpers import (
    barycentric_to_xy_coordinates,
    xy_to_barycentric_coordinates,
    calculate_stability,
    find_roots_in_discrete_barycentric_coordinates,
)
from .. import calculate_nb_states
from ..analytical import PairwiseComparison, replicator_equation, replicator_equation_n_player
from ..analytical.utils import (
    check_if_there_is_random_drift,
    check_replicator_stability_pairwise_games,
    find_roots,
)
from ..games import (
    AbstractGame,
    AbstractReplicatorGame,
    Matrix2PlayerGameHolder,
    MatrixNPlayerGameHolder,
)
from ..helpers.vectorized import vectorized_barycentric_to_xy_coordinates

ReplicatorInputGame = Optional[Union[AbstractGame, AbstractReplicatorGame]]


def _extract_group_size(game, default: int = 2) -> int:
    group_size_attr: int | Callable[[], int] = getattr(game, "group_size", None)
    if group_size_attr is None:
        return default
    if callable(group_size_attr):
        return group_size_attr()
    return int(group_size_attr)


def _normalize_replicator_inputs(
        payoff_matrix: Optional[NDArray[np.float64]],
        game: ReplicatorInputGame,
        group_size: int,
) -> tuple[Optional[NDArray[np.float64]], ReplicatorInputGame, int, bool]:
    """
    Normalize and validate inputs for replicator simplex plotting.

    Returns
    -------
    payoff_matrix : Optional[NDArray[np.float64]]
        Normalized payoff matrix if available.
    game : Optional[AbstractGame | AbstractReplicatorGame]
        Input game object.
    group_size : int
        Effective group size.
    use_replicator_game_logic : bool
        True only if the provided game is an AbstractReplicatorGame.
    """
    if payoff_matrix is None and game is None:
        raise ValueError("You must provide either a payoff matrix or a game object.")

    use_replicator_game_logic = False

    if game is not None:
        if isinstance(game, AbstractReplicatorGame):
            use_replicator_game_logic = True
            group_size = _extract_group_size(game, default=2)
            if payoff_matrix is None:
                try:
                    payoff_matrix = np.asarray(game.payoffs(), dtype=np.float64)
                except Exception:
                    payoff_matrix = None
        elif isinstance(game, AbstractGame):
            if payoff_matrix is None:
                payoff_matrix = np.asarray(game.payoffs(), dtype=np.float64)
            group_size = _extract_group_size(game, default=2)
        else:
            raise TypeError(
                "game must be an instance of egttools.games.AbstractGame or "
                "egttools.games.AbstractReplicatorGame."
            )

    if payoff_matrix is not None:
        payoff_matrix = np.asarray(payoff_matrix, dtype=np.float64)
        if payoff_matrix.ndim != 2:
            raise ValueError("payoff_matrix must be a 2D array.")

        nb_strategies = payoff_matrix.shape[0]

        if group_size == 2:
            if payoff_matrix.shape[1] != nb_strategies:
                raise ValueError(
                    "For pairwise games, payoff_matrix must have shape "
                    "(nb_strategies, nb_strategies)."
                )
        else:
            nb_group_configurations = calculate_nb_states(group_size, nb_strategies)
            if payoff_matrix.shape[1] != nb_group_configurations:
                raise ValueError(
                    "Mismatch between payoff matrix shape and number of group configurations. "
                    f"Expected {nb_group_configurations} columns for group_size={group_size} "
                    f"and nb_strategies={nb_strategies}, got {payoff_matrix.shape[1]}."
                )

    if use_replicator_game_logic and payoff_matrix is not None:
        if payoff_matrix.shape[0] != game.nb_strategies():
            raise ValueError(
                "Mismatch between payoff_matrix and game: the number of strategies differs."
            )

    return payoff_matrix, game, group_size, use_replicator_game_logic


def _make_replicator_gradient_function(
        payoff_matrix: Optional[NDArray[np.float64]],
        game: ReplicatorInputGame,
        group_size: int,
        use_replicator_game_logic: bool,
) -> Callable[[NDArray[np.float64]], NDArray[np.float64]]:
    """
    Build the appropriate gradient function for replicator dynamics.
    """
    if use_replicator_game_logic:
        if game is None:
            raise ValueError("Internal error: replicator-game logic selected but game is None.")
        if group_size > 2:
            return lambda u: np.asarray(replicator_equation_n_player(u, game), dtype=np.float64)
        return lambda u: np.asarray(replicator_equation(u, game), dtype=np.float64)

    if payoff_matrix is None:
        raise ValueError(
            "A payoff matrix is required unless an AbstractReplicatorGame is provided."
        )

    if group_size > 2:
        return lambda u: np.asarray(
            replicator_equation_n_player(u, payoff_matrix, group_size), dtype=np.float64
        )
    return lambda u: np.asarray(replicator_equation(u, payoff_matrix), dtype=np.float64)


def _vectorized_replicator_from_barycentric_grid(
        barycentric_grid: NDArray[np.float64],
        payoff_matrix: Optional[NDArray[np.float64]],
        game: ReplicatorInputGame,
        group_size: int,
        use_replicator_game_logic: bool,
) -> NDArray[np.float64]:
    """
    Evaluate the replicator gradient on a barycentric grid.
    """
    if barycentric_grid.shape[0] != 3:
        raise ValueError("This plotting helper currently supports only 3 strategies.")

    gradient_function = _make_replicator_gradient_function(
        payoff_matrix=payoff_matrix,
        game=game,
        group_size=group_size,
        use_replicator_game_logic=use_replicator_game_logic,
    )

    result = np.zeros_like(barycentric_grid, dtype=np.float64)
    nrows, ncols = barycentric_grid.shape[1], barycentric_grid.shape[2]

    for i in range(nrows):
        for j in range(ncols):
            u = barycentric_grid[:, i, j]
            if np.any(u < -1e-12):
                continue
            if not np.isclose(u.sum(), 1.0, atol=1e-8):
                continue
            result[:, i, j] = gradient_function(u)

    return result


def _edge_random_drift_from_gradient(
        gradient_function: Callable[[NDArray[np.float64]], NDArray[np.float64]],
        atol: float = 1e-7,
) -> NDArray[np.bool_]:
    """
    Detect random drift on the simplex edges using the gradient function directly.
    """
    edge_points = [
        np.array([[1.0, 0.0, 0.0],
                  [0.5, 0.5, 0.0],
                  [0.0, 1.0, 0.0]], dtype=np.float64),
        np.array([[1.0, 0.0, 0.0],
                  [0.5, 0.0, 0.5],
                  [0.0, 0.0, 1.0]], dtype=np.float64),
        np.array([[0.0, 1.0, 0.0],
                  [0.0, 0.5, 0.5],
                  [0.0, 0.0, 1.0]], dtype=np.float64),
    ]

    drift = []
    for pts in edge_points:
        is_zero = True
        for u in pts:
            g = np.asarray(gradient_function(u), dtype=np.float64)
            if not np.all(np.abs(g) <= atol):
                is_zero = False
                break
        drift.append(is_zero)

    return np.asarray(drift, dtype=bool)


def plot_replicator_dynamics_in_simplex(
        payoff_matrix: Optional[NDArray[np.float64]] = None,
        game: ReplicatorInputGame = None,
        group_size: int = 2,
        nb_points_simplex: int = 100,
        nb_of_initial_points_for_root_search: int = 10,
        atol: float = 1e-7,
        atol_equal: float = 1e-12,
        method_find_roots: str = "hybr",
        atol_stability_pos: float = 1e-4,
        atol_stability_neg: float = 1e-4,
        atol_stability_zero: float = 1e-4,
        figsize: Tuple[int, int] = (10, 8),
        ax: Optional[plt.Axes] = None,
        stability_mode: Literal["bool", "int"] = "int",
) -> Tuple[
    Simplex2D,
    Callable[[NDArray[np.float64], int], NDArray[np.float64]],
    List[NDArray[np.float64]],
    List[NDArray[np.float64]],
    List[int] | List[bool],
]:
    """
    Plot the replicator dynamics on a 2D simplex for 3-strategy 2- or N-player games.

    Backward compatibility:
    - If `game` is an `AbstractGame`, the function extracts `game.payoffs()` and uses the
      original matrix-based logic.
    - If `game` is an `AbstractReplicatorGame`, the function uses the new game-based
      replicator logic.
    """
    payoff_matrix, game, group_size, use_replicator_game_logic = _normalize_replicator_inputs(
        payoff_matrix=payoff_matrix,
        game=game,
        group_size=group_size,
    )

    if use_replicator_game_logic:
        nb_strategies = game.nb_strategies()
    elif payoff_matrix is not None:
        nb_strategies = payoff_matrix.shape[0]
    else:
        raise ValueError("Could not determine the number of strategies.")

    if nb_strategies != 3:
        raise ValueError(
            "plot_replicator_dynamics_in_simplex currently supports only 3 strategies."
        )

    simplex = Simplex2D(nb_points=nb_points_simplex)
    simplex.add_axis(figsize, ax)

    gradient_function = _make_replicator_gradient_function(
        payoff_matrix=payoff_matrix,
        game=game,
        group_size=group_size,
        use_replicator_game_logic=use_replicator_game_logic,
    )

    if payoff_matrix is not None and not use_replicator_game_logic:
        random_drift = check_if_there_is_random_drift(
            payoff_matrix=payoff_matrix,
            group_size=group_size,
            atol=atol,
        )
    else:
        random_drift = _edge_random_drift_from_gradient(gradient_function, atol=atol)

    simplex.add_edges_with_random_drift(random_drift)

    v = np.asarray(
        xy_to_barycentric_coordinates(simplex.X, simplex.Y, simplex.corners),
        dtype=np.float64,
    )

    results = _vectorized_replicator_from_barycentric_grid(
        barycentric_grid=v,
        payoff_matrix=payoff_matrix,
        game=game,
        group_size=group_size,
        use_replicator_game_logic=use_replicator_game_logic,
    )

    xy_results = vectorized_barycentric_to_xy_coordinates(results, simplex.corners)

    ux = xy_results[:, :, 0].astype(np.float64)
    uy = xy_results[:, :, 1].astype(np.float64)

    simplex.apply_simplex_boundaries_to_gradients(ux, uy)

    roots = find_roots(
        gradient_function=gradient_function,
        nb_strategies=nb_strategies,
        nb_initial_random_points=nb_of_initial_points_for_root_search,
        atol=atol_equal,
        tol_close_points=atol_equal,
        method=method_find_roots,
    )

    roots_xy = [barycentric_to_xy_coordinates(root, corners=simplex.corners) for root in roots]

    if payoff_matrix is not None and group_size == 2 and not use_replicator_game_logic:
        stability = check_replicator_stability_pairwise_games(
            roots,
            payoff_matrix,
            atol_neg=atol_stability_neg,
            atol_pos=atol_stability_pos,
            atol_zero=atol_stability_zero,
        )
    else:
        stability = calculate_stability(
            roots,
            gradient_function,
            atol=atol_stability_zero,
            return_mode=stability_mode,
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
        stability_mode: Literal["bool", "int"] = "int",
) -> Tuple[
    Simplex2D,
    Callable[[NDArray[np.float64], int], NDArray[np.float64]],
    List[NDArray[np.float64]],
    List[NDArray[np.float64]],
    List[bool] | List[int],
    AbstractGame,
    PairwiseComparison,
]:
    """
    Plot dynamics of a finite population using the pairwise comparison rule on a 2D simplex.
    """
    if payoff_matrix is None and game is None:
        raise ValueError("You must define either a payoff matrix or a game.")
    elif game is None:
        if payoff_matrix is None:
            raise ValueError("payoff_matrix must be provided when game is None.")
        if group_size is None or group_size < 2:
            raise ValueError("group_size must be >= 2 when constructing a game from a matrix.")

        payoff_matrix = np.asarray(payoff_matrix, dtype=np.float64)
        if group_size == 2:
            game = Matrix2PlayerGameHolder(payoff_matrix.shape[0], payoff_matrix)
        else:
            game = MatrixNPlayerGameHolder(payoff_matrix.shape[0], group_size, payoff_matrix)

    payoff_matrix = np.asarray(game.payoffs(), dtype=np.float64)
    group_size = _extract_group_size(game, default=2)

    if game.nb_strategies() != 3:
        raise ValueError(
            "plot_pairwise_comparison_rule_dynamics_in_simplex currently supports only 3 strategies."
        )

    simplex = Simplex2D(discrete=True, size=population_size, nb_points=population_size + 1)
    simplex.add_axis(figsize, ax)

    random_drift = check_if_there_is_random_drift(
        payoff_matrix=payoff_matrix,
        population_size=population_size,
        group_size=group_size,
        beta=beta,
        atol=atol,
    )
    simplex.add_edges_with_random_drift(random_drift)

    v = np.asarray(xy_to_barycentric_coordinates(simplex.X, simplex.Y, simplex.corners))
    v_int = np.floor(v * population_size).astype(np.int64)

    evolver = PairwiseComparison(population_size=population_size, game=game)
    result = np.zeros(shape=(v_int.shape[1], v_int.shape[2], 3), dtype=np.float64)

    if mu is None:
      gradient_fn = lambda u: evolver.calculate_gradient_of_selection(state=u, beta=beta)
    else:
      gradient_fn = lambda u: evolver.calculate_gradient_of_selection_with_mutation(state=u, beta=beta, mu=mu)

    for i in range(v_int.shape[1]):
        for j in range(v_int.shape[2]):
            if not (v_int[:, i, j] < 0).any() and v_int[:, i, j].sum() <= population_size:
                result[i, j, :] = gradient_fn(v_int[:, i, j])

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
        atol=1e-1,
    )
    roots_xy = [barycentric_to_xy_coordinates(x, simplex.corners) for x in roots]

    stability = calculate_stability(roots, gradient_fn, return_mode=stability_mode)

    return simplex, lambda u, t: gradient_fn(u), roots, roots_xy, stability, game, evolver


def plot_pairwise_comparison_rule_dynamics_in_simplex_without_roots(
        population_size: int,
        beta: float,
        mu: Optional[float] = None,
        payoff_matrix: Optional[NDArray[np.float64]] = None,
        game: Optional[AbstractGame] = None,
        group_size: Optional[int] = 2,
        figsize: Tuple[int, int] = (10, 8),
        ax: Optional[plt.Axes] = None,
) -> Tuple[
    Simplex2D,
    Callable[[NDArray[np.float64], int], NDArray[np.float64]],
    AbstractGame,
    PairwiseComparison,
]:
    """
    Plot dynamics on the simplex under the pairwise comparison rule, without computing roots.
    """
    if payoff_matrix is None and game is None:
        raise ValueError("You must define either a payoff matrix or a game.")
    elif game is None:
        if payoff_matrix is None:
            raise ValueError("payoff_matrix must be provided when game is None.")
        if group_size is None or group_size < 2:
            raise ValueError("group_size must be >= 2 when constructing a game from a matrix.")

        payoff_matrix = np.asarray(payoff_matrix, dtype=np.float64)
        if group_size == 2:
            game = Matrix2PlayerGameHolder(payoff_matrix.shape[0], payoff_matrix)
        else:
            game = MatrixNPlayerGameHolder(payoff_matrix.shape[0], group_size, payoff_matrix)

    if game.nb_strategies() != 3:
        raise ValueError(
            "plot_pairwise_comparison_rule_dynamics_in_simplex_without_roots currently "
            "supports only 3 strategies."
        )

    simplex = Simplex2D(discrete=True, size=population_size, nb_points=population_size + 1)
    simplex.add_axis(figsize, ax)

    v = np.asarray(xy_to_barycentric_coordinates(simplex.X, simplex.Y, simplex.corners))
    v_int = np.floor(v * population_size).astype(np.int64)

    evolver = PairwiseComparison(population_size=population_size, game=game)
    result = np.zeros(shape=(v_int.shape[1], v_int.shape[2], 3), dtype=np.float64)

    if mu is None:
      gradient_fn = lambda u: evolver.calculate_gradient_of_selection(state=u, beta=beta)
    else:
      gradient_fn = lambda u: evolver.calculate_gradient_of_selection_with_mutation(state=u, beta=beta, mu=mu)

    for i in range(v_int.shape[1]):
        for j in range(v_int.shape[2]):
            if not (v_int[:, i, j] < 0).any() and v_int[:, i, j].sum() <= population_size:
                result[i, j, :] = gradient_fn(v_int[:, i, j])

    result = result.swapaxes(0, 1).swapaxes(0, 2)
    xy_results = vectorized_barycentric_to_xy_coordinates(result, simplex.corners)

    ux = xy_results[:, :, 0].astype(np.float64)
    uy = xy_results[:, :, 1].astype(np.float64)

    simplex.apply_simplex_boundaries_to_gradients(ux, uy)

    return simplex, lambda u, t: population_size * evolver.calculate_gradient_of_selection(beta, u), game, evolver

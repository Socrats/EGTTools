"""
Helper functions for producing plots on simplexes
"""
from typing import Tuple, List, Callable, Optional, Union

import numpy
import numpy as np
from scipy.optimize import root


def simplex_iterator(scale, boundary=True):
    """
    Systematically iterates through a lattice of points on the 2-simplex.
    Parameters
    ----------
    scale: Int
        The normalized scale of the simplex, i.e. N such that points (x,y,z)
        satisify x + y + z == N
    boundary: bool, True
        Include the boundary points (tuples where at least one
        coordinate is zero)
    Yields
    ------
    3-tuples, There are binom(n+2, 2) points (the triangular
    number for scale + 1, less 3*(scale+1) if boundary=False

    Citing
    ----
    This function has been copied from: https://github.com/marcharper/python-ternary/blob/master/ternary/helpers.py
    """

    start = 0
    if not boundary:
        start = 1
    for i in range(start, scale + (1 - start)):
        for j in range(start, scale + (1 - start) - i):
            k = scale - i - j
            yield i, j, k


def xy_to_barycentric_coordinates(x: Union[float, np.ndarray], y: Union[float, np.ndarray],
                                  corners: np.ndarray) -> np.ndarray:
    """

    Parameters
    ----------
    x : Union[float, numpy.ndarray]
        first component of the cartesian coordinates
    y : Union[float, numpy.ndarray]
        second component of the cartesian coordinates
    corners : numpy.ndarray
        a list or a vector containing the coordinates of the corners

    Returns
    -------
    numpy.ndarray
        The transformmation of the coordinates into barycentric.
    """
    corner_x = corners.T[0]
    corner_y = corners.T[1]
    x_1 = corner_x[0]
    x_2 = corner_x[1]
    x_3 = corner_x[2]
    y_1 = corner_y[0]
    y_2 = corner_y[1]
    y_3 = corner_y[2]
    l1 = ((y_2 - y_3) * (x - x_3) + (x_3 - x_2) * (y - y_3)) / (
            (y_2 - y_3) * (x_1 - x_3) + (x_3 - x_2) * (y_1 - y_3))
    l2 = ((y_3 - y_1) * (x - x_3) + (x_1 - x_3) * (y - y_3)) / (
            (y_2 - y_3) * (x_1 - x_3) + (x_3 - x_2) * (y_1 - y_3))
    l3 = 1 - l1 - l2
    return np.asarray([l1, l2, l3])


def barycentric_to_xy_coordinates(point_barycentric: np.ndarray, corners: np.ndarray) -> np.ndarray:
    return (corners.T @ point_barycentric.T).T


def calculate_stability(roots: List[np.ndarray], f: Callable[[np.ndarray], np.ndarray]) -> List[bool]:
    """

    Parameters
    ----------
    roots
    f

    Returns
    -------

    """
    stability = []
    for stationary_point in roots:
        stable = False
        # first we check if the root is in one of the edges
        if np.isclose(stationary_point, 0., atol=1e-7).any():
            sign_plus = 0
            sign_minus = 0
            # Check if we are in an edge
            edge = np.where(~np.isclose(stationary_point, 0., atol=1e-7))[0]
            if edge.shape[0] > 1:  # we are at an edge
                # Now we perturb the edge
                if stationary_point[edge[0]] + 0.1 <= 1.:
                    tmp = stationary_point.copy()
                    tmp[edge[0]] += 0.1
                    tmp[edge[1]] -= 0.1
                    grad = f(tmp)
                    if grad[edge[0]] > 0:
                        sign_plus = 1
                if stationary_point[edge[0]] - 0.1 >= 0.:
                    tmp = stationary_point.copy()
                    tmp[edge[0]] -= 0.1
                    tmp[edge[1]] += 0.1
                    grad = f(tmp)
                    if grad[edge[0]] > 0:
                        sign_minus = 1
                if sign_minus and not sign_plus:
                    stable = True
            else:  # we are at a vertex
                # Now we perturb the vertex
                tmp = stationary_point.copy()
                tmp[edge[0]] -= 0.1
                tmp[(edge[0] + 1) % 3] += 0.1
                grad = f(tmp)
                if grad[edge[0]] > 0.:
                    sign_plus = 1
                tmp[(edge[0] + 1) % 3] -= 0.1
                tmp[(edge[0] + 2) % 3] += 0.1
                grad = f(tmp)
                if grad[edge[0]] > 0.:
                    sign_minus = 1

                if sign_plus and sign_minus:
                    stable = True
        else:  # we are in the interior of the simples
            # here to analyse stability we need to
            # check wether change in any direction leads back to the point
            unstable = False
            for vertex in range(stationary_point.shape[0]):
                tmp = stationary_point.copy()
                if stationary_point[vertex] + 0.1 <= 1.:
                    tmp[vertex] += 0.1
                    if tmp[(vertex + 1) % 3] - 0.1 >= 0.:
                        tmp[(vertex + 1) % 3] -= 0.1
                        grad = f(tmp)
                        if grad[vertex] > 0.:
                            unstable = True
                            break
                        tmp[(vertex + 1) % 3] += 0.1
                    if tmp[(vertex + 2) % 3] - 0.1 >= 0.:
                        tmp[(vertex + 1) % 3] -= 0.1
                        grad = f(tmp)
                        if grad[vertex] > 0.:
                            unstable = True
                            break
                        tmp[vertex + 1 % 3] += 0.1
                    tmp[vertex] -= 0.1
                if stationary_point[vertex] - 0.1 >= 0.:
                    tmp[vertex] -= 0.1
                    if tmp[(vertex + 1) % 3] + 0.1 <= 1.:
                        tmp[(vertex + 1) % 3] += 0.1
                        grad = f(tmp)
                        if grad[vertex] < 0.:
                            unstable = True
                            break
                        tmp[(vertex + 1) % 3] += 0.1
                    if tmp[(vertex + 2) % 3] + 0.1 <= 1.:
                        tmp[(vertex + 1) % 3] += 0.1
                        grad = f(tmp)
                        if grad[vertex] < 0.:
                            unstable = True
                            break
            stable = not unstable
        stability.append(stable)
    return stability


def calculate_stationary_points(x: np.ndarray, y: np.ndarray, corners: np.ndarray,
                                f: Callable[[np.ndarray], np.ndarray],
                                border: Optional[int] = 5,
                                delta: Optional[float] = 1e-12) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """

    Parameters
    ----------
    x
    y
    corners
    f
    border
    delta

    Returns
    -------

    """
    roots = []
    for x, y in zip(x[border:-border], y[border:-border]):
        start = xy_to_barycentric_coordinates(x, y, corners)
        sol = root(f, start, method="hybr")  # ,xtol=1.49012e-10,maxfev=1000
        if sol.success:
            # check if FP is in simplex
            if not np.isclose(np.sum(sol.x), 1., atol=1.e-3):
                continue
            if not np.all((sol.x > -delta) & (sol.x < 1 + delta)):  # only if fp in simplex
                continue

            # only add new fixed points to list
            if not np.array([np.allclose(sol.x, x, atol=1e-7) for x in roots]).any():
                roots.append(sol.x)

    return roots, [barycentric_to_xy_coordinates(x, corners) for x in roots]


def perturb_state(state: Union[Tuple[float, float, float], np.ndarray],
                  perturbation: Optional[float] = 0.01) -> List[np.ndarray]:
    """
    Produces a number of points in the simplex close to the state.

    If the sate is a vertex or in an edge, the perturbation is only made
    across the edges (we don't look for points in the interior of the simplex).

    Parameters
    ----------
    state
    perturbation

    Returns
    -------
    List[numpy.array]
        A list of points (in barycentric coordinates) which are close to the state in the simplex.
    """
    new_states = []
    point_location, indexes = find_where_point_is_in_simplex(state)
    # first we check where the point is

    if point_location == 0:
        # If the point is a vertex, then we only perturb across that axis
        tmp1 = state.copy()
        tmp1[indexes[0]] -= perturbation
        tmp1[(indexes[0] + 1) % 3] += perturbation
        new_states.append(tmp1)
        tmp2 = state.copy()
        tmp2[indexes[0]] -= perturbation
        tmp2[(indexes[0] + 2) % 3] += perturbation
        new_states.append(tmp2)
    elif point_location == 1:
        # If the point in an edge, we will produce 2 points
        if state[indexes[0]] >= perturbation:
            tmp1 = state.copy()
            tmp1[indexes[0]] -= perturbation
            tmp1[indexes[1]] += perturbation
            new_states.append(tmp1)
        if state[indexes[1]] >= perturbation:
            tmp2 = state.copy()
            tmp2[indexes[0]] += perturbation
            tmp2[indexes[1]] -= perturbation
            new_states.append(tmp2)
    else:
        # if the point is in the interior of the simplex, we will
        # produce 3 points, each in the direction of a vertex
        for i in range(len(state)):
            tmp = state.copy()

            if tmp[i] <= 1. - perturbation:
                tmp[i] += perturbation
                if tmp[(i + 1) % 3] >= perturbation:
                    tmp[(i + 1) % 3] -= perturbation
                else:
                    tmp[(i + 2) % 3] -= perturbation
                new_states.append(tmp)

    return new_states


def add_arrow(line, position: Optional[float] = None, direction: Optional[str] = 'right', size: Optional[int] = 15,
              color: Optional[Union[str, Tuple[int, int, int]]] = None, arrowstyle: Optional[str] = '-|>',
              zorder: Optional[int] = 0):
    """
    add an arrow to a line.

    line:       Line2D object
    position:   x-position of the arrow. If None, mean of xdata is taken
    direction:  'left' or 'right'
    size:       size of the arrow in fontsize points
    color:      if None, line color is taken.
    """
    if color is None:
        color = line.get_color()

    xdata = line.get_xdata()
    ydata = line.get_ydata()

    if position is None:
        position = xdata.mean()
    # find closest index
    start_ind = np.argmin(np.absolute(xdata - position))
    if direction == 'right':
        end_ind = start_ind + 1
    else:
        end_ind = start_ind - 1

    line.axes.annotate('',
                       xytext=(xdata[start_ind], ydata[start_ind]),
                       xy=(xdata[end_ind], ydata[end_ind]),
                       arrowprops=dict(arrowstyle=arrowstyle, color=color),
                       size=size, zorder=zorder
                       )


def find_where_point_is_in_simplex(point: np.ndarray) -> Tuple[int, np.ndarray]:
    """
    Finds in which part of the 2D simplex the point is.

    This function will return:
    0 -> if the point is a vertex
    1 -> if the point in an edge
    2 -> if the point is in the interior of the simplex

    Parameters
    ----------
    point : numpy.nparray
        The barycentric coordinates of the point

    Returns
    -------
    Tuple[int, numpy.ndarray]
        An integer which indicates where the point is in the simplex and
        the index of the non zero entries.
    """
    # first we check if the root is in one of the edges
    if np.isclose(point, 0., atol=1e-7).any():
        # Then we check if it might be a vertex
        edge = np.where(~np.isclose(point, 0., atol=1e-7))[0]
        if edge.shape[0] > 1:  # we are at an edge
            return 1, edge
        else:  # we are at a vertex
            return 0, edge
    else:  # we are in the interior of the simplex
        return 2, np.array([0, 1, 2])

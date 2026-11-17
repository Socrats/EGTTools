"""
Helper functions for producing plots on simplexes
"""
from typing import Tuple, List, Callable, Optional, Union, Literal

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import root

from .. import (calculate_nb_states, sample_simplex, )


def simplex_iterator(scale: int, boundary: bool = True) -> Tuple[int, int, int]:
    """
    Systematically iterates through a lattice of points on the 2-simplex.

    Parameters
    ----------
    scale: int
        The normalized scale of the simplex, i.e. N such that points (x,y,z)
        satisify x + y + z == N
    boundary: bool
        Include the boundary points (tuples where at least one
        coordinate is zero)
    Yields
    ------
    Tuple[int, int, int]
        3-tuples, There are binom(n+2, 2) points (the triangular
        number for scale + 1, less 3*(scale+1) if boundary=False

    Citing
    ------
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
    Transforms cartesian into barycentric coordinates.

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

    Examples
    --------
    >>> from egttools.plotting import Simplex2D
    >>> simplex = Simplex2D()
    >>> cartesian_coords = np.array([0.2, 0.])
    >>> xy_to_barycentric_coordinates(cartesian_coords[0], cartesian_coords[1], simplex.corners)
    array([0.2, 0. ])
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
    """
    Transforms barycentric into cartesian coordinates.

    Parameters
    ----------
    point_barycentric: numpy.ndarray
        An array containing the 3 barycentric coordinates.
    corners: numpy.ndarray
        An matrix containing the cartesian coordinates of the corners of the triangle that represents the 2-simplex.

    Returns
    -------
    numpy.ndarray
        An array containing the cartesian coordinates of the input point.
    """
    return (corners.T @ point_barycentric.T).T


def calculate_stability(
        roots: List[NDArray[np.float64]],
        f: Callable[[NDArray[np.float64]], NDArray[np.float64]],
        atol: float = 1e-7,
        return_mode: Literal["bool", "int"] = "bool",
        delta: float = 0.1,
) -> List[bool] | List[int]:
    """
    Calculates the stability of the roots. Returns stability classification for each root.

    From each root :math:`x`, the function steps a distance ``delta`` in every feasible
    direction :math:`d = e_i - e_j` (moving mass from strategy :math:`j` to strategy :math:`i`
    without leaving the simplex) and checks whether the flow points back towards the root
    (:math:`f \\cdot d < 0`) or away from it (:math:`f \\cdot d > 0`). The root is stable if the
    flow points back in every direction, unstable if it points away in every direction, and a
    saddle otherwise. Directions that leave an edge or a vertex towards the interior count,
    so a root that is stable along its edge but invadable by the third strategy is a saddle.

    Parameters
    ----------
    roots : List[NDArray[np.float64]]
        A list of barycentric coordinates of the roots.
    f : Callable[[NDArray[np.float64]], NDArray[np.float64]]
        A function which computes the gradient at any point in the simplex, in barycentric
        coordinates.
    atol : float, default=1e-7
        Numerical tolerance for zero comparisons.
    return_mode : {'bool', 'int'}, default='bool'
        - 'bool' returns True for stable, False otherwise.
        - 'int' returns 1 (stable), -1 (unstable), or 0 (saddle).
    delta : float, default=0.1
        Size of the perturbation, in barycentric coordinates.

    Returns
    -------
    List[bool] or List[int]
        Stability of each root, as booleans or ternary integers.
    """
    stability = []
    nb_strategies = 3

    for stationary_point in roots:
        stationary_point = np.asarray(stationary_point, dtype=np.float64)
        signs = []
        for i in range(nb_strategies):
            for j in range(nb_strategies):
                if i == j or stationary_point[j] - delta < -atol:
                    continue  # moving mass from j to i would leave the simplex
                direction = np.zeros(nb_strategies)
                direction[i], direction[j] = 1., -1.
                grad = np.asarray(f(stationary_point + delta * direction), dtype=np.float64)
                flow = float(grad @ direction)
                signs.append(0. if abs(flow) <= atol else np.sign(flow))

        if signs and all(s < 0 for s in signs):
            result = 1  # Stable: the flow returns from every direction
        elif signs and all(s > 0 for s in signs):
            result = -1  # Unstable: the flow leaves in every direction
        else:
            result = 0  # Saddle (mixed or neutral directions)

        if return_mode == "bool":
            stability.append(result == 1)
        else:
            stability.append(result)

    return stability


def calculate_stationary_points(x: np.ndarray, y: np.ndarray, corners: np.ndarray,
                                f: Callable[[np.ndarray], np.ndarray],
                                border: Optional[int] = 5,
                                delta: Optional[float] = 1e-12,
                                atol: Optional[float] = 1e-7) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Finds the roots of f (the points where the gradient is 0), given a number of points.

    Parameters
    ----------
    x: numpy.ndarray
        x (cartesian) coordinates of the points for which to look for the gradients.
    y: numpy.ndarray
        y (cartesian) coordinates of the points for which to look for the gradients.
    corners: numpy.ndarray
        A matrix containing the cartesian coordinates of the vertices of the triangle that forms the 2-simplex.
    f: Callable[[numpy.ndarray], numpy.ndarray]
        A function that calculates the gradient at any point in the simplex.
    border: int
        Indicates how close to the simplex borders should we look for the gradients. This allows to avoid
        boundary problems.
    delta: float
        tolerance for considering points outside the simplex.
    atol: float
        tolerance for considering that two roots are equal.

    Returns
    -------
    Tuple[List[numpy.ndarray], List[numpy.ndarray]]
        A list with the barycentric coordinates of all the roots that were found and another list with
        the cartesian coordinates.
    """
    roots = []
    for x, y in zip(x[border:-border], y[border:-border]):
        start = xy_to_barycentric_coordinates(x, y, corners)
        sol = root(f, start, method="hybr")  # ,xtol=1.49012e-10,maxfev=1000
        if sol.success:
            v = sol.x
            if check_if_point_in_unit_simplex(v, delta):
                # only add new fixed points to list
                if not np.array([np.allclose(v, x, atol=atol) for x in roots]).any():
                    roots.append(v)

    return roots, [barycentric_to_xy_coordinates(x, corners) for x in roots]


def find_roots_in_discrete_barycentric_coordinates(f: Callable[[np.ndarray], np.ndarray],
                                                   simplex_size: int,
                                                   nb_edge_points: Optional[int] = None,
                                                   nb_interior_points: Optional[int] = 1000,
                                                   delta: Optional[float] = 1e-12,
                                                   atol: Optional[float] = 1e-3) -> List[np.ndarray]:
    """
    Searches for the roots inside the simplex and returns them in barycentric coordinates.

    Parameters
    ----------
    f: Callable[[numpy.ndarray], numpy.ndarray]
        A function that calculates the gradient of any point inside the simplex.
    simplex_size : int
        Discrete size of the edges of the simplex. This should correspond to the size of the finite population
        in Moran dynamics.
    nb_edge_points: int
        Can be used to explore more points than the existing simplex size.
    nb_interior_points: int
        Number of points to explore inside the simplex.
    delta: float
        Tolerance to consider a point outside the unit simplex.
    atol: float
        Tolerance to consider two roots to be equal.

    Returns
    -------
    List[numpy.ndarray]
        A list with the barycentric coordinates of the roots.

    See Also
    --------
    egttools.plotting.helpers.calculate_stationary_points
    """
    roots = []

    if nb_edge_points is None:
        nb_edge_points = simplex_size

    # We first test values along the edges
    values = np.linspace(0, simplex_size, nb_edge_points, )
    point = np.zeros(shape=(3,), dtype=np.int64)
    for value in values:
        for i in range(3):
            point[i] = value
            point[(i + 1) % 3] = simplex_size - value
            point[(i + 2) % 3] = 0

            sol = root(f, point, method="hybr")  # ,xtol=1.49012e-10,maxfev=1000
            if sol.success:
                v = sol.x / simplex_size
                if check_if_point_in_unit_simplex(v, delta):
                    # only add new fixed points to list
                    if not np.array([np.allclose(v, x, atol=atol) for x in roots]).any():
                        roots.append(v)

    # finally we can explore values inside the simplex
    nb_states = calculate_nb_states(simplex_size, 3)

    if nb_interior_points > nb_states:
        nb_interior_points = nb_states
    initial_points = np.random.choice(range(nb_states), size=nb_interior_points, replace=False)

    for i in initial_points:
        point = sample_simplex(i, simplex_size, 3)
        if np.any(np.isclose(point, 0., atol=1e-7)):
            continue
        sol = root(f, point, method="hybr")  # ,xtol=1.49012e-10,maxfev=1000
        if sol.success:
            v = sol.x / simplex_size
            if check_if_point_in_unit_simplex(v, delta):
                # only add new fixed points to list
                if not np.array([np.allclose(v, x, atol=atol) for x in roots]).any():
                    roots.append(v)

    return roots


def check_if_point_in_unit_simplex(point: np.ndarray, delta: Optional[float] = 1e-12) -> bool:
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


def perturb_state(state: Union[Tuple[float, float, float], np.ndarray],
                  perturbation: Optional[float] = 0.01) -> List[np.ndarray]:
    """
    Produces a number of points in the simplex close to the state.

    If the sate is a vertex or in an edge, the perturbation is only made
    across the edges (we don't look for points in the interior of the simplex).

    Parameters
    ----------
    state: Union[Tuple[float, float, float], numpy.ndarray]
        Barycentric coordinates of a point inside the simplex.
    perturbation: float
        The amount of perturbation to apply to the point.

    Returns
    -------
    List[numpy.ndarray]
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


def perturb_state_discrete(state: Union[Tuple[float, float, float], np.ndarray],
                           size: int,
                           perturbation: Optional[int] = 1) -> List[np.ndarray]:
    """
    Produces a number of points in the simplex close to the state.

    If the sate is a vertex or in an edge, the perturbation is only made
    across the edges (we don't look for points in the interior of the simplex).

    Parameters
    ----------
    state: Union[Tuple[float, float, float], numpy.ndarray]
        The barycentric coordinates of a point inside the simplex.
    size: int
        The size of the edges of the simplex. This should coincide with the size of the finite population
        in Moran dynamics.
    perturbation: int
        The amount of perturbation to apply to the point.

    Returns
    -------
    List[numpy.ndarray]
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
            if tmp1.sum() < size:
                tmp1[indexes[0]] += size - tmp1.sum()
            tmp1[indexes[0]] -= perturbation
            tmp1[indexes[1]] += perturbation
            new_states.append(tmp1)
        if state[indexes[1]] >= perturbation:
            tmp2 = state.copy()
            if tmp2.sum() < size:
                tmp2[indexes[0]] += size - tmp2.sum()
            tmp2[indexes[0]] += perturbation
            tmp2[indexes[1]] -= perturbation
            new_states.append(tmp2)
    else:
        # if the point is in the interior of the simplex, we will
        # produce 3 points, each in the direction of a vertex
        for i in range(len(state)):
            tmp = state.copy()
            if tmp.sum() < size:
                tmp[0] += size - tmp.sum()

            if tmp[i] + perturbation <= size:
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
    point : numpy.ndarray
        The barycentric coordinates of the point

    Returns
    -------
    Tuple[int, numpy.ndarray]
        An integer which indicates where the point is in the simplex and
        the index of the non-zero entries.
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


def lattice_gradient_interpolator(
        gradient_fn: Callable[[NDArray[np.int64]], NDArray[np.float64]],
        population_size: int,
) -> Tuple[Callable[[NDArray[np.float64]], NDArray[np.float64]], NDArray[np.float64]]:
    """
    Continuous extension of a gradient defined on the states of a finite population of 3 strategies.

    The gradient of selection of a finite population is defined only at the integer states
    :math:`(n_0, n_1, n_2)` with :math:`n_0 + n_1 + n_2 = Z`. This function evaluates
    ``gradient_fn`` once at every such state and interpolates linearly inside each triangle of
    the lattice, which gives a continuous, piecewise-linear field that equals the exact gradient
    at every population state. Root finding, stability analysis and trajectory integration need
    such a continuous field.

    Parameters
    ----------
    gradient_fn : Callable[[numpy.ndarray], numpy.ndarray]
        Function returning the gradient (length 3) at an integer population state.
    population_size : int
        Population size :math:`Z`.

    Returns
    -------
    Tuple[Callable[[numpy.ndarray], numpy.ndarray], numpy.ndarray]
        The interpolating function, which takes a state in counts (3 floats; it is projected
        onto the plane :math:`\\sum_i n_i = Z` and clipped to the simplex), and the array of
        lattice gradients of shape ``(Z + 1, Z + 1, 3)``, where entry ``[i, j]`` holds the
        gradient at state ``(i, j, Z - i - j)`` and entries with ``i + j > Z`` are NaN.
    """
    z = int(population_size)
    if z < 1:
        raise ValueError("population_size must be >= 1")
    lattice = np.full((z + 1, z + 1, 3), np.nan)
    for i in range(z + 1):
        for j in range(z + 1 - i):
            lattice[i, j] = gradient_fn(np.array([i, j, z - i - j], dtype=np.int64))

    def interpolate(state: NDArray[np.float64]) -> NDArray[np.float64]:
        u = np.asarray(state, dtype=np.float64)
        u = u + (z - u.sum()) / 3.  # project onto the plane n_0 + n_1 + n_2 = Z
        a, b = min(max(u[0], 0.), float(z)), min(max(u[1], 0.), float(z))
        if a + b > z:
            a, b = a * z / (a + b), b * z / (a + b)
        i = min(int(np.floor(a)), z - 1)
        j = min(int(np.floor(b)), z - 1 - i)
        fa, fb = a - i, b - j
        if fa + fb <= 1. or i + j + 2 > z:
            # lower triangle (i, j), (i + 1, j), (i, j + 1)
            g0 = lattice[i, j]
            return g0 + fa * (lattice[i + 1, j] - g0) + fb * (lattice[i, j + 1] - g0)
        # upper triangle (i + 1, j + 1), (i, j + 1), (i + 1, j)
        g0 = lattice[i + 1, j + 1]
        return g0 + (1. - fa) * (lattice[i, j + 1] - g0) + (1. - fb) * (lattice[i + 1, j] - g0)

    return interpolate, lattice


def find_roots_of_lattice_gradient(
        lattice: NDArray[np.float64],
        atol: float = 1e-10,
) -> List[NDArray[np.float64]]:
    """
    Zeros of the piecewise-linear gradient built by :func:`lattice_gradient_interpolator`.

    Inside each lattice triangle the interpolated gradient is affine, so its zero is the solution
    of a 2x2 linear system and is found exactly; no initial guesses are needed. Population
    states where the gradient itself is zero (e.g. monomorphic states without mutation) are
    included. Triangles where the gradient vanishes identically (neutral drift) contribute only
    their vertices.

    Parameters
    ----------
    lattice : numpy.ndarray
        Lattice gradients of shape ``(Z + 1, Z + 1, 3)`` as returned by
        :func:`lattice_gradient_interpolator`.
    atol : float
        Tolerance below which a gradient (or a determinant) counts as zero.

    Returns
    -------
    List[numpy.ndarray]
        Roots in barycentric coordinates (fractions summing to 1), without duplicates.
    """
    z = lattice.shape[0] - 1
    points = []
    for i in range(z + 1):
        for j in range(z + 1 - i):
            if np.max(np.abs(lattice[i, j])) <= atol:
                points.append((float(i), float(j)))

    def solve(origin, g0, e1, g1, e2, g2):
        # g0 + s (g1 - g0) + t (g2 - g0) = 0, using two components (the third follows: sum is 0)
        m = np.array([[g1[0] - g0[0], g2[0] - g0[0]], [g1[1] - g0[1], g2[1] - g0[1]]])
        det = np.linalg.det(m)
        if abs(det) <= atol:
            return
        s, t = np.linalg.solve(m, -g0[:2])
        if s >= -1e-12 and t >= -1e-12 and s + t <= 1. + 1e-12:
            points.append((origin[0] + s * e1[0] + t * e2[0], origin[1] + s * e1[1] + t * e2[1]))

    for i in range(z):
        for j in range(z - i):
            solve((i, j), lattice[i, j], (1, 0), lattice[i + 1, j], (0, 1), lattice[i, j + 1])
            if i + j + 2 <= z:
                solve((i + 1, j + 1), lattice[i + 1, j + 1], (-1, 0), lattice[i, j + 1], (0, -1),
                      lattice[i + 1, j])

    roots: List[NDArray[np.float64]] = []
    for a, b in points:
        root_point = np.array([a, b, z - a - b]) / z
        if not any(np.allclose(root_point, other, atol=1e-6) for other in roots):
            roots.append(root_point)
    return roots

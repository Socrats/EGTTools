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

import matplotlib.colors
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import numpy as np

from scipy.integrate import odeint
from matplotlib.patches import Circle
from typing import Optional, Tuple, List, Union, Callable, TypeVar
from numpy.typing import ArrayLike
from .. import (sample_simplex, sample_unit_simplex, calculate_nb_states, )
from .helpers import (barycentric_to_xy_coordinates, perturb_state, add_arrow,
                      perturb_state_discrete, find_where_point_is_in_simplex,
                      xy_to_barycentric_coordinates)

SelfSimplex2D = TypeVar("SelfSimplex2D", bound="Simplex2D")


class Simplex2D:
    top_corner = np.sqrt(3) / 2
    side_slope = np.sqrt(3)
    corners = np.array([
        [0, 0],
        [1 / 2, top_corner],
        [1, 0]
    ])
    triangle = tri.Triangulation(corners[:, 0], corners[:, 1])
    refiner = tri.UniformTriRefiner(triangle)
    trimesh = refiner.refine_triangulation(subdiv=5)

    def __init__(self, nb_points: Optional[int] = 1000, discrete: Optional[bool] = False,
                 size: Optional[Union[int, None]] = None):
        """
        Plots a 2-dimensional simplex in a cartesian plane.

        This class offers utility methods to plot gradients and equilibrium points on a 2-simplex (triangle).

        The plotting is always done on the unit simplex for convenience. At the moment no rotations are
        implemented, but we plan to add this feature, so that the triangle can be rotated before the plot.

        We discern between continuous and discrete dynamics. The main reason is that this class' objective
        is to plot evolutionary dynamics on a simplex. When we are working with the replicator equation
        it is straightforward to calculate all the gradients on the unit simplex. However, when working
        with finite populations using the social learning model (social imitation), we are actually working
        with a simplex with size equivalent to the population size (so all the dimensions of the simplex must
        sum to `Z`) and we only consider discrete (integer) values inside the simplex (the population may
        only have integer individuals). Of course this can be translated into frequencies, which gets us
        back to the unit simplex, but it is not so simple to transform any value between 0-1 sampled with
        numpy.linspace to a discrete value.

        Therefore, for the discrete case, will will sample directly discrete points in barycentric
        coordinates and only then, translate them into cartesian cooordinates.

        Parameters
        ----------
        nb_points : int
            number of points for which to calculate the gradients
        discrete : bool
            indicates whether we are in the continuous or discrete case
        size : int
            if we are in the discrete case, indicates the size of the simplex

        See Also
        --------
        egttools.plotting.plot_gradient,
        egttools.plotting.draw_invasion_diagram,
        egttools.analytical.replicator_equation,
        egttools.analytical.StochDynamics

        Cite
        -----
        This class has been inspired from: https://github.com/marvinboe/egtsimplex

        Examples
        --------
        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> from egttools.plotting.helpers import (xy_to_barycentric_coordinates, calculate_stationary_points,
            ... calculate_stability)
        >>> from egttools.helpers.vectorized import (vectorized_replicator_equation,
            ... vectorized_barycentric_to_xy_coordinates)
        >>> from egttools.analytical import replicator_equation
        >>> simplex = Simplex2D()
        >>> payoffs = np.array([[1, 0, 0],
            ...        [0, 2, 0],
            ...        [0, 0, 3]])
        >>> v = np.asarray(xy_to_barycentric_coordinates(simplex.X, simplex.Y, simplex.corners))
        >>> results = vectorized_replicator_equation(v, payoffs)
        >>> xy_results = vectorized_barycentric_to_xy_coordinates(results, simplex.corners)
        >>> Ux = xy_results[:, :, 0].astype(np.float64)
        >>> Uy = xy_results[:, :, 1].astype(np.float64)
        >>> calculate_gradients = lambda u: replicator_equation(u, payoffs)
        >>> roots, roots_xy = calculate_stationary_points(simplex.trimesh.x, simplex.trimesh.y,
            ... simplex.corners, calculate_gradients)
        >>> stability = calculate_stability(roots, calculate_gradients)
        >>> type_labels = ['A', 'B', 'C']
        >>> fig, ax = plt.subplots(figsize=(10,8))
        >>> plot = (simplex.add_axis(ax=ax)
            ...            .apply_simplex_boundaries_to_gradients(Ux, Uy)
            ...            .draw_triangle()
            ...            .draw_gradients(zorder=0)
            ...            .add_colorbar()
            ...            .draw_stationary_points(roots_xy, stability)
            ...            .add_vertex_labels(type_labels)
            ...            .draw_trajectory_from_roots(lambda u, t: replicator_equation(u, payoffs),
            ...                                        roots,
            ...                                        stability,
            ...                                        trajectory_length=15,
            ...                                        linewidth=1,
            ...                                        step=0.01,
            ...                                        color='k', draw_arrow=True, arrowdirection='right',
            ...                                        arrowsize=30, zorder=4, arrowstyle='fancy')
            ...            .draw_scatter_shadow(lambda u, t: replicator_equation(u, payoffs), 300, color='gray',
            ...                                 marker='.', s=0.1, zorder=0)

        .. image:: ../images/simplex_example_infinite_pop_1.png

        >>> plot = (simplex.add_axis(ax=ax)
            ...            .apply_simplex_boundaries_to_gradients(Ux, Uy)
            ...            .draw_triangle()
            ...            .draw_stationary_points(roots_xy, stability)
            ...            .add_vertex_labels(type_labels)
            ...            .draw_trajectory_from_roots(lambda u, t: replicator_equation(u, payoffs),
            ...                                        roots,
            ...                                        stability,
            ...                                        trajectory_length=15,
            ...                                        linewidth=1,
            ...                                        step=0.01,
            ...                                        color='k', draw_arrow=True, arrowdirection='right',
            ...                                        arrowsize=30, zorder=4, arrowstyle='fancy')
            ...            .draw_scatter_shadow(lambda u, t: replicator_equation(u, payoffs), 300, color='gray',
            ...                                 marker='.', s=0.1, zorder=0)

        .. image:: ../images/simplex_example_infinite_pop_2.png
        """
        self.nb_points = nb_points
        self.Ux = None
        self.Uy = None
        self.stream = None
        x = np.linspace(0, 1, nb_points)
        y = np.linspace(0, 1, nb_points)
        self.X, self.Y = np.meshgrid(x, y)
        self.figure = None
        self.ax = None
        self.discrete = discrete
        self.size = size
        self.random_drift_edges = []

        if discrete:
            self.nb_states = calculate_nb_states(self.size, 3)
            xy_coords = []

            for i in range(self.nb_states):
                state = sample_simplex(i, size, 3)
                xy_coords.append(barycentric_to_xy_coordinates(state / size, self.corners))
            xy_coords = np.asarray(xy_coords)

            self.triangle_discrete = tri.Triangulation(xy_coords[:, 0].flatten(), xy_coords[:, 1].flatten())

    def add_axis(self, figsize: Optional[Tuple[int, int]] = (10, 8), ax: Optional[plt.axis] = None) -> SelfSimplex2D:
        """
        Creates or stores a new axis inside the class.

        Parameters
        ----------
        figsize: Optional[Tuple[int, int]]
            The size of the figure. This argument is only used if no ax is given.
        ax: Optional[matplotlib.pyplot.axis]
            If given, the axis will be stored inside the object. Otherwise, a new axis will be created.

        Returns
        -------
        Simplex2D
            The class object.
        """
        if ax is not None:
            self.ax = ax
            self.figure = ax.figure
        else:
            self.figure, self.ax = plt.subplots(figsize=figsize)

        return self

    def add_edges_with_random_drift(self, random_drift_edges: List[Tuple[int, int]]) -> SelfSimplex2D:
        """
        Adds information to the class about which edges have random drift.

        This will be used to avoid plotting a lot equilibria alongside an edge.

        Parameters
        ----------
        random_drift_edges: List[Tuple[int, int]]
            A list of tuples which indicate the (undirected) edges in which there is random drift.

        Returns
        -------
        Simplex2D
            The class object.
        """
        self.random_drift_edges = random_drift_edges
        return self

    def get_figure_and_axis(self) -> Tuple[plt.figure, plt.axis]:
        """
        Returns the stored figure and axis.

        Returns
        -------
        Tuple[matplotlib.pyplot.figure, matplotlib.pyplot.axis]
            The figure and axis stored in the current object.
        """
        return self.figure, self.ax

    def apply_simplex_boundaries_to_gradients(self, u: np.ndarray, v: np.ndarray) -> SelfSimplex2D:
        """
        Applies boundaries of the triangle to a list of gradient values over the cartesian grid.

        The boundaries are applied using the X Y grid defined in the instantiation of the class.

        Parameters
        ----------
        u: numpy.ndarray
            The X component of the gradients.
        v: numpy.ndarray
            The Y component of the gradients
        Returns
        -------
        Simplex2D
            A reference to the class object.
        """
        self.Ux = u.copy()
        self.Uy = v.copy()

        self.Ux = np.where(self.X >= self.Y / self.side_slope, self.Ux, np.nan)
        self.Ux = np.where(self.Y <= -self.side_slope * self.X + self.side_slope, self.Ux, np.nan)
        self.Ux = np.where(self.Y <= self.top_corner, self.Ux, np.nan)
        self.Ux = np.where(self.Y > 0, self.Ux, np.nan)
        self.Uy = np.where(self.X >= self.Y / self.side_slope, self.Uy, np.nan)
        self.Uy = np.where(self.Y <= -self.side_slope * self.X + self.side_slope, self.Uy, np.nan)
        self.Uy = np.where(self.Y <= self.top_corner, self.Uy, np.nan)
        self.Uy = np.where(self.Y > 0, self.Uy, np.nan)

        return self

    def draw_triangle(self, color: Optional[str] = 'k', linewidth: Optional[int] = 2,
                      linewidth_random_drift: Optional[int] = 4) -> SelfSimplex2D:
        """
        Draws the borders of a triangle enclosing the 2-simplex.

        Parameters
        ----------
        color: Optional[str]
            The color of the borders of the triangle.
        linewidth: Optional[int]
            The width of the borders of the triangle.
        linewidth_random_drift: Optional[int]
            The width of the dashed line that represents the edges with random drift.

        Returns
        -------
        Simplex2D
            A refernece to the class object.
        """
        self.ax.triplot(self.triangle, color=color, linewidth=linewidth)
        for edge in self.random_drift_edges:
            self.ax.plot([self.corners[edge[0], 0], self.corners[edge[1], 0]],
                         [self.corners[edge[0], 1], self.corners[edge[1], 1]], lw=linewidth_random_drift,
                         linestyle='dashed', color=color)
        return self

    def draw_gradients(self, arrowsize: Optional[float] = 2,
                       arrowstyle: Optional[str] = 'fancy',
                       color: Optional[Union[str, Tuple[int, int, int]]] = None, density: Optional[float] = 1,
                       linewidth: Optional[float] = 1.5,
                       cmap: Optional[Union[str, matplotlib.colors.Colormap]] = 'viridis',
                       zorder: Optional[int] = 0) -> SelfSimplex2D:
        """
        Draws the gradients inside the unit simplex using a streamplot.

        Parameters
        ----------
        arrowsize: Optional[float]
            The size of the arrows of the gradients
        arrowstyle: Optional[str]
            The style of the arrows. See matplotlib arrowstyles.
        color: Optional[Union[str, Tuple[int, int, int]]]
            The color of the arrows. If no color is given, it will be generated as a function of the gradients.
        density: Optional[float]
            The density of arrows (how many arrows) to plot.
        linewidth: Optional[float]
            The width of the arrows.
        cmap: Optional[Union[str, matplotlib.colors.Colormap]]
            The color map to be used.
        zorder: Optional[int]
            The order in which the gradients should appear in the plot (above or below other elements).

        Returns
        -------
        Simplex2D
            A reference to the class object.

        """
        if self.Ux is None or self.Uy is None:
            raise Exception("Please call Simplex.apply_simplex_boundaries_to_gradients first")

        velocities = (self.Ux ** 2 + self.Uy ** 2) ** 0.5
        colors = velocities if color is None else color

        self.stream = self.ax.streamplot(self.X, self.Y,
                                         self.Ux, self.Uy,
                                         arrowsize=arrowsize,
                                         arrowstyle=arrowstyle,
                                         color=colors,
                                         density=density,
                                         linewidth=linewidth,
                                         cmap=cmap,
                                         zorder=zorder
                                         )
        return self

    def add_colorbar(self, aspect: Optional[float] = 10,
                     anchor: Optional[Tuple[float, float]] = (-0.5, 0.5),
                     panchor: Optional[Tuple[float, float]] = (0, 0),
                     shrink: Optional[float] = 0.6,
                     label: Optional[str] = 'gradient of selection',
                     label_rotation: Optional[int] = 270,
                     label_fontsize: Optional[int] = 16,
                     labelpad: Optional[float] = 20) -> SelfSimplex2D:
        """
        Adds a color bar to indicate the meaning of the colors of the plotted gradients.
        This should only be used if the gradients were plotted and the colors have been drawn in function
        of the strength of the gradient.

        Parameters
        ----------
        aspect: Optional[float]
            Aspect ration of the color bar.
        anchor: Optional[Tuple[float, float]]
            Anchor point for the color bar.
        panchor: Optional[Tuple[float, float]]
        shrink: Optional[float]
            Ration for shrinking the color bar.
        label: Optional[str]
            Label for the color bar.
        label_rotation: Optional[int]
            Rotation of the label.
        label_fontsize: Optional[int]
            Font size of the label.
        labelpad: Optional[float]
            How much padding should be added to the label.

        Returns
        -------
        Simplex2D
            A reference to the class object.

        """
        cbar = self.figure.colorbar(self.stream.lines, aspect=aspect, anchor=anchor, panchor=panchor, shrink=shrink,
                                    ax=self.ax)
        cbar.set_label(label, rotation=label_rotation, fontsize=label_fontsize, labelpad=labelpad)

        return self

    def draw_stationary_points(self, roots: List[Union[Tuple[float, float], np.ndarray]],
                               stability: Union[List[bool], List[int]],
                               zorder: Optional[int] = 5, linewidth: Optional[float] = 3,
                               atol: Optional[float] = 1e-7) -> SelfSimplex2D:
        """
        Draws the black circles for stable points and white circles for unstable ones.

        Parameters
        ----------
        roots:
            A list of arrays (or tuples) containing the cartesian coordinates of the roots.
        stability:
            A list of boolean or integer values indicating whether the root is stable. If there are integer values
            -1 - unstable, 0 - saddle, 1 - stable.
        zorder:
            Indicates in which order these points should appear in the figure (above or below other plots).
        linewidth:
            Width of the border of the circles that represents the roots.
        atol:
            Tolerance to consider a value equal to 0. Used to check if a point is on an edge.

        Returns
        -------
        Simplex2D
            A reference to the class object.

        """
        for i, stationary_point in enumerate(roots):
            point = xy_to_barycentric_coordinates(stationary_point[0], stationary_point[1], self.corners)
            place, _ = find_where_point_is_in_simplex(point)
            if place == 1:
                # First let's check if stationary point is in an edge with random drift
                if np.isclose([point[3 - np.sum(edge)] for edge in self.random_drift_edges], 0.,
                              atol=atol).any():
                    continue

            if stability[i] == 1:
                facecolor = 'k'
            elif stability[i] == -1:
                facecolor = 'white'
            else:
                facecolor = 'grey'
            self.ax.add_artist(Circle(stationary_point, 0.015,
                                      edgecolor='k', facecolor=facecolor, zorder=zorder, linewidth=linewidth))
        return self

    def add_vertex_labels(self, labels: Union[Tuple[str, str, str], List[str]], epsilon_bottom: Optional[float] = 0.05,
                          epsilon_top: Optional[float] = 0.05,
                          fontsize: Optional[float] = 16,
                          horizontalalignment: Optional[str] = 'center') -> SelfSimplex2D:
        """
        Adds labels to the vertices of the triangle that represents the 2-simplex.

        Parameters
        ----------
        labels: Union[Tuple[str, str, str], List[str]]
            A tuple or a list containing 3 strings that give name to the vertices of the triangle. The order is
            bottom left corner, top corner, bottom right corner.
        epsilon_bottom: Optional[float]
            How much separation should the label have from the bottom vertices
        epsilon_top: Optional[float]
            How much separation should the label have from the top vertex.
        fontsize: Optional[float]
            Font size for the labels.
        horizontalalignment: Optional[str]
            Horizontal alignment for the label text.

        Returns
        -------
        Simplex2D
            A reference to the current object.

        """
        self.ax.annotate(labels[0], self.corners[0], xytext=self.corners[0] + np.array([-epsilon_bottom, 0.0]),
                         horizontalalignment=horizontalalignment, va='center', fontsize=fontsize)
        self.ax.annotate(labels[1], self.corners[1], xytext=self.corners[1] + np.array([0.0, epsilon_top + 0.02]),
                         horizontalalignment=horizontalalignment, va='top', fontsize=fontsize)
        self.ax.annotate(labels[2], self.corners[2], xytext=self.corners[2] + np.array([epsilon_bottom, 0.0]),
                         horizontalalignment=horizontalalignment, va='center', fontsize=fontsize)

        return self

    def draw_trajectories(self, f: Callable[[np.ndarray, int], np.ndarray], nb_trajectories: int,
                          trajectory_length: Optional[int] = 15, step: Optional[float] = 0.01,
                          color: Optional[Union[str, Tuple[int, int, int]]] = 'whitesmoke',
                          ms: Optional[float] = 0.5, zorder: Optional[int] = 0) -> SelfSimplex2D:
        """
        Draws trajectories inside the unit simplex starting from random initial points.

        Parameters
        ----------
        f: Callable[[np.ndarray, int], np.ndarray]
            Function that can calculate the gradient at any point in the simplex.
        nb_trajectories: int
            Number of trajectories to draw.
        trajectory_length: Optional[int]
            Length of the trajectory. This is used to calculate the amount of points odeint should calculate.
        step: Optional[float]
            The step size in time to get to the maximum trajectory length. Together with trajectory_length
            this indicates the amount of points odeint should calculate.
        color: Optional[Union[str, Tuple[int, int, int]]]
            The color of the points of the trajectory.
        ms: Optional[float]
            The size of the points.
        zorder: Optional[int]
            The order in which this plot should appear in the figure (above or bellow other plots).

        Returns
        -------
        Simplex2D
            A reference to the current object.

        """
        if self.discrete:
            if nb_trajectories > self.nb_states:
                nb_trajectories = self.nb_states
            initial_points = np.random.choice(range(self.nb_states), size=nb_trajectories, replace=False)
            for point in initial_points:
                u = sample_simplex(point, self.size, 3)
                x = odeint(f, u, np.arange(0, trajectory_length, step), full_output=False)
                # noinspection PyTypeChecker
                v = barycentric_to_xy_coordinates(x / self.size, self.corners)
                self.ax.plot(v[:, 0], v[:, 1], color, ms=ms, zorder=zorder)
        else:
            for _ in range(nb_trajectories):
                u = sample_unit_simplex(3)
                x = odeint(f, u, np.arange(0, trajectory_length, step), full_output=False)
                # noinspection PyTypeChecker
                v = barycentric_to_xy_coordinates(x, self.corners)
                self.ax.plot(v[:, 0], v[:, 1], color, ms=ms, zorder=zorder)

        return self

    def draw_trajectory_from_points(self, f: Callable[[np.ndarray, int], np.ndarray], points: List[np.ndarray],
                                    trajectory_length: Optional[int] = 15, step: Optional[float] = 0.1,
                                    color: Optional[Union[str, Tuple[int, int, int]]] = 'k',
                                    linewidth: Optional[float] = 0.5, zorder: Optional[int] = 0,
                                    draw_arrow: Optional[bool] = False, arrowstyle: Optional[str] = 'fancy',
                                    arrowsize: Optional[int] = 50,
                                    position: Optional[int] = None,
                                    arrowdirection: Optional[str] = 'right') -> SelfSimplex2D:
        """
        Draws trajectories inside the unit simplex starting from the indicated points.

        Parameters
        ----------
        f: Callable[[np.ndarray, int], np.ndarray]
            Function that can calculate the gradient at any point in the simplex.
        points: List[np.ndarray[np.float64[3,m]]
            A list of points in barycentric coordinates from which the trajectories should start.
        trajectory_length: Optional[int]
            Length of the trajectory. This is used to calculate the amount of points odeint should calculate.
        step: Optional[float]
            The step size in time to get to the maximum trajectory length. Together with trajectory_length
            this indicates the amount of points odeint should calculate.
        color: Optional[Union[str, Tuple[int, int, int]]]
            The color of the points of the trajectory.
        linewidth: Optional[float] = 0.5
            Width of the line to be plot.
        zorder: Optional[int]
            The order in which this plot should appear in the figure (above or bellow other plots).
        draw_arrow: Optional[bool]
            Indicates whether to draw an arrow along the trajectory.
        arrowstyle: Optional[str]
            Indicates the style of the arrow to be plotted.
        arrowsize: Optional[int]
            The size of the arrow.
        position: Optional[int]
            Where should the arrow be pltoted.
        arrowdirection: Optional[str]
            Indicates whether the arrow should be plotted in the direction of the advancing trajectory (right) or
            the opposite.

        Returns
        -------
        Simplex2D
            A reference to the current object.
        """

        for i, point in enumerate(points):
            x = odeint(f, point, np.arange(0, trajectory_length, step), full_output=False)

            # noinspection PyTypeChecker
            v = barycentric_to_xy_coordinates(x, self.corners)
            line = self.ax.plot(v[:, 0], v[:, 1], color, linewidth=linewidth, zorder=zorder)[0]
            if draw_arrow:
                add_arrow(line, size=arrowsize, arrowstyle=arrowstyle, position=position,
                          direction=arrowdirection)

        return self

    def draw_trajectory_from_roots(self, f: Callable[[np.ndarray, int], np.ndarray], roots: List[np.ndarray],
                                   stability: List[bool],
                                   trajectory_length: Optional[int] = 15, step: Optional[float] = 0.1,
                                   perturbation: Optional[Union[int, float]] = 0.01,
                                   color: Optional[Union[str, Tuple[int, int, int]]] = 'k',
                                   linewidth: Optional[float] = 0.5, zorder: Optional[int] = 0,
                                   draw_arrow: Optional[bool] = False, arrowstyle: Optional[str] = 'fancy',
                                   arrowsize: Optional[int] = 50,
                                   position: Optional[int] = None,
                                   arrowdirection: Optional[str] = 'right',
                                   atol: Optional[float] = 1e-7) -> SelfSimplex2D:
        """
        Draws trajectories inside the unit simplex starting from the stationary points.

        Parameters
        ----------
        f: Callable[[np.ndarray, int], np.ndarray]
            Function that can calculate the gradient at any point in the simplex.
        roots: List[np.ndarray[np.float64[3,m]]
            A list of points in barycentric coordinates from which the trajectories should start.
        stability: List[bool]
            Indicates whether the root is a stable or unstable point.
        trajectory_length: Optional[int]
            Length of the trajectory. This is used to calculate the amount of points odeint should calculate.
        step: Optional[float]
            The step size in time to get to the maximum trajectory length. Together with trajectory_length
            this indicates the amount of points odeint should calculate.
        perturbation: Optional[Union[int, float]]
            Indicates how much perturbation should be applied to the root to start drawing the trajectory.
            If no perturbation is applied, since the gradient is 0, the system will never leave the root.
        color: Optional[Union[str, Tuple[int, int, int]]]
            The color of the points of the trajectory.
        linewidth: Optional[float] = 0.5
            Width of the line to be plot.
        zorder: Optional[int]
            The order in which this plot should appear in the figure (above or bellow other plots).
        draw_arrow: Optional[bool]
            Indicates whether to draw an arrow along the trajectory.
        arrowstyle: Optional[str]
            Indicates the style of the arrow to be plotted.
        arrowsize: Optional[int]
            The size of the arrow.
        position: Optional[int]
            Where should the arrow be pltoted.
        arrowdirection: Optional[str]
            Indicates whether the arrow should be plotted in the direction of the advancing trajectory (right) or
            the opposite.
        atol: Optional[float]
            Tolerance to consider a value equal to 0. Used to check if a point is on an edge of the simplex.

        Returns
        -------
        Simplex2D
            A reference to the current object.
        """
        if self.discrete:
            if type(perturbation) is float:
                perturbation = 1

            for i, stationary_point in enumerate(roots):
                # First let's check if stationary point is in an edge with random drift
                if np.isclose([stationary_point[3 - np.sum(edge)] for edge in self.random_drift_edges], 0.,
                              atol=atol).any():
                    continue
                if stability[i]:  # we don't plot arrows starting at stable points
                    continue
                stationary_point_discrete = (stationary_point * self.size)
                states = perturb_state_discrete(stationary_point_discrete, self.size, perturbation=perturbation)
                for state in states:
                    x = odeint(f, state, np.arange(0, trajectory_length, step), full_output=False)
                    # check if point is outside simplex
                    # noinspection PyUnresolvedReferences
                    tmp_check = np.where(~np.isclose(x.sum(axis=1) / self.size, 1., atol=1e-2))[0]
                    if len(tmp_check) > 0:
                        last_inside = np.min(tmp_check)
                        if last_inside == 0:
                            continue
                        x = x[:last_inside]
                    # noinspection PyTypeChecker
                    v = barycentric_to_xy_coordinates(x / self.size, self.corners)
                    line = self.ax.plot(v[:, 0], v[:, 1], color, linewidth=linewidth, zorder=zorder)[0]
                    if draw_arrow:
                        add_arrow(line, size=arrowsize, arrowstyle=arrowstyle, position=position,
                                  direction=arrowdirection)
        else:
            for i, stationary_point in enumerate(roots):
                # First let's check if stationary point is in an edge with random drift
                if np.isclose([stationary_point[3 - np.sum(edge)] for edge in self.random_drift_edges], 0.,
                              atol=atol).any():
                    continue
                if stability[i] == 1:  # we don't plot arrows starting at stable points
                    continue
                states = perturb_state(stationary_point, perturbation=perturbation)
                for state in states:
                    x = odeint(f, state, np.arange(0, trajectory_length, step), full_output=False)

                    # noinspection PyTypeChecker
                    v = barycentric_to_xy_coordinates(x, self.corners)
                    line = self.ax.plot(v[:, 0], v[:, 1], color, linewidth=linewidth, zorder=zorder)[0]
                    if draw_arrow:
                        add_arrow(line, size=arrowsize, arrowstyle=arrowstyle, position=position,
                                  direction=arrowdirection)

        return self

    def draw_trajectory_from_vector(self, trajectory: np.ndarray,
                                    color: Optional[Union[str, Tuple[int, int, int]]] = 'k',
                                    linewidth: Optional[float] = 0.5, zorder: Optional[int] = 0, ):
        points = np.asarray([barycentric_to_xy_coordinates(x, self.corners) for x in trajectory])

        self.ax.plot(points[:, 0], points[:, 1], color, linewidth=linewidth, zorder=zorder)

        return self

    def draw_scatter_shadow(self, f: Callable[[np.ndarray, int], np.ndarray], nb_trajectories: int,
                            trajectory_length: Optional[int] = 15, step: Optional[float] = 0.1,
                            s: Optional[Union[float, ArrayLike]] = 0.1,
                            color: Optional[Union[str, Tuple[int, int, int]]] = 'whitesmoke',
                            marker: Optional[str] = '.', zorder: Optional[int] = 0) -> SelfSimplex2D:
        """
        Draws a series of point which follows trajectories in the simplex starting from random points.

        The visual effect is as if there were shadows in the direction of the gradient.

        Parameters
        ----------
        f: Callable[[np.ndarray, int], np.ndarray]
            Function that can calculate the gradient at any point in the simplex.
        nb_trajectories: int
            Number of trajectories to draw.
        trajectory_length: Optional[int]
            Length of the trajectory. This is used to calculate the amount of points odeint should calculate.
        step: Optional[float]
            The step size in time to get to the maximum trajectory length. Together with trajectory_length
            this indicates the amount of points odeint should calculate.
        s: Optional[Union[str, Tuple[int, int, int]]]
            Size of the points.
        color: Optional[Union[str, Tuple[int, int, int]]]
            The color of the points of the trajectory.
        marker: Optional[str]
            Style of the points to be drawn. See matplotlib markers.
        zorder: Optional[int]
            The order in which this plot should appear in the figure (above or bellow other plots).

        Returns
        -------
        Simplex2D
            A reference to the current object.
        """

        if self.discrete:
            if nb_trajectories > self.nb_states:
                nb_trajectories = self.nb_states
            initial_points = np.random.choice(range(self.nb_states), size=nb_trajectories, replace=False)
            for point in initial_points:
                u = sample_simplex(point, self.size, 3)
                x = odeint(f, u, np.arange(0, trajectory_length, step), full_output=False)
                # noinspection PyTypeChecker
                v = barycentric_to_xy_coordinates(x / self.size, self.corners)
                self.ax.scatter(v[:, 0], v[:, 1], s, color=color, marker=marker, zorder=zorder)
        else:
            for _ in range(nb_trajectories):
                u = sample_unit_simplex(3)
                x = odeint(f, u, np.arange(0, trajectory_length, step), full_output=False)
                # noinspection PyTypeChecker
                v = barycentric_to_xy_coordinates(x, self.corners)
                self.ax.scatter(v[:, 0], v[:, 1], s, color=color, marker=marker, zorder=zorder)

        return self

    def draw_stationary_distribution(self, stationary_distribution: np.ndarray,
                                     cmap: Optional[Union[str, matplotlib.colors.Colormap]] = 'binary',
                                     shading: Optional[str] = 'gouraud',
                                     alpha: Optional[float] = 1., edgecolors: Optional[str] = 'grey',
                                     vmin: Optional[float] = None, vmax: Optional[float] = None,
                                     zorder: Optional[int] = 0,
                                     colorbar: Optional[bool] = True,
                                     aspect: Optional[float] = 10,
                                     anchor: Optional[Tuple[float, float]] = (-0.5, 0.5),
                                     panchor: Optional[Tuple[float, float]] = (0, 0),
                                     shrink: Optional[float] = 0.6,
                                     label: Optional[str] = 'stationary distribution',
                                     label_rotation: Optional[int] = 270,
                                     label_fontsize: Optional[int] = 16,
                                     labelpad: Optional[float] = 20):
        """
        Draws the stationary distribution inside the simplex using a matplotlib.pyplot.tripcolor

        Parameters
        ----------
        stationary_distribution: numpy.ndarray
            An array containing the values of the stationary distribution. The order of these points
            must follow the order given by egttools.sample_simplex when iterating from 0-nb_states.
        cmap: Optional[Union[str, matplotlib.colors.Colormap]]
            Color map to be used.
        shading: Optional[str]
            Type of shading to be used in the plot. Can be either "gouraud" or "flat".
        alpha: Optional[float]
            The level of transparency.
        edgecolors: Optional[str]
            The colors of the edges of the triangular grid.
        vmin: Optional[flaot]
            The minimum value to take into account for the color range to plot.
        vmax: Optional[float]
            The maximum value to take into account for the color range to plot.
        zorder: Optional[int]
            The order in which this plot should appear in the figure (above or bellow other plots).
        colorbar: Optional[bool] = True
            Indicates whether to add a color bar to the plot.
        aspect: Optional[float]
            The aspect ration of the color bar.
        anchor: Optional[Tuple[float, float]]
            The anchor of the color bar.
        panchor: Optional[Tuple[float, float]]
            The panchor of the colorbar
        shrink: Optional[float]
            Ratio of shrinking the color bar.
        label: Optional[str]
            Label of the color bar.
        label_rotation: Optional[int]
            Rotation of the label.
        label_fontsize: Optional[int]
            Font size of the label.
        labelpad: Optional[float]
            How much padding should be added to the label.


        Returns
        -------
        Simplex2D
            A reference to the current object
        """
        if not self.discrete:
            raise Exception("The stationary distribution only exists in Finite populations modeled as a Markov Chain.")

        if shading != 'flat' and shading != 'gouraud':
            raise Exception("Shading can only the be 'flat' or 'gouraud'")

        sd_plot = self.ax.tripcolor(self.triangle_discrete, stationary_distribution, cmap=cmap, shading=shading,
                                    alpha=alpha, edgecolor=edgecolors,
                                    vmin=vmin, vmax=vmax,
                                    zorder=zorder)

        if colorbar:
            cbar = self.figure.colorbar(sd_plot, aspect=aspect, anchor=anchor, panchor=panchor, shrink=shrink,
                                        ax=self.ax)
            cbar.set_label(label, rotation=label_rotation, fontsize=label_fontsize, labelpad=labelpad)

        return self

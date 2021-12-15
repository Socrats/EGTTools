import matplotlib.pyplot as plt
import matplotlib.tri as tri
import numpy as np

from scipy.integrate import odeint
from matplotlib.patches import Circle
from typing import Optional, Tuple, List, Union, Callable, TypeVar
from numpy.typing import ArrayLike
from egttools.numerical import sample_unit_simplex
from egttools.plotting.helpers import barycentric_to_xy_coordinates, perturb_state, add_arrow

top_corner = np.sqrt(3) / 2
side_slope = np.sqrt(3)

SelfSimplex2D = TypeVar("SelfSimplex2D", bound="Simplex2D")


class Simplex2D:
    corners = np.array([
        [0, 0],
        [1 / 2, top_corner],
        [1, 0]
    ])
    triangle = tri.Triangulation(corners[:, 0], corners[:, 1])
    refiner = tri.UniformTriRefiner(triangle)
    trimesh = refiner.refine_triangulation(subdiv=5)

    def __init__(self, nb_points: Optional[int] = 1000):
        self.nb_points = nb_points
        self.Ux = None
        self.Uy = None
        self.stream = None
        x = np.linspace(0, 1, nb_points)
        y = np.linspace(0, 1, nb_points)
        self.X, self.Y = np.meshgrid(x, y)
        self.figure = None
        self.ax = None

    def add_axis(self, figsize: Optional[Tuple[int, int]] = (10, 8), ax: Optional[plt.axis] = None) -> SelfSimplex2D:
        if ax is not None:
            self.ax = ax
        else:
            self.figure, self.ax = plt.subplots(figsize=figsize)

        return self

    def get_figure_and_axis(self) -> Tuple[plt.figure, plt.axis]:
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

        """
        self.Ux = u.copy()
        self.Uy = v.copy()

        self.Ux = np.where(self.X >= self.Y / side_slope, self.Ux, np.nan)
        self.Ux = np.where(self.Y <= -side_slope * self.X + side_slope, self.Ux, np.nan)
        self.Ux = np.where(self.Y <= top_corner, self.Ux, np.nan)
        self.Ux = np.where(self.Y > 0, self.Ux, np.nan)
        self.Uy = np.where(self.X >= self.Y / side_slope, self.Uy, np.nan)
        self.Uy = np.where(self.Y <= -side_slope * self.X + side_slope, self.Uy, np.nan)
        self.Uy = np.where(self.Y <= top_corner, self.Uy, np.nan)
        self.Uy = np.where(self.Y > 0, self.Uy, np.nan)

        return self

    def draw_triangle(self, color: str = 'k', linewidth: int = 2) -> SelfSimplex2D:
        self.ax.triplot(self.triangle, color=color, linewidth=linewidth)
        return self

    def draw_gradients(self, arrowsize: Optional[float] = 2,
                       arrowstyle: Optional[str] = 'fancy',
                       color: Optional[Union[str, Tuple[int, int, int]]] = None, density: Optional[float] = 1,
                       linewidth: Optional[float] = 1.5) -> SelfSimplex2D:
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
                                         )
        return self

    def add_colorbar(self, aspect: Optional[float] = 10,
                     anchor: Optional[Tuple[float, float]] = (-0.5, 0.5),
                     panchor: Optional[Tuple[float, float]] = (0, 0),
                     shrink: Optional[float] = 0.6,
                     label: Optional[str] = 'intensity of selection',
                     label_rotation: Optional[int] = 270,
                     label_fontsize: Optional[int] = 16,
                     labelpad: Optional[float] = 20) -> SelfSimplex2D:
        cbar = plt.colorbar(self.stream.lines, aspect=aspect, anchor=anchor, panchor=panchor, shrink=shrink)
        cbar.set_label(label, rotation=label_rotation, fontsize=label_fontsize, labelpad=labelpad)

        return self

    def draw_stationary_points(self, roots: List[Union[Tuple[float, float], np.ndarray]], stability: List[bool],
                               zorder: Optional[int] = 5, linewidth: Optional[float] = 3) -> SelfSimplex2D:
        for i, stationary_point in enumerate(roots):
            if stability[i]:
                facecolor = 'k'
            else:
                facecolor = 'white'
            self.ax.add_artist(Circle(stationary_point, 0.015,
                                      edgecolor='k', facecolor=facecolor, zorder=zorder, linewidth=linewidth))
        return self

    def add_vertex_labels(self, labels=Union[Tuple[str, str, str], List[str]], epsilon_bottom: Optional[float] = 0.05,
                          epsilon_top: Optional[float] = 0.05,
                          fontsize: Optional[float] = 16,
                          horizontalalignment: Optional[str] = 'center') -> SelfSimplex2D:
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
        for _ in range(nb_trajectories):
            u = sample_unit_simplex(3)
            x = odeint(f, u, np.arange(0, trajectory_length, step), full_output=False)
            # noinspection PyTypeChecker
            v = barycentric_to_xy_coordinates(x, self.corners)
            self.ax.plot(v[:, 0], v[:, 1], color, ms=ms, zorder=zorder)

        return self

    def draw_trajectory_from_roots(self, f: Callable[[np.ndarray, int], np.ndarray], roots: List[np.ndarray],
                                   stability: List[np.ndarray],
                                   trajectory_length: Optional[int] = 15, step: Optional[float] = 0.1,
                                   color: Optional[Union[str, Tuple[int, int, int]]] = 'k',
                                   linewidth: Optional[float] = 0.5, zorder: Optional[int] = 0,
                                   draw_arrow: Optional[bool] = False, arrowstyle: Optional[str] = 'fancy',
                                   arrowsize: Optional[int] = 50,
                                   position: Optional[int] = None,
                                   arrowdirection: Optional[str] = 'right') -> SelfSimplex2D:
        for i, stationary_point in enumerate(roots):
            if stability[i]:  # we don't plot arrows starting at stable points
                continue
            states = perturb_state(stationary_point)
            for state in states:
                x = odeint(f, state, np.arange(0, trajectory_length, step), full_output=False)

                # noinspection PyTypeChecker
                v = barycentric_to_xy_coordinates(x, self.corners)
                line = self.ax.plot(v[:, 0], v[:, 1], color, linewidth=linewidth, zorder=zorder)[0]
                if draw_arrow:
                    add_arrow(line, size=arrowsize, arrowstyle=arrowstyle, position=position, direction=arrowdirection)

        return self

    def draw_scatter_shadow(self, f: Callable[[np.ndarray, int], np.ndarray], nb_trajectories: int,
                            trajectory_length: Optional[int] = 15, step: Optional[float] = 0.1,
                            s: Optional[Union[float, ArrayLike]] = 0.1,
                            color: Optional[Union[str, Tuple[int, int, int]]] = 'whitesmoke',
                            marker: Optional[str] = '.', zorder: Optional[int] = 0) -> SelfSimplex2D:
        for _ in range(nb_trajectories):
            u = sample_unit_simplex(3)
            x = odeint(f, u, np.arange(0, trajectory_length, step), full_output=False)
            # noinspection PyTypeChecker
            v = barycentric_to_xy_coordinates(x, self.corners)
            self.ax.scatter(v[:, 0], v[:, 1], s, color=color, marker=marker, zorder=zorder)

        return self

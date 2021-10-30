# # Copyright (c) 2019-2020  Elias Fernandez
# #
# # This file is part of EGTtools.
# #
# # EGTtools is free software: you can redistribute it and/or modify
# # it under the terms of the GNU General Public License as published by
# # the Free Software Foundation, either version 3 of the License, or
# # (at your option) any later version.
# #
# # EGTtools is distributed in the hope that it will be useful,
# # but WITHOUT ANY WARRANTY; without even the implied warranty of
# # MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# # GNU General Public License for more details.
# #
# # You should have received a copy of the GNU General Public License
# # along with EGTtools.  If not, see <http://www.gnu.org/licenses/>
#
# # This code was taken from Marvin Boe's repository
#
# import math
#
# import matplotlib.pyplot as plt
# import matplotlib.tri as tri
# import numpy as np
# import scipy.optimize
#
# from typing import Callable
#
# try:
#     from egttools.numerical import sample_simplex, calculate_nb_states
# except ImportError:
#     raise
#
#
# class SimplexDynamics:
#     """draws dynamics of given function and
#     corresponding fixed points into triangle"""
#     # corners of triangle and calculation of points
#     r0 = np.array([1 / 2., np.sqrt(3) / 2.])
#     r1 = np.array([0, 0])
#     r2 = np.array([1, 0])
#     corners = np.array([r0, r1, r2])
#     triangle = tri.Triangulation(corners[:, 0], corners[:, 1])
#     refiner = tri.UniformTriRefiner(triangle)
#     trimesh = refiner.refine_triangulation(subdiv=5)
#     trimesh_fine = refiner.refine_triangulation(subdiv=5)
#
#     def __init__(self, fun: Callable[[np.ndarray, int], np.ndarray], discrete=False, pop_size=None,
#                  nb_points=100) -> None:
#         self.f = fun
#         self.discrete = discrete
#         if self.discrete:
#             nb_states = calculate_nb_states(pop_size, 3)
#             points = np.random.choice(nb_states, nb_points, replace=False)
#             self.simplex_points = np.asarray([sample_simplex(point, pop_size, 3) for point in points])
#             tmp = np.asarray([self.ba2xy(point) for point in self.simplex_points / Z])
#             self.trimesh.x = tmp[:, 0]
#             self.trimesh.y = tmp[:, 1]
#         self.calculate_stationary_points()
#         self.calc_direction_and_strength()
#
#     # barycentric coordinates
#     def xy2ba(self, x: float, y: float) -> np.ndarray:
#         corner_x = self.corners.T[0]
#         corner_y = self.corners.T[1]
#         x_1 = corner_x[0]
#         x_2 = corner_x[1]
#         x_3 = corner_x[2]
#         y_1 = corner_y[0]
#         y_2 = corner_y[1]
#         y_3 = corner_y[2]
#         l1 = ((y_2 - y_3) * (x - x_3) + (x_3 - x_2) * (y - y_3)) / (
#                 (y_2 - y_3) * (x_1 - x_3) + (x_3 - x_2) * (y_1 - y_3))
#         l2 = ((y_3 - y_1) * (x - x_3) + (x_1 - x_3) * (y - y_3)) / (
#                 (y_2 - y_3) * (x_1 - x_3) + (x_3 - x_2) * (y_1 - y_3))
#         l3 = 1 - l1 - l2
#         return np.array([l1, l2, l3])
#
#     def ba2xy(self, x: np.array) -> np.array:
#         return self.corners.T.dot(x.T).T
#
#     def calculate_stationary_points(self) -> None:
#         fp_raw = []
#         border = 5  # don't check points close to simplex border
#         delta = 1e-12
#         for x, y in zip(self.trimesh.x[border:-border], self.trimesh.y[border:-border]):
#             start = self.xy2ba(x, y)
#             fp_try = np.array([])
#
#             sol = scipy.optimize.root(self.f, start, args=(0,), method="hybr")  # ,xtol=1.49012e-10,maxfev=1000
#             if sol.success:
#                 fp_try = sol.x
#                 # check if FP is in simplex
#                 if not math.isclose(np.sum(fp_try), 1., abs_tol=2.e-3):
#                     continue
#                 if not np.all((fp_try > -delta) & (fp_try < 1 + delta)):  # only if fp in simplex
#                     continue
#             else:
#                 continue
#             # only add new fixed points to list
#             if not np.array([np.allclose(fp_try, x, atol=1e-7) for x in fp_raw]).any():
#                 fp_raw.append(fp_try.tolist())
#         # add fixed points in correct coordinates to fixpoints list
#         fp_raw = np.array(fp_raw)
#         if fp_raw.shape[0] > 0:
#             self.fixpoints = self.corners.T.dot(np.array(fp_raw).T).T
#         else:
#             self.fixpoints = np.array([])
#
#     def calc_direction_and_strength(self) -> None:
#         if self.discrete:
#             direction = [self.f(x, 0) for x in self.simplex_points]
#         else:
#             direction = [self.f(self.xy2ba(x, y), 0) for x, y in zip(self.trimesh.x, self.trimesh.y)]
#         self.direction_norm = np.array(
#             [self.ba2xy(v) / np.linalg.norm(v) if np.linalg.norm(v) > 0 else np.array([0, 0]) for v in direction])
#         self.direction_norm = self.direction_norm
#         # print(direction_ba_norm)
#         self.pvals = [np.linalg.norm(v) for v in direction]
#         self.direction = np.array([self.ba2xy(v) for v in direction])
#
#     def plot_simplex(self, ax: plt.axis, cmap='viridis', type_labels=("A", "B", "C"), **kwargs: {None, dict}) -> None:
#
#         ax.triplot(self.triangle, linewidth=0.8, color="black")
#         ax.tricontourf(self.trimesh, self.pvals, alpha=0.8, cmap=cmap, **kwargs)
#
#         # arrow plot options:
#         Q = ax.quiver(self.trimesh.x, self.trimesh.y, self.direction_norm.T[0], self.direction_norm.T[1], angles='xy',
#                       pivot='mid')  # pivot='tail')#
#
#         ax.axis('equal')
#         ax.axis('off')
#         margin = 0.02
#         ax.set_ylim(ymin=-margin, ymax=self.r0[1] + margin)
#         ax.set_xlim(xmin=-margin, xmax=1. + margin)
#
#         # timescatter=ax.scatter(points[::5,0],points[::5,1],c=t[::5],linewidth=0.0,cmap='viridis',alpha=.5)
#         if self.fixpoints.shape[0] > 0:
#             ax.scatter(self.fixpoints[:, 0], self.fixpoints[:, 1], c="black", s=70, linewidth=0.3)
#         # fig.colorbar(timescatter,label="time")
#         ax.annotate(type_labels[0], self.corners[0], xytext=self.corners[0] + np.array([0.0, 0.05]),
#                     horizontalalignment='center', va='top')
#         ax.annotate(type_labels[1], self.corners[1], xytext=self.corners[1] + np.array([0.0, -0.02]),
#                     horizontalalignment='center', va='top')
#         ax.annotate(type_labels[2], self.corners[2], xytext=self.corners[2] + np.array([0.0, -0.05]),
#                     horizontalalignment='center', va='bottom')
#
#
# if __name__ == "__main__":
#     import matplotlib.pyplot as plt
#     import egttools as egt
#
#     # from egtplot import plot_static
#
#     A = np.array([[1, 0, 0],
#                   [0, 2, 0],
#                   [0, 0, 3]])
#
#     Z = 100
#     beta = 1
#     nb_strategies = 3
#     # evolver = egt.analytical.StochDynamics(3, A, Z)
#     # evolver.mu = 1e-3
#     evolver = egt.analytical.replicator_equation
#
#
#     def f(x, t):
#         # x_int = (Z * x).astype(int)
#         # return evolver.full_gradient_selection(x, beta)
#         return evolver(x, A)
#
#
#     dynamics = SimplexDynamics(f, discrete=False, pop_size=Z, nb_points=5000)
#     fig, ax = plt.subplots()
#     dynamics.plot_simplex(ax)
#     plt.show()
#     #
#     #
#     # def f2(x, t):
#     #     return egt.analytical.replicator_equation(x, A)
#     #
#     #
#     # dynamics = SimplexDynamics(f2)
#     # fig2, ax = plt.subplots()
#     # dynamics.plot_simplex(ax)
#     #
#     # plot_static(A.reshape((9, 1)).tolist())
#     # plt.show()

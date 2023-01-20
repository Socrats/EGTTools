# Copyright (c) 2019-2021  Elias Fernandez
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

"""
Set of vectorized functions that can be used to apply these functions on large tensors.
"""
try:
    from ..numerical.numerical_ import \
        vectorized_replicator_equation_n_player as cpp_vectorized_replicator_equation_n_player
except Exception:
    raise Exception("numerical package not initialized")
else:
    import numpy as np
    from ..plotting.helpers import barycentric_to_xy_coordinates
    from ..analytical import replicator_equation_n_player


def vectorized_replicator_equation(frequencies: np.ndarray, payoffs: np.ndarray) -> np.ndarray:
    """
    This function provides an easy way to calculate a matrix of gradients in a simplex in one go.

    The input `frequencies` is expected to be a 3 dimensional tensor of shape (p, m, n) while the payoffs
    matrix is expected to be of shape (p, p).

    The main intention of this helper function is to facilitate
    the calculation of the gradients that are required by the `plot_gradients` method of the
    `egttools.Simplex2D` class.

    Parameters
    ----------
    frequencies: numpy.ndarray[p,m,n]
        A 3 dimensional tensor containing the set of population frequencies for which the gradient should be
        calculated.
    payoffs: numpy.ndarray[p,p]
        A 2 dimensional matrix containing the payoffs of the game.

    Returns
    -------
    numpy.ndarray[p,m,n]
        The gradients for each of the input frequencies.
    """
    axs = [np.dot(payoffs, frequencies[:, i, :]) for i in range(frequencies.shape[1])]
    return np.asarray([frequencies[:, i, :] * (axs[i] - np.diagonal(np.dot(frequencies[:, i, :].T, axs[i]))) for i in
                       range(frequencies.shape[1])]).swapaxes(0, 1)


def vectorized_replicator_equation_n_player(frequencies: np.ndarray, payoffs: np.ndarray,
                                            group_size: int) -> np.ndarray:
    """
    This function provides an easy way to calculate a matrix of gradients in a simplex in one go.

    The input `frequencies` is expected to be a 3 dimensional tensor of shape (p, m, n) while the payoffs
    matrix is expected to be of shape (p, p).

    The main intention of this helper function is to facilitate
    the calculation of the gradients that are required by the `plot_gradients` method of the
    `egttools.Simplex2D` class.

    Parameters
    ----------
    frequencies: numpy.ndarray[p,m,n]
        A 3 dimensional tensor containing the set of population frequencies for which the gradient should be
        calculated.
    payoffs: numpy.ndarray[p,p]
        A 2 dimensional matrix containing the payoffs of the game.
    group_size: int
        Size of the group

    Returns
    -------
    numpy.ndarray[p,m,n]
        The gradients for each of the input frequencies.
    """
    res1, res2, res3 = cpp_vectorized_replicator_equation_n_player(frequencies[0, :, :], frequencies[1, :, :],
                                                                   frequencies[2, :, :], payoffs, group_size)

    results = np.zeros_like(frequencies)
    results[0, :, :] = res1[:, :]
    results[1, :, :] = res2[:, :]
    results[2, :, :] = res3[:, :]

    return results


def vectorized_barycentric_to_xy_coordinates(barycentric_coordinates: np.ndarray, corners: np.ndarray) -> np.ndarray:
    """
    Transform a tensor of barycentric coordinates to cartesian coordinates.

    Parameters
    ----------
    barycentric_coordinates : numpy.ndarray[3,m,n]
        Expects a matrix in which the first dimension corresponds to the vector of 3-demensional barycentric
        coordinates.

    corners : numpy.ndarray[3,]
        The corners of the triangle

    Returns
    -------
    numpy.ndarray[2,m,n]
        The tensor of cartesian coordinates.
    """
    return np.asarray([[barycentric_to_xy_coordinates(barycentric_coordinates[:, i, j], corners)
                        for j in range(barycentric_coordinates.shape[1])]
                       for i in range(barycentric_coordinates.shape[2])])

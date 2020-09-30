"""
Copyright (c) 2019 Elias Fernandez

This python module contains some utility functions
to find saddle points and plot gradients in 2 player, 2 strategy games.
"""

import numpy as np


def find_saddle_type_and_gradient_direction(gradient, saddle_points_idx, offset=0.01):
    """
    Finds whether a saddle point is stable or not. And defines the direction of the
    gradient among stable and unstable points.

    Parameters
    ----------
    gradient : array containing the gradient of selection for all states of the population
    saddle_points_idx : array containing the saddle points indices
    offset : offset for the gradient_directions, so that arrows don't overlap with point

    Returns
    -------
     tuple containing an array that indicates the type of saddle points and another array indicating
     the direction of the gradient between unstable and stable points
    """
    saddle_type = []
    gradient_direction = []
    nb_points = len(gradient)
    real_offset = offset * (nb_points - 1)
    for i, point in enumerate(saddle_points_idx):
        if point < nb_points - 1:
            if point > 0:
                if gradient[point + 1] > 0:
                    saddle_type.append(False)
                    gradient_direction.append((point, saddle_points_idx[i + 1] - real_offset))
                elif gradient[point - 1] < 0:
                    saddle_type.append(False)
                    gradient_direction.append((point, saddle_points_idx[i - 1] + real_offset))
                else:
                    gradient_direction.append((point + 1, 0 + real_offset))
                    saddle_type.append(True)
            else:
                if gradient[point + 1] > 0:
                    saddle_type.append(False)
                    gradient_direction.append((point, saddle_points_idx[i + 1] - real_offset))
                else:
                    saddle_type.append(True)
        else:
            if gradient[point - 1] < 0:
                saddle_type.append(False)
                gradient_direction.append((point, saddle_points_idx[i - 1] + real_offset))
            else:
                saddle_type.append(True)
    saddle_type = np.asarray(saddle_type)
    gradient_direction = np.asarray(gradient_direction) / (nb_points - 1)
    return saddle_type, gradient_direction

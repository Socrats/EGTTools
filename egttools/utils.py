"""
Copyright (c) 2019 Elias Fernandez

This python module contains some utility functions
to find saddle points and plot gradients in 2 player, 2 strategy games.
"""

import numpy as np
import matplotlib.pyplot as plt


def find_saddle_type_and_gradient_direction(gradient, saddle_points_idx, offset=0.01):
    """
    Finds whether a saddle point is stable or not. And defines the direction of the
    gradient among stable and unstable points.
    :param gradient : array containing the gradient of selection for all states of the population
    :param saddle_points_idx : array containing the saddle points indices
    :param offset : offset for the gradient_directions, so that arrows don't overlap with point
    :return tuple containing an array that indicates the type of saddle points and another array indicating
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


def plot_gradient(x, gradients, saddle_points, saddle_type, gradient_direction, fig_title='', xlabel='', figsize=(5, 4),
                  **kwargs):
    """
    Creates a figure plotting the gradient of selection toghether with the saddle points,
    and the gradient arrows.
    :param x: vector containing the possible states in x axis. It must have the same length as gradient
    :param gradients: vector containing the gradient for each possible state
    :param saddle_points: vector containing all saddle points
    :param saddle_type: vector of booleans indicating whether or not the saddle point is stable
    :param gradient_direction: vector of points indicating the direction of the gradient
                               between unstable and stable saddle points
    :param fig_title: a string containing the title of the figure
    :param xlabel: label for x axis
    :param figsize: a tuple indicating the size of the figure
    :param kwargs: you may pass an axis object
    :returns a figure object
    """
    if 'ax' not in kwargs:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        ax = kwargs['ax']
    ax.plot(x, gradients, zorder=1)
    ax.plot([0, 1], [0, 0], 'k', zorder=1)
    # First we plot unstable points
    if saddle_points[~saddle_type].any():
        points1 = ax.scatter(x=saddle_points[~saddle_type],
                             y=np.zeros((len(saddle_points[~saddle_type], ))),
                             marker="o", color='k', s=80,
                             facecolors='white',
                             zorder=3)
        points1.set_clip_on(False)  # Plot points over axis
    # Then we plot stable points
    if saddle_points[saddle_type].any():
        points2 = ax.scatter(x=saddle_points[saddle_type],
                             y=np.zeros((len(saddle_points[saddle_type], ))),
                             marker="o",
                             color='k',
                             s=80,
                             zorder=3)
        points2.set_clip_on(False)
    # Plot arrows following the gradient
    for point in gradient_direction:
        ax.annotate("",
                    xy=(point[1], 0.0), xycoords='data',
                    xytext=(point[0], 0.0), textcoords='data',
                    arrowprops=dict(arrowstyle="->",
                                    connectionstyle="arc3"),
                    zorder=2
                    )
    # Set the axis style
    ax.set_xlim(0, 1)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('$G(x)$')
    ax.set_title(fig_title)
    return ax

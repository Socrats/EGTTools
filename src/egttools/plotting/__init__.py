"""
API reference documentation for the `plotting` submodule.
"""

from .indicators import plot_gradient, plot_gradients, draw_stationary_distribution
from .simplex2d import Simplex2D
from .simplified import plot_replicator_dynamics_in_simplex, plot_moran_dynamics_in_simplex

__all__ = ['plot_gradient', 'plot_gradients', 'draw_stationary_distribution', 'Simplex2D',
           'plot_replicator_dynamics_in_simplex', 'plot_moran_dynamics_in_simplex']

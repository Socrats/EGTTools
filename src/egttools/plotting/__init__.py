"""
API reference documentation for the `plotting` submodule.
"""

from .indicators import plot_gradient, plot_gradients, draw_invasion_diagram
from .simplex2d import Simplex2D
from .simplified import plot_replicator_dynamics_in_simplex, plot_pairwise_comparison_rule_dynamics_in_simplex

__all__ = ['plot_gradient', 'plot_gradients', 'draw_invasion_diagram', 'Simplex2D',
           'plot_replicator_dynamics_in_simplex', 'plot_pairwise_comparison_rule_dynamics_in_simplex']

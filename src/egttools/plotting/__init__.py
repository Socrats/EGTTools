"""
API reference documentation for the `plotting` submodule.
"""

from .indicators import (
    plot_gradient,
    plot_gradients,
    draw_invasion_diagram
)
from .network_plots import (
    plot_network_state,
    plot_strategy_evolution,
    animate_network_evolution,
    plot_edge_homophily,
    plot_parameter_sweep,
    plot_strategy_by_degree,
)
from .simplex2d import Simplex2D
try:
    from .simplex3d import Simplex3D
except ImportError:
    pass  # plotly not installed; Simplex3D unavailable
from .simplified import (
    plot_replicator_dynamics_in_simplex,
    plot_pairwise_comparison_rule_dynamics_in_simplex,
    plot_pairwise_comparison_rule_dynamics_in_simplex_without_roots
)

__all__ = ['plot_gradient', 'plot_gradients', 'draw_invasion_diagram', 'Simplex2D',
           'plot_replicator_dynamics_in_simplex', 'plot_pairwise_comparison_rule_dynamics_in_simplex',
           'plot_pairwise_comparison_rule_dynamics_in_simplex_without_roots']

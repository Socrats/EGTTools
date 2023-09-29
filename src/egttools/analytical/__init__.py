"""
API reference documentation for the `analytical` submodule.
"""

try:
    from ..numerical.numerical_ import PairwiseComparison, replicator_equation_n_player
except Exception:
    raise Exception("numerical package not initialized")
else:
    from .sed_analytical import replicator_equation
    from .sed_analytical import StochDynamics

__all__ = ['replicator_equation', 'StochDynamics', 'PairwiseComparison', 'replicator_equation_n_player']

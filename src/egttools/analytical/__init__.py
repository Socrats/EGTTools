"""
API reference documentation for `sed_analytical` submodule.
"""

from .sed_analytical import replicator_equation
from .sed_analytical import StochDynamics

from ..numerical.numerical import PairwiseComparison, replicator_equation_n_player

__all__ = ['replicator_equation', 'StochDynamics', 'PairwiseComparison', 'replicator_equation_n_player']

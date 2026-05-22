"""The structure submodule contains population structures"""

try:
    from ..numerical_.structure import (
        AbstractStructure,
        Network, NetworkGroup, NetworkSync, NetworkGroupSync,
        NetworkMCEstimatorPC, NetworkMCEstimatorBD,
        NetworkMCEstimatorDB, NetworkMCEstimatorTDPC,
        NetworkCoEvolutionaryPC, NetworkCoEvolutionaryPCHomophilic,
    )
except Exception:
    raise Exception("numerical package not initialized")

from ..estimators import NetworkEstimator  # noqa: E402

__all__ = [
    'AbstractStructure',
    'Network', 'NetworkGroup', 'NetworkSync', 'NetworkGroupSync',
    'NetworkMCEstimatorPC', 'NetworkMCEstimatorBD',
    'NetworkMCEstimatorDB', 'NetworkMCEstimatorTDPC',
    'NetworkCoEvolutionaryPC', 'NetworkCoEvolutionaryPCHomophilic',
    'NetworkEstimator',
]

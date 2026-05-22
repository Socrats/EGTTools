"""The structure submodule contains population structures"""

try:
    from ..numerical_.structure import (
        AbstractStructure,
        NetworkMCEstimatorPC, NetworkMCEstimatorBD,
        NetworkMCEstimatorDB, NetworkMCEstimatorTDPC, NetworkMCEstimatorLP,
        NetworkCoEvolutionaryPC, NetworkCoEvolutionaryPCHomophilic,
        run_network_sweep,
    )
except Exception:
    raise Exception("numerical package not initialized")

from ..estimators import NetworkEstimator  # noqa: E402

__all__ = [
    'AbstractStructure',
    'NetworkMCEstimatorPC', 'NetworkMCEstimatorBD',
    'NetworkMCEstimatorDB', 'NetworkMCEstimatorTDPC', 'NetworkMCEstimatorLP',
    'NetworkCoEvolutionaryPC', 'NetworkCoEvolutionaryPCHomophilic',
    'NetworkEstimator',
    'run_network_sweep',
]

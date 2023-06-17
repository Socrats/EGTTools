"""The structure submodule contains population structures"""

try:
    from ..numerical_.structure import (AbstractStructure, Network, NetworkGroup,
                                        NetworkSync, NetworkGroupSync)
except Exception:
    raise Exception("numerical package not initialized")

__all__ = ['AbstractStructure', 'Network', 'NetworkGroup', 'NetworkSync', 'NetworkGroupSync']

"""
API reference documentation for `behaviors.NormalForm.TwoActions` submodule.
"""

try:
    from egttools.numerical.numerical_.behaviors.NormalForm.TwoActions import (ActionInertia, Cooperator, Defector,
                                                                               GRIM,
                                                                               GenerousTFT,
                                                                               GradualTFT, ImperfectTFT, Pavlov, Random,
                                                                               SuspiciousTFT,
                                                                               TFT, TFTT, TTFT, )
except Exception:
    raise Exception("numerical package not initialized")
else:
    from .nfg_strategies import EpsilonTFT, EpsilonGRIM, Detective, MemoryOneStrategy

__all__ = ['ActionInertia', 'Cooperator', 'Defector', 'GRIM', 'GenerousTFT', 'GradualTFT', 'ImperfectTFT', 'Pavlov',
           'Random', 'SuspiciousTFT', 'TFT', 'TFTT', 'TTFT', 'EpsilonTFT', 'EpsilonGRIM', 'Detective',
           'MemoryOneStrategy']

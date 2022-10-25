"""
API reference documentation for `behaviors.NormalForm.TwoActions` submodule.
"""

from egttools.numerical.numerical.behaviors.NormalForm.TwoActions import (ActionInertia, Cooperator, Defector, GRIM,
                                                                          GenerousTFT,
                                                                          GradualTFT, ImperfectTFT, Pavlov, Random,
                                                                          SuspiciousTFT,
                                                                          TFT, TFTT, TTFT, )
from .nfg_strategies import EpsilonTFT, EpsilonGRIM, Detective, MemoryOneStrategy

__all__ = ['ActionInertia', 'Cooperator', 'Defector', 'GRIM', 'GenerousTFT', 'GradualTFT', 'ImperfectTFT', 'Pavlov',
           'Random', 'SuspiciousTFT', 'TFT', 'TFTT', 'TTFT', 'EpsilonTFT', 'EpsilonGRIM', 'Detective',
           'MemoryOneStrategy']

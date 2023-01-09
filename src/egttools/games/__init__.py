"""
API reference documentation for the `games` submodule.
"""

from egttools.numerical.numerical.games import (AbstractGame,
                                                AbstractSpatialGame,
                                                AbstractNPlayerGame,
                                                NormalFormGame,
                                                NormalFormNetworkGame,
                                                CRDGame,
                                                CRDGameTU,
                                                OneShotCRD,
                                                Matrix2PlayerGameHolder,
                                                MatrixNPlayerGameHolder, )

from .pgg import PGG
from .informal_risk import InformalRiskGame
from .abstract_games import AbstractTwoPLayerGame
from .nonlinear_games import NPlayerStagHunt, CommonPoolResourceDilemma, CommonPoolResourceDilemmaCommitment

__all__ = ['AbstractGame', 'AbstractSpatialGame', 'AbstractNPlayerGame', 'NormalFormGame', 'NormalFormNetworkGame',
           'CRDGame', 'CRDGameTU',
           'OneShotCRD', 'PGG',
           'InformalRiskGame', 'AbstractTwoPLayerGame', 'NPlayerStagHunt', 'CommonPoolResourceDilemma',
           'CommonPoolResourceDilemmaCommitment', 'Matrix2PlayerGameHolder', 'MatrixNPlayerGameHolder']

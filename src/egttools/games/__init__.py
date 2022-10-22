"""
API reference documentation for the `games` submodule.
"""

from egttools.numerical.numerical.games import (AbstractGame,
                                                AbstractNPlayerGame,
                                                NormalFormGame,
                                                CRDGame,
                                                CRDGameTU,
                                                OneShotCRD, )

from .pgg import PGG
from .informal_risk import InformalRiskGame
from .abstract_games import AbstractTwoPLayerGame
from .nonlinear_games import NPlayerStagHunt, CommonPoolResourceDilemma, CommonPoolResourceDilemmaCommitment

__all__ = ['AbstractGame', 'AbstractNPlayerGame', 'NormalFormGame', 'CRDGame', 'CRDGameTU', 'OneShotCRD', 'PGG',
           'InformalRiskGame', 'AbstractTwoPLayerGame', 'NPlayerStagHunt', 'CommonPoolResourceDilemma',
           'CommonPoolResourceDilemmaCommitment']

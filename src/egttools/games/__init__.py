"""
API reference documentation for the `games` submodule.
"""

try:
    from ..numerical.numerical_.games import (AbstractGame,
                                              AbstractSpatialGame,
                                              AbstractNPlayerGame,
                                              NormalFormGame,
                                              NormalFormNetworkGame,
                                              CRDGame,
                                              CRDGameTU,
                                              OneShotCRD,
                                              OneShotCRDNetworkGame,
                                              Matrix2PlayerGameHolder,
                                              MatrixNPlayerGameHolder, )
except Exception:
    raise Exception("numerical package not initialized")
else:
    from .pgg import PGG
    from .informal_risk import InformalRiskGame
    from .abstract_games import AbstractTwoPLayerGame
    from .nonlinear_games import NPlayerStagHunt, CommonPoolResourceDilemma, CommonPoolResourceDilemmaCommitment

__all__ = ['AbstractGame', 'AbstractSpatialGame', 'AbstractNPlayerGame', 'NormalFormGame', 'NormalFormNetworkGame',
           'CRDGame', 'CRDGameTU',
           'OneShotCRD', 'OneShotCRDNetworkGame', 'PGG',
           'InformalRiskGame', 'AbstractTwoPLayerGame', 'NPlayerStagHunt', 'CommonPoolResourceDilemma',
           'CommonPoolResourceDilemmaCommitment', 'Matrix2PlayerGameHolder', 'MatrixNPlayerGameHolder']

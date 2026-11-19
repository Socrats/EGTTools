"""
The `egttools.numerical.behaviors` submodule contains the available strategies to evolve.
"""
from __future__ import annotations
import typing
from . import CRD
from . import NormalForm
__all__ = ['CRD', 'NormalForm', 'call_get_action']
def call_get_action(strategies: list, time_step: typing.SupportsInt | typing.SupportsIndex, prev_action: typing.SupportsInt | typing.SupportsIndex) -> str:
    """
                Return a string representation of the actions chosen by a list of strategies.
    
                Parameters
                ----------
                strategies : list[AbstractNFGStrategy]
                    Strategies to query.
                time_step : int
                    Current round.
                prev_action : int
                    Previous action of the opponent.
    
                Returns
                -------
                str
                    Tuple-like string with the action chosen by each strategy.
    """
__init__: str = 'The `egttools.numerical.behaviors` submodule contains the available strategies to evolve.'

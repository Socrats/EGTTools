"""
The `egttools.numerical.behaviors.NormalForm` submodule contains the strategies for normal-form games.
"""
from __future__ import annotations
from . import TwoActions
__all__: list[str] = ['AbstractNFGStrategy', 'TwoActions']
class AbstractNFGStrategy:
    """
    
                Abstract base class for strategies in repeated two-action normal-form games.
            
    """
    def __init__(self) -> None:
        ...
    def get_action(self, time_step: int, action_prev: int) -> int:
        """
                        Return the action chosen by the strategy.
        
                        Parameters
                        ----------
                        time_step : int
                            Current round.
                        action_prev : int
                            Previous action of the opponent.
        
                        Returns
                        -------
                        int
                            Action selected by the strategy.
        
                        See Also
                        --------
                        egttools.behaviors.NormalForm.TwoActions.Cooperator
                        egttools.behaviors.NormalForm.TwoActions.Defector
                        egttools.behaviors.NormalForm.TwoActions.Random
                        egttools.behaviors.NormalForm.TwoActions.TFT
                        egttools.behaviors.NormalForm.TwoActions.SuspiciousTFT
                        egttools.behaviors.NormalForm.TwoActions.GenerousTFT
                        egttools.behaviors.NormalForm.TwoActions.GradualTFT
                        egttools.behaviors.NormalForm.TwoActions.ImperfectTFT
                        egttools.behaviors.NormalForm.TwoActions.TFTT
                        egttools.behaviors.NormalForm.TwoActions.TTFT
                        egttools.behaviors.NormalForm.TwoActions.GRIM
                        egttools.behaviors.NormalForm.TwoActions.Pavlov
        """
    def is_stochastic(self) -> bool:
        """
        Return whether the strategy is stochastic.
        """
    def type(self) -> str:
        """
        Return the strategy type.
        """

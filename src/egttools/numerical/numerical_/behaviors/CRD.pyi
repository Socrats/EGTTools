"""
The `egttools.numerical.behaviors.CRD` submodule contains the strategies for collective-risk dilemma games.
"""
from __future__ import annotations
__all__: list[str] = ['AbstractCRDStrategy', 'CRDMemoryOnePlayer']
class AbstractCRDStrategy:
    """
    
                Abstract base class for strategies in collective-risk dilemma games.
            
    """
    def __init__(self) -> None:
        ...
    def get_action(self, time_step: int, group_contributions_prev: int) -> int:
        """
                        Return the action chosen by the strategy.
        
                        Parameters
                        ----------
                        time_step : int
                            Current round.
                        group_contributions_prev : int
                            Sum of contributions of the other group members in the previous round.
        
                        Returns
                        -------
                        int
                            Action selected by the strategy.
        
                        See Also
                        --------
                        egttools.behaviors.CRD.CRDMemoryOnePlayer
        """
    def type(self) -> str:
        """
        Return the strategy type.
        """
class CRDMemoryOnePlayer(AbstractCRDStrategy):
    """
    Memory-one strategy for collective-risk dilemma games.
    """
    def __init__(self, personal_threshold: int, initial_action: int, action_above: int, action_equal: int, action_below: int) -> None:
        """
                        Construct a memory-one strategy for a collective-risk dilemma.
        
                        The strategy contributes `initial_action` in the first round. In later rounds,
                        it compares the sum of contributions of the other group members in the previous
                        round to `personal_threshold`.
        
                        - if the previous group contribution is greater than `personal_threshold`,
                          the player chooses `action_above`;
                        - if it is equal to `personal_threshold`, the player chooses `action_equal`;
                        - if it is smaller than `personal_threshold`, the player chooses `action_below`.
        
                        Parameters
                        ----------
                        personal_threshold : int
                            Threshold against which the previous group contribution is compared.
                        initial_action : int
                            Contribution in the first round.
                        action_above : int
                            Contribution used when the previous group contribution is above the threshold.
                        action_equal : int
                            Contribution used when the previous group contribution equals the threshold.
                        action_below : int
                            Contribution used when the previous group contribution is below the threshold.
        
                        See Also
                        --------
                        egttools.behaviors.CRD.AbstractCRDStrategy
                        egttools.games.CRDGame
                        egttools.games.CRDGameTU
        """
    def __str__(self) -> str:
        ...
    def get_action(self, time_step: int, group_contributions_prev: int) -> int:
        """
                        Return the action chosen by the strategy.
        
                        Parameters
                        ----------
                        time_step : int
                            Current round.
                        group_contributions_prev : int
                            Sum of contributions of the other group members in the previous round.
        
                        Returns
                        -------
                        int
                            Action selected by the strategy.
        
                        Examples
                        --------
                        >>> from egttools.behaviors.CRD import CRDMemoryOnePlayer
                        >>> strategy = CRDMemoryOnePlayer(4, 2, 4, 2, 0)
                        >>> strategy.get_action(0, 0)
                        2
        """
    def type(self) -> str:
        """
        Return the strategy type.
        """

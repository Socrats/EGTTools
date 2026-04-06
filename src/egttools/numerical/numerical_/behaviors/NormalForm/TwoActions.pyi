"""
The `egttools.numerical.behaviors.NormalForm.TwoActions` submodule contains the strategies for two-action normal-form games.
"""
from __future__ import annotations
import egttools.numerical.numerical_.behaviors.NormalForm
__all__: list[str] = ['ActionInertia', 'Cooperator', 'Defector', 'GRIM', 'GenerousTFT', 'GradualTFT', 'ImperfectTFT', 'Pavlov', 'Random', 'SuspiciousTFT', 'TFT', 'TFTT', 'TTFT']
class ActionInertia(egttools.numerical.numerical_.behaviors.NormalForm.AbstractNFGStrategy):
    """
    Strategy that tends to repeat its previous action.
    """
    def __init__(self, epsilon: float, p: float) -> None:
        """
                        Construct an ActionInertia strategy.
        
                        The strategy repeats its current action, but explores a different action
                        with probability `epsilon`. In the first round it cooperates with probability `p`.
        
                        Parameters
                        ----------
                        epsilon : float
                            Probability of changing action.
                        p : float
                            Probability of cooperation in the first round.
        """
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
        """
    def is_stochastic(self) -> bool:
        """
        Return whether the strategy is stochastic.
        """
    def type(self) -> str:
        """
        Return the strategy type.
        """
class Cooperator(egttools.numerical.numerical_.behaviors.NormalForm.AbstractNFGStrategy):
    """
    Strategy that always cooperates.
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
        """
    def is_stochastic(self) -> bool:
        """
        Return whether the strategy is stochastic.
        """
    def type(self) -> str:
        """
        Return the strategy type.
        """
class Defector(egttools.numerical.numerical_.behaviors.NormalForm.AbstractNFGStrategy):
    """
    Strategy that always defects.
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
        """
    def is_stochastic(self) -> bool:
        """
        Return whether the strategy is stochastic.
        """
    def type(self) -> str:
        """
        Return the strategy type.
        """
class GRIM(egttools.numerical.numerical_.behaviors.NormalForm.AbstractNFGStrategy):
    """
    Grim Trigger: cooperate until the opponent defects once, then defect forever.
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
        """
    def is_stochastic(self) -> bool:
        """
        Return whether the strategy is stochastic.
        """
    def type(self) -> str:
        """
        Return the strategy type.
        """
class GenerousTFT(egttools.numerical.numerical_.behaviors.NormalForm.AbstractNFGStrategy):
    """
    Generous Tit for Tat.
    """
    def __init__(self, R: float, P: float, T: float, S: float) -> None:
        """
                        Construct a Generous Tit for Tat strategy.
        
                        Following a defection, the strategy cooperates with probability
        
                        p(R, P, T, S) = min(1 - (T - R) / (R - S), (R - P) / (T - P))
        
                        where R, P, T, and S are the reward, punishment, temptation, and sucker's payoff.
        
                        Parameters
                        ----------
                        R : float
                            Reward payoff.
                        P : float
                            Punishment payoff.
                        T : float
                            Temptation payoff.
                        S : float
                            Sucker's payoff.
        """
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
        """
    def is_stochastic(self) -> bool:
        """
        Return whether the strategy is stochastic.
        """
    def type(self) -> str:
        """
        Return the strategy type.
        """
class GradualTFT(egttools.numerical.numerical_.behaviors.NormalForm.AbstractNFGStrategy):
    """
    Gradual Tit for Tat.
    """
    def __init__(self) -> None:
        """
                        Tit for Tat with two modifications:
        
                        1. The number of punishing defections increases with each additional defection by the opponent.
                        2. After each punishment phase, the strategy apologizes by cooperating in the following two rounds.
        """
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
        """
    def is_stochastic(self) -> bool:
        """
        Return whether the strategy is stochastic.
        """
    def type(self) -> str:
        """
        Return the strategy type.
        """
class ImperfectTFT(egttools.numerical.numerical_.behaviors.NormalForm.AbstractNFGStrategy):
    """
    Tit for Tat with implementation errors.
    """
    def __init__(self, error_probability: float) -> None:
        """
                        Construct an Imperfect Tit for Tat strategy.
        
                        Parameters
                        ----------
                        error_probability : float
                            Probability of choosing the opposite action by mistake.
        """
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
        """
    def is_stochastic(self) -> bool:
        """
        Return whether the strategy is stochastic.
        """
    def type(self) -> str:
        """
        Return the strategy type.
        """
class Pavlov(egttools.numerical.numerical_.behaviors.NormalForm.AbstractNFGStrategy):
    """
    Win-stay, lose-shift strategy.
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
        """
    def is_stochastic(self) -> bool:
        """
        Return whether the strategy is stochastic.
        """
    def type(self) -> str:
        """
        Return the strategy type.
        """
class Random(egttools.numerical.numerical_.behaviors.NormalForm.AbstractNFGStrategy):
    """
    Strategy that cooperates with probability 0.5.
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
        """
    def is_stochastic(self) -> bool:
        """
        Return whether the strategy is stochastic.
        """
    def type(self) -> str:
        """
        Return the strategy type.
        """
class SuspiciousTFT(egttools.numerical.numerical_.behaviors.NormalForm.AbstractNFGStrategy):
    """
    Suspicious Tit for Tat: defect in the first round, then copy the opponent's previous action.
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
        """
    def is_stochastic(self) -> bool:
        """
        Return whether the strategy is stochastic.
        """
    def type(self) -> str:
        """
        Return the strategy type.
        """
class TFT(egttools.numerical.numerical_.behaviors.NormalForm.AbstractNFGStrategy):
    """
    Tit for Tat: cooperate in the first round, then copy the opponent's previous action.
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
        """
    def is_stochastic(self) -> bool:
        """
        Return whether the strategy is stochastic.
        """
    def type(self) -> str:
        """
        Return the strategy type.
        """
class TFTT(egttools.numerical.numerical_.behaviors.NormalForm.AbstractNFGStrategy):
    """
    Tit for two tats: defect only after two consecutive defections.
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
        """
    def is_stochastic(self) -> bool:
        """
        Return whether the strategy is stochastic.
        """
    def type(self) -> str:
        """
        Return the strategy type.
        """
class TTFT(egttools.numerical.numerical_.behaviors.NormalForm.AbstractNFGStrategy):
    """
    Two tits for tat: defect twice after a defection.
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
        """
    def is_stochastic(self) -> bool:
        """
        Return whether the strategy is stochastic.
        """
    def type(self) -> str:
        """
        Return the strategy type.
        """

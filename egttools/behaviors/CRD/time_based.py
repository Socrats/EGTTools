from typing import List

from . import AbstractCRDStrategy


class TimeBasedCRDStrategy(AbstractCRDStrategy):
    def __init__(self, actions_per_round: List[int]):
        """
        A CRD strategy which adapts in function of a moving average of the contributions of the rest of the group.

        Parameters
        ----------
        actions_per_round: List[int]
            Defines the action which this strategy will play in every round. If you change the number of rounds
            of the game, you should redefine this list, otherwise there will be an exception!!
        """
        super().__init__()
        self.actions_per_round = actions_per_round

    def get_action(self, time_step: int, group_contributions_prev: int):
        return self.actions_per_round[time_step]

    def type(self):
        return "TimeBasedCRDStrategy({})".format(self.actions_per_round)

    def __str__(self):
        return self.type()

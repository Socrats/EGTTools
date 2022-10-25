# Copyright (c) 2019-2021  Elias Fernandez
#
# This file is part of EGTtools.
#
# EGTtools is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# EGTtools is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with EGTtools.  If not, see <http://www.gnu.org/licenses/>

from typing import Dict, Tuple, Union

from .. import AbstractNFGStrategy
from egttools import Random

import numpy as np

__all__ = ['EpsilonTFT', 'EpsilonGRIM', 'Detective', 'MemoryOneStrategy']


class EpsilonTFT(AbstractNFGStrategy):
    def __init__(self, p: float, epsilon: float):
        """
        A TFT player with randomized first action and probability
        of making mistakes.

        This player acts exactly as Tit-for-Tat (repeats
        the last action of the opponent), however in the
        first round it will cooperate with probability :param p
        and in the subsequent rounds it has a probability
        :param epsilon of making a mistake and changing its action.

        Parameters
        ----------
        p : float
            Probability of cooperating in the first round
        epsilon : float
            Probability of making a mistake in any round after round 1.
        """
        super().__init__()
        self.p_ = p
        self.epsilon_ = epsilon
        self.random_device = np.random.default_rng(Random.generate())
        self.is_stochastic_strategy = True

    def get_action(self, time_step: int, action_prev: int):
        if time_step == 0:
            return 0 if self.random_device.random() < self.p_ else 1
        else:
            return action_prev if self.random_device.random() >= self.epsilon_ else (action_prev + 1) % 2

    def type(self):
        return "NFGStrategies::EpsilonTFT"

    def is_stochastic(self):
        return self.is_stochastic_strategy

    def __str__(self):
        return self.type()


class EpsilonGRIM(AbstractNFGStrategy):
    def __init__(self, p: float, epsilon: float):
        """
        A GRIM player with randomized first action and probability
        of making mistakes.

        This player acts exactly as GRIM (cooperates until the opponent defects),
        however in the first round it will cooperate with probability :param p
        and in the subsequent rounds it has a probability
        :param epsilon of making a mistake and changing its action.

        Parameters
        ----------
        p : float
            Probability of cooperating in the first round
        epsilon : float
            Probability of making a mistake in any round after round 1.
        """
        super().__init__()
        self.p_ = p
        self.epsilon_ = epsilon
        self.random_device = np.random.default_rng(Random.generate())
        self.action_ = 1
        self.is_stochastic_strategy = True

    def get_action(self, time_step: int, action_prev: int):
        if time_step == 0:
            self.action_ = 1  # reset action to cooperation
            return 0 if self.random_device.random() < self.p_ else 1

        if self.action_ == 1:
            if action_prev == 0:
                self.action_ = 0

        action = self.action_

        if self.random_device.random() < self.epsilon_:
            # In this case we reset the trigger
            self.action_ = 1
            # And the real action in this round is the opposite of the original intention
            action = (action + 1) % 2

        return action

    def type(self):
        return "NFGStrategies::EpsilonGRIM"

    def is_stochastic(self):
        return self.is_stochastic_strategy

    def __str__(self):
        return self.type()


class Detective(AbstractNFGStrategy):
    def __init__(self):
        """
        A Detective player who tries to analyze the opponent.

        This player will always play the same initial sequence of
        Cooperate, Defect, Cooperate, Cooperate. If the opponent defects
        during this initial sequence, then Defective will play TFT from the 5th
        round on. Otherwise, Detective will play always Defect.
        """
        super().__init__()
        self.initial_sequence_ = [1, 0, 1, 1]
        self.cheated_ = False
        self.is_stochastic_strategy = False

    def get_action(self, time_step: int, action_prev: int):
        if time_step == 0:  # reset internal variables
            self.cheated_ = False
        if time_step < 4:
            if time_step > 0 and action_prev == 0:
                self.cheated_ = True
            return self.initial_sequence_[time_step]
        else:
            if self.cheated_:
                return action_prev
            else:
                return 0

    def type(self):
        return "NFGStrategies::Detective"

    def is_stochastic(self):
        return self.is_stochastic_strategy

    def __str__(self):
        return self.type()


class MemoryOneStrategy(AbstractNFGStrategy):
    def __init__(self, action_first_round: Union[int, float],
                 strategy: Union[Dict[Tuple[int, int], int], Dict[Tuple[int, int], float]], is_stochastic: bool):
        """
        Defines a Memory One strategy.

        Parameters
        ----------
        action_first_round: Union[int, float]
            Indicates the action this strategy will play in the first round. In the case that `is_stochastic`
            is True, then this value should be a probability of Cooperation
        strategy: Union[Dict[Tuple[int, int], int], Dict[Tuple[int, int], float]]
            A dictionary with tuples defining the action/probability of cooperation for
            each pair of previous actions of self and the opponent, e.g., CC, DC....
        is_stochastic: bool
            Indicates whether the strategy is stochastic or not. If it is stochastic, then the values both
            for the `action_first_round` and the strategy should be probabilities of Cooperation. If it is
            False, then 1 - indicates Cooperation and 0 - indicates Defection.
        """
        super().__init__()
        self.action_first_round_ = action_first_round
        self.strategy_ = strategy
        self.action_prev_self_ = action_first_round
        self.random_device = np.random.default_rng(Random.generate())
        self.is_stochastic_strategy = is_stochastic

    def get_action(self, time_step: int, action_prev: int):
        if time_step == 0:
            if not self.is_stochastic_strategy:
                self.action_prev_self_ = self.action_first_round_
            else:
                self.action_prev_self_ = 1 if self.random_device.random() < self.strategy_[
                    (self.action_prev_self_, action_prev)] else 0
            return self.action_first_round_
        else:
            if not self.is_stochastic_strategy:
                action = self.strategy_[(self.action_prev_self_, action_prev)]
            else:
                action = 1 if self.random_device.random() < self.strategy_[(self.action_prev_self_, action_prev)] else 0
            self.action_prev_self_ = action
            return action

    def type(self):
        return "NFGStrategies::MemoryOneStrategy"

    def is_stochastic(self):
        return self.is_stochastic_strategy

    def __str__(self):
        return self.type()

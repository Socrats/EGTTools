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

from .. import AbstractNFGStrategy
from egttools import Random

import numpy as np

__all__ = ['EpsilonTFT', 'EpsilonGRIM', 'Detective']


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
            return 0 if self.random_device.random() < self.p_ else 1
        elif self.action_ == 1:
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

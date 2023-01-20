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

from typing import List, Union
import numpy as np

from .. import (sample_simplex, )
from . import AbstractNPlayerGame
from ..behaviors.pgg_behaviors import PGGOneShotStrategy


class PGG(AbstractNPlayerGame):
    def __init__(self, group_size: int, cost: float, multiplying_factor: float,
                 strategies: List[PGGOneShotStrategy]) -> None:
        """
        Classical Public Goods game with only 2 possible contributions (o or cost).

        Parameters
        ----------
        group_size: int
            Size of the group playing the game.
        cost: float
            Cost of cooperation.
        multiplying_factor: float
            The sum of contributions to the public good is multiplied by this factor before being divided equally
            among all players.
        strategies: List[egttools.behaviors.pgg_behaviors.PGGOneShotStrategy]
            A list of strategies that will play the game.
        """
        AbstractNPlayerGame.__init__(self, len(strategies), group_size)
        self.group_size_ = group_size
        self.c_ = cost
        self.r_ = multiplying_factor
        self.strategies_ = strategies
        self.nb_strategies_ = len(strategies)

        self.nb_group_configurations_ = self.nb_group_configurations()
        self.calculate_payoffs()

    def play(self, group_composition: Union[List[int], np.ndarray], game_payoffs: np.ndarray) -> None:
        # Gather contributions
        contributions = 0.0
        non_zero = []
        for i, strategy_count in enumerate(group_composition):
            if strategy_count == 0:
                continue
            else:
                non_zero.append(i)
                action = self.strategies_[i].get_action()
                if action == 1:
                    contributions += strategy_count * self.c_
                    game_payoffs[i] = - self.c_

        benefit = (contributions * self.r_) / self.group_size_
        game_payoffs[non_zero] += benefit

    def calculate_payoffs(self) -> np.ndarray:
        payoffs_container = np.zeros(shape=(self.nb_strategies_,), dtype=np.float64)
        for i in range(self.nb_group_configurations_):
            # Get group composition
            group_composition = sample_simplex(i, self.group_size_, self.nb_strategies_)
            self.play(group_composition, payoffs_container)
            for strategy_index, strategy_payoff in enumerate(payoffs_container):
                self.update_payoff(strategy_index, i, strategy_payoff)
            # Reinitialize payoff vector
            payoffs_container[:] = 0

        return self.payoffs()

    def __str__(self) -> str:
        string = f'''
        Python implementation of a public goods game.\n
        Game parameters
        -------
        cost = {self.c_}\n
        multiplication_factor = {self.r_}\n
        Strategies
        -------
        nb_strategies = {self.nb_strategies_}\n
        strategies = {self.strategies_}\n
        '''
        return string

    def type(self) -> str:
        return "PGG"

    def save_payoffs(self, file_name: str) -> None:
        with open(file_name, 'w') as f:
            f.write('Payoffs for each type of player and each possible state:\n')
            f.write(f'rows: {" ,".join([strategy.type for strategy in self.strategies_])}\n')
            f.write('cols: all possible group compositions starting at (0, 0, ..., group_size)\n')
            f.write(f'{self.payoffs_}')
            f.write(f'group_size = {self.group_size_}\n')
            f.write(f'cost = {self.c_}\n')
            f.write(f'multiplying_factor = {self.r_}\n')

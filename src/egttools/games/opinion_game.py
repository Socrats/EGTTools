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


class OpinionGame(AbstractNPlayerGame):
    def __init__(self, group_size: int, peer_pressure_importance: float, peer_pressure_ratio: float,
                 opinion_values: List[float]) -> None:
        """
        Classical Public Goods game with only 2 possible contributions (o or cost).

        Parameters
        ----------
        group_size:
            Size of the group playing the game.
        peer_pressure_importance:
            Importance of being close in opinion to your group
        peer_pressure_ratio:
            Ratio at which the peer pressure is more important
        opinion_values:
            Value of each opinion
        """
        AbstractNPlayerGame.__init__(self, len(opinion_values), group_size)
        self.group_size_ = group_size
        self.peer_pressure_importance_ = peer_pressure_importance
        self.peer_pressure_ratio_ = peer_pressure_ratio
        self.opinion_values_ = opinion_values
        self.nb_strategies_ = len(opinion_values)
        self.strategies_ = np.arange(self.nb_strategies_)

        self.nb_group_configurations_ = self.nb_group_configurations()

        self.calculate_payoffs()

    def play(self, group_composition: Union[List[int], np.ndarray], game_payoffs: np.ndarray) -> None:
        # Each player's payoff is a function of the value of their opinion
        # and the distribution of opinions in the group (in function of the peer pressure importance)
        for i, strategy_count in enumerate(group_composition):
            if strategy_count == 0:
                game_payoffs[i] = 0
            else:
                group_composition[i] -= 1
                mean = np.average(self.strategies_, weights=group_composition)
                var = moment(self.strategies_, group_composition, mean, 2)
                group_composition[i] += 1
                # calculate distance of opinion to the group
                distance = np.abs(self.strategies_[i] - mean)
                game_payoffs[i] = (1 - self.peer_pressure_ratio_) * self.opinion_values_[
                    i] + self.peer_pressure_ratio_ * self.opinion_values_[i] * (
                                          1 - sigmoid(distance * (1 - var), self.peer_pressure_importance_))

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
        return "OpinionGame"

    def save_payoffs(self, file_name: str) -> None:
        with open(file_name, 'w') as f:
            f.write('Payoffs for each type of player and each possible state:\n')
            f.write(f'rows: {" ,".join([strategy.type() for strategy in self.strategies_])}\n')
            f.write('cols: all possible group compositions starting at (0, 0, ..., group_size)\n')
            f.write(f'{self.payoffs_}')
            f.write(f'group_size = {self.group_size_}\n')
            f.write(f'peer_pressure_importance = {self.peer_pressure_importance_}\n')
            f.write(f'opinion_values = {self.opinion_values_}\n')


def moment(x, counts, c, n):
    return np.sum(counts * (x - c) ** n) / np.sum(counts)


def sigmoid(x, temperature):
    return 1. / (1. + np.exp(-temperature * (x - 1)))

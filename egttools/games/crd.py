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

# from typing import List, Union
# import numpy as np
# from scipy.stats import multivariate_hypergeom
#
# from egttools import calculate_nb_states, calculate_state, sample_simplex
# from . import AbstractGame


# class OneShotCRD(AbstractGame):
#     def __init__(self, endowment: float, cost: float, risk: float, group_size: int, min_nb_cooperators: int) -> None:
#         """
#         This class implements a One-Shot Collective Risk Dilemma.
#
#         This N-player game was first introduced in "Santos, F. C., & Pacheco, J. M. (2011).
#         Risk of collective failure provides an escape from the tragedy of the commons.
#         Proceedings of the National Academy of Sciences of the United States of America, 108(26), 10421–10425.".
#
#         The game consists of a group of size ``group_size`` (N) which can be composed of
#         Cooperators (Cs) who will contribute a fraction ``cost`` (c) of their
#         :param endowment (b) to the public good. And of Defectors (Ds) who contribute 0.
#
#         If the total contribution of the group is equal or surpasses the collective target Mcb,
#         with M being the ``min_nb_cooperators``, then all participants will receive as payoff
#         their remaining endowment. Which is, Cs receive b - cb and Ds receive b. Otherwise, all
#         participants receive 0 endowment with a probability equal to ``risk`` (r), and will
#         keep their endowment with probability 1-r. This means that each group must have at least
#         M Cs for the collective target to be achieved.
#
#         Parameters
#         ----------
#         endowment : float
#             The initial endowment (b) received by all participants
#         cost : float
#             The fraction of the endowment that Cooperators contribute to the public good.
#             This value must be in the interval [0, 1]
#         risk : float
#             The risk that all members of the group will lose their remaining endowment if the
#             collective target is not achieved.
#         group_size : int
#             The size of the group (N)
#         min_nb_cooperators : int
#             The minimum number of cooperators (M) required to reach the collective target.
#             In other words, the collective target is reached if the collective effort is
#             at least Mcb. This value must be in the discrete interval [[0, N]].
#
#         See Also
#         --------
#         egttools.games.CRDGame, egttools.games.CRDGameTU
#         """
#         AbstractGame.__init__(self)
#         self.endowment_ = endowment
#         assert 0 <= cost <= 1
#         self.cost_ = cost
#         self.risk_ = risk
#         self.one_minus_risk_ = 1 - risk
#         self.endowment_times_one_minus_risk_ = endowment * (1 - risk)
#         self.endowment_minus_cost_ = endowment * (1 - cost)
#         self.endowment_minus_cost_times_one_minus_risk_ = self.endowment_times_one_minus_risk_ - endowment*cost
#         self.group_size_ = group_size
#         assert 0 <= min_nb_cooperators <= group_size
#         self.min_nb_cooperators_ = min_nb_cooperators
#         self.nb_strategies_ = 2  # for now we only allow binary contributions so we only have Cs and Ds
#         self.nb_group_compositions_ = calculate_nb_states(self.group_size_, self.nb_strategies_)
#         self.payoffs_ = np.zeros(shape=(self.nb_strategies_, self.nb_group_compositions_), dtype=np.float64)
#         self.calculate_payoffs()
#
#     def play(self, group_composition: Union[List[int], np.ndarray], game_payoffs: np.ndarray) -> None:
#         """
#         Plays the One-shop CRD and update the game_payoffs given the group_composition.
#
#         We always assume that strategy 0 is D and strategy 1 is C.
#
#         The payoffs of Defectors and Cooperators are described by the following equations:
#
#         .. math::
#             \\Pi_{D}(k) = b{\\theta(k-M)+ (1-r)[1 - \\theta(k-M)]}
#
#             \\Pi_{C}(k) = \\Pi_{D}(k) - cb
#
#             \\text{where } \\theta(x) = 0 \\text{if } x < 0 \\text{ and 1 otherwise.}
#
#         Parameters
#         ----------
#         group_composition : Union[List[int], numpy.ndarray]
#             A list or array containing the counts of how many members of each strategy are
#             present in the group.
#         game_payoffs: numpy.ndarray
#             A vector in which the payoffs of the game will be stored.
#         """
#         if group_composition[1] < self.min_nb_cooperators_:
#             game_payoffs[0] = self.endowment_times_one_minus_risk_
#             game_payoffs[1] = self.endowment_minus_cost_times_one_minus_risk_
#         else:
#             game_payoffs[0] = self.endowment_
#             game_payoffs[1] = self.endowment_minus_cost_
#
#     def calculate_payoffs(self) -> np.ndarray:
#         payoffs_container = np.zeros(shape=(self.nb_strategies_,), dtype=np.float64)
#         for i in range(self.nb_group_compositions_):
#             # Get group composition
#             group_composition = sample_simplex(i, self.group_size_, self.nb_strategies_)
#             self.play(group_composition, payoffs_container)
#             for strategy_index, strategy_payoff in enumerate(payoffs_container):
#                 self.payoffs_[strategy_index, i] = strategy_payoff
#
#         return self.payoffs_
#
#     def calculate_fitness(self, player_type: int, pop_size: int, population_state: np.ndarray) -> float:
#         """
#         Calculates the fitness of a strategy given a population state.
#
#         Parameters
#         ----------
#         player_type : int
#             The index of the strategy whose fitness will be calculated.
#         pop_size : int
#             The size of the population (Z).
#         population_state : numpy.ndarray
#             A vector containing the counts of each strategy in the population.
#
#         Returns
#         -------
#         float
#             The fitness of the strategy in the current population state.
#         """
#         state = population_state.copy()
#         # multivariate PDF
#         state[player_type] -= 1
#         rv = multivariate_hypergeom(state, self.group_size_ - 1)
#         state[player_type] += 1
#
#         fitness = 0.0
#
#         # Iterate over all possible group compositions
#         for i in range(self.nb_group_compositions_):
#             group_composition = sample_simplex(i, self.group_size_ - 1, self.nb_strategies_)
#             # Estimate probability of the current group composition
#             if group_composition[player_type] > 0:
#                 group_composition[player_type] -= 1
#                 fitness += self.payoffs_[player_type, i] * rv.pmf(x=group_composition)
#
#         return fitness
#
#     def __str__(self) -> str:
#         string = f'''
#         Python implementation of a public goods one-shot Collective Risk Dilemma.\n
#         Game parameters
#         -------
#         b = {self.endowment_}\n
#         c = {self.cost_}\n
#         r = {self.risk_}\n
#         N = {self.group_size_}\n
#         M = {self.min_nb_cooperators_}\n
#         Strategies
#         -------
#         Currently the only strategies are Cooperators and Defectors.
#         '''
#         return string
#
#     def nb_strategies(self):
#         return self.nb_strategies_
#
#     def type(self):
#         return "egttools::games::OneShotCRD"
#
#     def payoffs(self):
#         return self.payoffs_
#
#     def payoff(self, strategy, group_composition, p_int=None):
#         if strategy > self.nb_strategies_:
#             raise IndexError(f'You must specify a valid index for the strategy [0, {self.nb_strategies_}].')
#         elif len(group_composition) != self.nb_strategies_:
#             raise Exception(f'The group composition list must be of size {self.nb_strategies_}')
#
#         return self.payoffs_[strategy, calculate_state(self.group_size_, group_composition)]
#
#     def save_payoffs(self, file_name: str) -> None:
#         with open(file_name, 'w') as f:
#             f.write('Payoffs for each type of player and each possible state:\n')
#             f.write(f'rows: {" D, C"}\n')
#             f.write('cols: all possible group compositions starting at (0, 0, ..., group_size)\n')
#             f.write(f'{self.payoffs_}')
#             f.write(f'group_size = {self.group_size_}\n')
#             f.write(f'endowment = {self.endowment_}\n')
#             f.write(f'cost = {self.cost_}\n')
#             f.write(f'risk = {self.risk_}\n')
#             f.write(f'group_size = {self.group_size_}\n')
#             f.write(f'min_nb_cooperators = {self.min_nb_cooperators_}\n')

from typing import Union, List

import numpy as np

from .abstract_games import AbstractNPlayerGame

from egttools import sample_simplex


class NPlayerStagHunt(AbstractNPlayerGame):
    """
    This game is based on the article
    Pacheco et al., ‘Evolutionary Dynamics of Collective Action in N -Person Stag Hunt Dilemmas’.
    """

    def __init__(self, group_size, enhancement_factor, cooperation_threshold, cost):
        self.group_size_ = group_size  # N
        self.enhancement_factor_ = enhancement_factor  # F
        self.cooperation_threshold_ = cooperation_threshold  # M
        self.cost_ = cost  # c
        self.strategies = ['Defect', 'Cooperate']

        self.nb_strategies_ = 2
        super().__init__(self.nb_strategies_, self.group_size_)

    def play(self, group_composition: Union[List[int], np.ndarray], game_payoffs: np.ndarray) -> None:
        """
        Calculate the payoff of each strategy inside the group.

        $\\Pi_{D}(k) = (kFc)\theta(k-M)$
        $\\Pi_{C}(k) = \\Pi_{D}(k) - c$

        Parameters
        ----------
        group_composition: Union[List[int], numpy.ndarray]
            counts of each strategy inside the group.
        game_payoffs: numpy.ndarray
            container for the payoffs of each strategy

        """
        if group_composition[0] == 0:
            game_payoffs[0] = 0
            game_payoffs[1] = self.cost_ * (self.enhancement_factor_ - 1)
        elif group_composition[1] == 0:
            game_payoffs[0] = 0
            game_payoffs[1] = 0
        else:
            game_payoffs[0] = ((group_composition[1]
                                * self.enhancement_factor_)
                               / self.group_size_) if group_composition[
                                                          1] >= self.cooperation_threshold_ else 0  # Defectors
            game_payoffs[1] = game_payoffs[0] - self.cost_  # Cooperators

    def calculate_payoffs(self) -> np.ndarray:
        payoffs_container = np.zeros(shape=(self.nb_strategies_,), dtype=np.float64)
        for i in range(self.nb_group_configurations_):
            # Get group composition
            group_composition = sample_simplex(i, self.group_size_, self.nb_strategies_)
            self.play(group_composition, payoffs_container)
            for strategy_index, strategy_payoff in enumerate(payoffs_container):
                self.payoffs_[strategy_index, i] = strategy_payoff
            # Reinitialize payoff vector
            payoffs_container[:] = 0

        return self.payoffs_

    def __str__(self) -> str:
        string = f'''
        Python implementation the N-player Stag Hunt Game.\n
        Game parameters
        -------
        N = {self.group_size_}\n
        F = {self.enhancement_factor_}\n
        M = {self.cooperation_threshold_}\n
        cost = {self.cost_}\n
        '''
        return string

    def type(self) -> str:
        return "NPlayerStagHunt"

    def save_payoffs(self, file_name: str) -> None:
        with open(file_name, 'w') as f:
            f.write('Payoffs for each type of player and each possible state:\n')
            f.write(f'rows: {" ,".join([strategy for strategy in self.strategies_])}\n')
            f.write('cols: all possible group compositions starting at (0, 0, ..., group_size)\n')
            f.write(f'{self.payoffs_}')
            f.write(f'N = {self.group_size_}\n')
            f.write(f'F = {self.enhancement_factor_}\n')
            f.write(f'M = {self.cooperation_threshold_}\n')
            f.write(f'cost = {self.c_}\n')


class CommonPoolResourceDilemma(AbstractNPlayerGame):
    def __init__(self, group_size: int, a: float, b: float, min_e: float, cost_, strategies: List[int] = None):
        self.group_size_ = group_size  # N
        self.a_ = a
        self.b_ = b
        self.min_e = min_e
        self.cost_ = cost_
        self.strategies_ = strategies
        if strategies is None:
            self.nb_strategies_ = 3
        else:
            self.nb_strategies_ = len(strategies)

        super().__init__(self.nb_strategies_, self.group_size_)

    def calculate_total_extraction(self, group_composition):
        extraction = 0

        for i, nb_players_with_strategy_i in enumerate(group_composition):
            extraction += self.extraction_strategy(i) * nb_players_with_strategy_i
        return extraction

    def extraction(self, x, x_total):
        cost = 0
        if x <= self.min_e:
            cost = self.cost_
        return ((x / x_total) * ((self.a_ * x_total) - (self.b_ * x_total * x_total))) - cost

    def play(self, group_composition: Union[List[int], np.ndarray], game_payoffs: np.ndarray) -> None:
        # Each player's payoff is a function of the value of theirs and other extractiers' extraction
        x_total = self.calculate_total_extraction(group_composition)

        for i, strategy_count in enumerate(group_composition):
            if strategy_count == 0:
                game_payoffs[i] = 0
            else:
                x = self.extraction_strategy(i)
                game_payoffs[i] = self.extraction(x, x_total)  # UPDATED: ONLY MARKET 2

    def calculate_payoffs(self) -> np.ndarray:
        payoffs_container = np.zeros(shape=(self.nb_strategies_,), dtype=np.float64)
        for i in range(self.nb_group_configurations_):
            # Get group composition
            group_composition = sample_simplex(i, self.group_size_, self.nb_strategies_)
            self.play(group_composition, payoffs_container)
            for strategy_index, strategy_payoff in enumerate(payoffs_container):
                self.payoffs_[strategy_index, i] = strategy_payoff
            # Reinitialize payoff vector
            payoffs_container[:] = 0

        return self.payoffs_

    def extraction_strategy(self, strategy_id):
        if self.strategies_ is not None:
            return self.strategies_[strategy_id]
        else:
            extraction = 0
            if strategy_id == 0:
                extraction = self.min_extraction()
            elif strategy_id == 1:
                extraction = self.low_extraction()
            elif strategy_id == 2:
                extraction = self.high_extraction()
            return extraction

    def high_extraction(self):
        return self.a_ / (self.b_ * self.group_size_)

    def low_extraction(self):
        group_optimal = self.a_ / (2 * self.b_)  # CHECK
        return group_optimal / self.group_size_

    def min_extraction(self):
        return self.min_e

    def conditional_low(self, x_total):
        x = 0
        if x_total / self.resource_state_ > self.belief_low_:
            x = self.low_extraction()
        else:
            x = self.high_extraction()
        return x

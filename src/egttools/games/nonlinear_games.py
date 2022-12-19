from typing import Union, List

import numpy as np

from . import AbstractNPlayerGame
from .. import sample_simplex, calculate_state
from ..behaviors.CPR.abstract_cpr_strategy import AbstractCPRStrategy

from scipy.stats import multivariate_hypergeom


class NPlayerStagHunt(AbstractNPlayerGame):
    """
    This game is based on the article
    Pacheco et al., ‘Evolutionary Dynamics of Collective Action in N -Person Stag Hunt Dilemmas’.
    """

    def __init__(self, group_size, enhancement_factor, cooperation_threshold, cost):
        super().__init__(2, group_size)  # Must be initialized at the beginning!
        self.group_size_ = group_size  # N
        self.enhancement_factor_ = enhancement_factor  # F
        self.cooperation_threshold_ = cooperation_threshold  # M
        self.cost_ = cost  # c
        self.strategies = ['Defect', 'Cooperate']

        self.nb_strategies_ = 2
        self.nb_group_configurations_ = self.nb_group_configurations()

        self.calculate_payoffs()

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
                self.update_payoff(strategy_index, i, strategy_payoff)
            # Reinitialize payoff vector
            payoffs_container[:] = 0

        return self.payoffs()

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
            f.write(f'{self.payoffs()}')
            f.write(f'N = {self.group_size_}\n')
            f.write(f'F = {self.enhancement_factor_}\n')
            f.write(f'M = {self.cooperation_threshold_}\n')
            f.write(f'cost = {self.c_}\n')


class CommonPoolResourceDilemma(AbstractNPlayerGame):
    def __init__(self, group_size: int, a: float, b: float, min_e: float, cost: float, nb_strategies: int = 3,
                 strategies: List[int] = None):
        AbstractNPlayerGame.__init__(self, nb_strategies, group_size)  # Must be initialized before anything else!

        self.group_size_ = group_size  # N
        self.a_ = a
        self.b_ = b
        self.min_e = min_e
        self.cost_ = cost
        self.strategies_ = strategies
        self.nb_strategies_ = nb_strategies

        self.nb_group_configurations_ = self.nb_group_configurations()

        self.calculate_payoffs()

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
                self.update_payoff(strategy_index, i, strategy_payoff)
            # Reinitialize payoff vector
            payoffs_container[:] = 0

        return self.payoffs()

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


class CommonPoolResourceDilemmaCommitment(AbstractNPlayerGame):
    def __init__(self, group_size: int, a: float, b: float, cost: float, fine: float,
                 strategies: List[AbstractCPRStrategy]):
        """
        Common Pool resource game with commitment, but the threshold is set as another strategy
        Parameters
        ----------
        group_size: int
            Size of the group playing the game.
        a and b: flaot
            Both a and b are the parameters of the game, if a>>b, then it's attractive to invest in the CPR
        cost:
            cost of making the commitment for commitment proposing strategy
        fine: float
            fine for defection when accepting the commitment
        strategies: Lit[int]
            List of strategies which will play the game
        """
        AbstractNPlayerGame.__init__(self, len(strategies), group_size)  # Must be initialized before anything else!
        self.group_size_ = group_size

        self.a_ = a
        self.b_ = b
        self.fine_ = fine
        self.cost_ = cost
        self.strategies_ = strategies
        self.nb_strategies_ = len(strategies)

        self.nb_group_configurations_ = self.nb_group_configurations()
        self.extractions_ = np.zeros(self.nb_group_configurations_, dtype=np.float64)

        self.calculate_payoffs()

    def play(self, group_composition: Union[List[int], np.ndarray], game_payoffs: np.ndarray) -> None:
        # Each player's payoff is a function of the value of theirs and other extractiers' extraction
        commitment_accepted = np.zeros(shape=(self.nb_strategies_,), dtype=bool)
        nb_committed = self.get_nb_committed(group_composition)
        self.check_if_commitment_validated(nb_committed, group_composition, commitment_accepted)
        group_extraction = self.calculate_total_extraction(commitment_accepted, group_composition)

        for index, strategy_count in enumerate(group_composition):
            if strategy_count == 0:
                game_payoffs[index] = 0
            else:
                extraction = self.strategies_[index].get_extraction(self.a_, self.b_, self.group_size_,
                                                                    commitment_accepted[index])
                game_payoffs[index] = self.strategies_[index].get_payoff(self.a_, self.b_, extraction, group_extraction,
                                                                         self.fine_, self.cost_,
                                                                         commitment_accepted[index])
        self.add_group_extraction(group_extraction, group_composition)

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

    def calculate_total_extraction(self, commitment_accepted: np.ndarray,
                                   group_composition: Union[List[int], np.ndarray]) -> float:
        extraction = 0
        for index, strategy_count in enumerate(group_composition):
            if strategy_count > 0:
                extraction += strategy_count * self.strategies_[index].get_extraction(self.a_, self.b_,
                                                                                      self.group_size_,
                                                                                      commitment_accepted[index])
        return extraction

    def get_nb_committed(self, group_composition: Union[List[int], np.ndarray]) -> int:
        nb_committed = 0
        for index, strategy_count in enumerate(group_composition):
            nb_committed += strategy_count * self.strategies_[index].would_like_to_commit()
        return nb_committed

    def check_if_commitment_validated(self, nb_committed: int, group_composition: Union[List[int]],
                                      commitment_accepted: np.ndarray) -> None:
        validated = False
        # A commitment will only be validated if a commitment proposing strategy is present in the group
        for index, strategy_count in enumerate(group_composition):
            if strategy_count > 0:
                if self.strategies_[index].proposes_commitment():
                    commitment_accepted[index] = self.strategies_[index].is_commitment_validated(nb_committed)
                    validated = commitment_accepted[index]
        # If there is a commitment proposing strategy in the group for which the commitment would be validated
        # Then we check for every other strategy if they would be part of the commitment
        if validated:
            for index, strategy_count in enumerate(group_composition):
                if strategy_count > 0:
                    if not self.strategies_[index].proposes_commitment():
                        commitment_accepted[index] = self.strategies_[index].is_commitment_validated(validated)

    def add_group_extraction(self, group_extraction, group_composition):
        self.extractions_[
            calculate_state(group_size=self.group_size_, group_composition=group_composition)] = group_extraction

    def calculate_expected_consumption(self, population_state: np.ndarray) -> float:

        # multivariate PDF
        rv = multivariate_hypergeom(population_state, self.group_size_)

        expected_extraction = 0.0
        # Iterate over all possible group compositions
        for i in range(self.nb_group_configurations_):
            group_composition = sample_simplex(i, self.group_size_, self.nb_strategies_)
            # Estimate probability of the current group composition
            expected_extraction += self.extractions_[i] * rv.pmf(x=group_composition)

        return expected_extraction

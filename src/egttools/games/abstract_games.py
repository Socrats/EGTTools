import numpy as np

from typing import Union, List
from abc import abstractmethod
from . import AbstractGame, AbstractNPlayerGame
from .. import calculate_state, sample_simplex


class AbstractNPlayerGameExpectedPayoff(AbstractNPlayerGame):
    """
    This abstract Game class can be used in most scenarios where the fitness of a strategy is calculated as its
    expected payoff given the population state.

    It assumes that the game is N player, since the fitness of a strategy given a population state is calculated
    as the expected payoff of that strategy over all possible group combinations in the given state.

    Notes
    -----
    It might be a good idea to overwrite the methods `__str__`, `type`, and `save_payoffs` to adapt to your
    given game implementation

    It assumes that you have at least the following attributes:
     1. And an attribute `self.nb_strategies_` which contains the number of strategies
     that you are going to analyse for the given game.
     2. `self.payoffs()` returns a numpy.ndarray and contain the payoff matrix of the game. This array
     is of shape (self.nb_strategies(), self.nb_group_configurations()), where self.nb_group_configurations()
     is the number
     of possible combinations of strategies in the group. Thus, each row should give the (expected) payoff of the row
     strategy when playing in a group with the column configuration. The `payoff` method provides an easy way to access
     the payoffs for any group composition, by taking as arguments the index of the row strategy
     and a List with the count of each possible strategy in the group.

     You must still implement the methods `play` which should define how the game assigns
     payoffs to each strategy for a given game context. In particular, `calculate_payoffs` should fill the
     array `self.payoffs_` with the correct values as explained above. We recommend that you run this method in
     the `__init__` (initialization of the object) since, these values must be set before passing the game object
     to the numerical simulator (e.g., egttools.numerical.PairwiseComparisonNumerical).

    """

    @abstractmethod
    def play(self, group_composition: Union[List[int], np.ndarray], game_payoffs: np.ndarray) -> None:
        """
        This method fills the `game_payoffs` container with the payoff of each strategy given the `group_composition`.

        Strategies not present in the group will receive 0 payoff by default.

        Parameters
        ----------
        group_composition: Union[List[int], numpy.ndarray]
            A List or a numpy.ndarray containing the counts of each strategy in the group (e.g., for a game with 3
            possible strategies and group size 4, the following List is possible [3, 0, 1]).
        game_payoffs: numpy.ndarray
            A container for the payoffs that will be calculated. This avoids needing to create a new array at each
            call and should speed up computation.
        """
        pass

    def calculate_payoffs(self) -> np.ndarray:
        """
        This method calculates the payoffs for each strategy in each possible group configuration. Thus, it must
        fill the `self.payoffs_` numpy.ndarray with these payoffs values. This array
        must be of shape (self.nb_strategies_, nb_group_configurations), where nb_group_configurations is the number
        of possible combinations of strategies in the group. Thus, each row should give the (expected) payoff of the row
        strategy when playing in a group with the column configuration.

        Returns
        -------
        numpy.ndarray
            The payoff matrix of the game.

        """
        payoffs_container = np.zeros(shape=(self.nb_strategies(),), dtype=np.float64)
        for i in range(self.nb_group_configurations()):
            # Get group composition
            group_composition = sample_simplex(i, self.group_size(), self.nb_strategies())
            self.play(group_composition, payoffs_container)
            for strategy_index, strategy_payoff in enumerate(payoffs_container):
                self.update_payoff(strategy_index, i, strategy_payoff)
            # Reinitialize payoff vector
            payoffs_container[:] = 0

        return self.payoffs()


class AbstractTwoPLayerGame(AbstractGame):
    """
    This abstract Game class can be used in most scenarios where the fitness of a strategy is calculated as its
    expected payoff given the population state.

    It assumes that the game is 2 player and the fitness is calculated with this assumption!

    Notes
    -----
    It might be a good idea to overwrite the methods `__str__`, `type`, and `save_payoffs` to adapt to your
    given game implementation

    It assumes that you have at least the following attributes:
     1. And an attribute `self.nb_strategies_` which contains the number of strategies
     that you are going to analyse for the given game.
     2. `self.payoffs_` which must be a numpy.ndarray and contain the payoff matrix of the game. This array
     must be of shape (self.nb_strategies_, self.nb_strategies_).

    For normal form games:
    1. There is already a class called NormalFormGame available which you can use for these types of games. If
    for any reason this does not cover your needs then:
    2. If your game is normal form, but iterated, you should create another variable to contain the payoff matrix
    for one round of the game, since `self.payoffs_` will contain the expected payoffs over the several rounds
    of the game.
    3. If the game is one-shot and normal form, `self.payoffs_` is the payoff matrix of the game, and you do not
    need to do anything in calculate_payoffs besides calling this matrix.


     You must still implement the methods `play` and `calculate_payoffs` which should define how the game assigns
     payoffs to each strategy for each possible game context. In particular, `calculate_payoffs` should fill the
     array `self.payoffs_` with the correct values as explained above. We recommend that you run this method in
     the `__init__` (initialization of the object) since, these values must be set before passing the game object
     to the numerical simulator (e.g., egttools.numerical.PairwiseComparisonNumerical).

    """

    def __init__(self, nb_strategies: int):
        """
        This class must be initialized with the total number of strategies
        that will be used and the size of the group in which the game takes place.
        This is required to calculate the number of group configurations and the correct
        shape of the payoff matrix.

        Parameters
        ----------
        nb_strategies: int
            total number of possible strategies.
        """
        super().__init__()
        self.nb_strategies_ = nb_strategies
        self.payoffs_ = np.zeros(shape=(self.nb_strategies_, self.nb_strategies_))

        # initialize the payoffs matrix
        self.calculate_payoffs()

    @abstractmethod
    def play(self, group_composition: Union[List[int], np.ndarray], game_payoffs: np.ndarray) -> None:
        """
        This method fills the `game_payoffs` container with the payoff of each strategy given the `group_composition`.

        Strategies not present in the group will receive 0 payoff by default.

        Parameters
        ----------
        group_composition: Union[List[int], numpy.ndarray]
            A List or a numpy.ndarray containing the counts of each strategy in the group (e.g., for a game with 3
            possible strategies and group size 4, the following List is possible [3, 0, 1]).
        game_payoffs: numpy.ndarray
            A container for the payoffs that will be calculated. This avoids needing to create a new array at each
            call and should speed up computation.
        """
        pass

    @abstractmethod
    def calculate_payoffs(self) -> np.ndarray:
        """
        This method calculates the payoffs for each strategy in each possible group configuration. Thus, it must
        fill the `self.payoffs_` numpy.ndarray with these payoffs values. This array
        must be of shape (self.nb_strategies_, nb_group_configurations), where nb_group_configurations is the number
        of possible combinations of strategies in the group. Thus, each row should give the (expected) payoff of the row
        strategy when playing in a group with the column configuration.

        Returns
        -------
        numpy.ndarray
            The payoff matrix of the game.

        """
        pass

    def calculate_fitness(self, player_strategy: int, pop_size: int, population_state: np.ndarray) -> float:
        """
        Calculates the Fitness of a strategy for a given population state.

        The calculation is done by computing the expected payoff over all possible strategy matches.

        Parameters
        ----------
        player_strategy : int
            index of the strategy.
        pop_size : int
            size of the population - Only necessary for compatibility with the C++ implementation
            (might be eliminated in the future).
        population_state : numpy.ndarray[numpy.uint64[m, 1]]
            vector with the population state (the number of players adopting each strategy).

        Returns
        -------
        float
            The fitness of the population.
        """

        population_state[player_strategy] -= 1
        fitness = 0.0
        for i in range(self.nb_strategies_):
            fitness += (population_state[i] / (pop_size - 1)) * self.payoffs_[player_strategy, i]
        population_state[player_strategy] += 1

        return fitness

    def __str__(self) -> str:
        return "AbstractTwoPLayerGame"

    def nb_strategies(self) -> int:
        return self.nb_strategies_

    def type(self) -> str:
        return "AbstractTwoPLayerGame"

    def payoffs(self) -> np.ndarray:
        return self.payoffs_

    def payoff(self, strategy: int, group_composition: List[int]) -> float:
        if strategy > self.nb_strategies_:
            raise IndexError(f'You must specify a valid index for the strategy [0, {self.nb_strategies_}].')
        elif len(group_composition) != self.nb_strategies_:
            raise Exception(f'The group composition list must be of size {self.nb_strategies_}')

        return self.payoffs_[strategy, calculate_state(self.group_size_, group_composition)]

    def save_payoffs(self, file_name: str) -> None:
        with open(file_name, 'w') as f:
            f.write('Payoff matrix of the game:\n')
            f.write(f'{self.payoffs_}')

How to create a new game
========================

Implementing a new game from scratch: the `AbstractGame` class
--------------------------------------------------------------

The `Game` class is core to the EGTtools library, as it defines the
environment in which strategic interactions take place. All
`games` must extend the `AbstractGame` abstract class.
This class nine methods:

- `play`
- `calculate_payoffs`
- `calculate_fitness`
- `__str__`
- `nb_strategies`
- `type`
- `payoffs`
- `payoff`
- `save_payoffs`

Below you can see a description of this class in Python and the
purpose of each method:

.. code-block:: Python

    class AbstractGame:
        def play(self, group_composition: Union[List[int], numpy.ndarray], game_payoffs: numpy.ndarray) -> None:
            """
            Calculates the payoff of each strategy inside the group.

            Parameters
            ----------
            group_composition: Union[List[int], numpy.ndarray]
                counts of each strategy inside the group.
            game_payoffs: numpy.ndarray
                container for the payoffs of each strategy
            """
            pass

        def calculate_payoffs(self) -> np.ndarray:
        """
        This method should set a numpy.ndarray called self.payoffs_ with the
        expected payoff of each strategy in each possible
        state of the game
        """"
            pass

        def calculate_fitness(self, strategy_index: int, pop_size: int, population_state: numpy.ndarray) -> float:
        """
        This method should return the fitness of strategy
        with index `strategy_index` for the given `population_state`.
        """
            pass

        def __str__(self) -> str:
        """
        This method should return a string representation of the game.
        """
            pass

        def nb_strategies(self) -> int:
        """
        This method should return the number of strategies which can play the game.
        """
            pass

        def type(self) -> str:
        """
        This method should return a string representing the type of game.
        """
            pass

        def payoffs(self) -> np.ndarray:
        """
        This method should return the payoff matrix of the game,
        which gives the payoff of each strategy
        in each given context.
        """
            pass

        def payoff(self, strategy: int, group_configuration: List[int]) -> float:
        """
        This method should return the payoff of a strategy
        for a given `group_configuration`, which gives
        the counts of each strategy in the group.
        This method only needs to be implemented for N-player games
        """
            pass

        def save_payoffs(self, file_name: str) -> None:
        """
        This method should implement a mechanism to save
        the payoff matrix and parameters of the game to permanent storage.
        """
            pass


Simplifying game implementation: `AbstractTwoPLayerGame` and `AbstractNPlayerGame` classes
------------------------------------------------------------------------------------------

However, in most scenarios the fitness of a strategy at a given population
state is its expected payoff at that state. For this reason,
`egttools` provides two other abstract classes to simplify the
implementation of new games:

- `egttools.games.AbstractTwoPLayerGame`, for two-player games;
- and `egttools.games.AbstractNPlayerGame` for N-player games.

When using these abstract classes, you only need to implement two methods:

- `play` and `calculate_payoffs`.

Example: The N-player Stag-Hunt Game
------------------------------------

Below you can find an example on how to implement the
N-player Stag Hunt game from :cite:t:`Pacheco2009` :

.. code-block:: Python

    from egttools.games import AbstractNPlayerGame
    from egttools import sample_simplex

    class NPlayerStagHunt(AbstractNPlayerGame):

        def __init__(self, group_size, enhancement_factor, cooperation_threshold, cost):
            self.group_size_ = group_size  # N
            self.enhancement_factor_ = enhancement_factor  # F
            self.cooperation_threshold_ = cooperation_threshold  # M
            self.cost_ = cost  # c
            self.strategies = ['Defect', 'Cooperate']

            self.nb_strategies_ = 2
            super().__init__(self.nb_strategies_, self.group_size_)

        def play(self, group_composition: Union[List[int], np.ndarray], game_payoffs: np.ndarray) -> None:
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


What if you already have calculated a matrix of expected payoffs?
-----------------------------------------------------------------

In case you have calculated a matrix of expected payoffs for all strategies, and do not want to waste time in implementing
a new game. You can make use of the container classes `egttools.games.Matrix2PlayerGameHolder` for 2-player games
and `egttools.games.MatrixNPlayerGameHolder` for N-player games.

- `Matrix2PlayerGameHolder`:
                                This class expects the number of strategies (`nb_strategies`) in the game and the matrix of expected payoffs
                                as a parameter. The payoff matrix must be square and have the shape (nb_strategies, nb_strategies). So that
                                each entry gives the expected payoff of the row strategy versus the column strategy.
- `MatrixNPlayerGameHolder`:
                                This class expects the number of strategies (`nb_strategies`), size of the group (`group_size`)
                                and the matrix of expected payoffs as a parameter. In this case the payoff matrix must have the shape
                                (nb_strategies, nb_group_configurations), where `nb_group_configurations` is the total number of
                                combinations of strategies in the group, and can be obtained using
                                `egttools.calculate_nb_states(group_size, nb_strategies)`. Thus, each entry in the matrix
                                must give the expected payoff of the row strategy in the group configuration given by the column index.
                                You can obtain the group configuration from an index using `egttools.sample_simplex(index, group_size, nb_strategies)`.
                                **When the row strategy in not present in the column group configuration, the payoff in this entry must be 0.**

.. note::
    You can find an example of how to use these classes :doc:`here <../examples/hawk_dove_dynamics>`.

List of implemented games
-------------------------
- NormalFormGame: implements iterated normal form games (matrix games).
                    This class expects as parameters the number of rounds of the game, a payoff matrix, and a list
                    of strategies that will play the game. You can find more information on how to use implemented
                    strategies or implement new ones :doc:`here <create_new_behaviors>`.

- PGG : implements a version of a Public Goods Game.
        This game expects the size of the group, the cost of cooperation, a multiplication factor and the set of
        strategies that will play the game as parameters. You can find more information on how to use implemented
        strategies or implement new ones :doc:`here <create_new_behaviors>`.

- OneShotCRD : implements the one-shot collective risk dilemma described by :cite:t:`santosRiskCollectiveFailure2011`.
        This game takes as parameters and endowment, which will be equal for all group members, the cost of cooperation,
        the risk of collective failure, the size of the group, and the minimum number of cooperators in a group
        required to reach the collective target.

- CRDGame : implements the collective risk dilemma proposed by :cite:t:`milinski2008collective`.
        This game takes as parameters and endowment, which will be equal for all group members, a threshold, i.e., the
        collective target contributions, the number of rounds of the game, the size of the group, the risk of
        collective failure, an enhancement factor, which multiples the payoffs of all members of the group
        when they reach the collective target, and a `List` with the strategies that will play the game. You can find
        more information on how to use implemented
        strategies or implement new ones :doc:`here <create_new_behaviors>`.

- NPlayerStagHunt : implements the N-player stag hunt game proposed in :cite:t:`Pacheco2009`.
        This game takes as parameters the size of the group, and enhancement factor, a cooperation threshold, i.e.,
        the minimum number of cooperators required to provide the public good, and the cost of cooperation.
        The game is implemented so that the only strategies playing the game is `Cooperate` and `Defect`.
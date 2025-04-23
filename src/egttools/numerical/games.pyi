"""

    The `egttools.numerical.games` submodule provides access to all game implementations available in EGTtools.

    It includes abstract base classes for defining new games, as well as concrete implementations such as
    `NormalFormGame`, `CRDGame`, `NPlayerStagHunt`, and others.

    These classes support the modeling of evolutionary dynamics in finite populations and can be used
    with various numerical tools available in EGTtools to simulate and analyze game-theoretic behavior.

    See Also
    --------
    egttools.numerical.PairwiseComparisonNumerical
    egttools.analytical.PairwiseComparison
    egttools.plotting
    
"""
from __future__ import annotations
import egttools.numerical.numerical_.distributions
import numpy as np
import typing
from abc import ABC
from numpy.typing import NDArray

__all__ = ['AbstractGame', 'AbstractNPlayerGame', 'AbstractSpatialGame', 'CRDGame', 'CRDGameTU',
           'Matrix2PlayerGameHolder', 'MatrixNPlayerGameHolder', 'NPlayerStagHunt', 'NormalFormGame',
           'NormalFormNetworkGame', 'OneShotCRD', 'OneShotCRDNetworkGame']


class AbstractGame(ABC):
    """
    
    Base class for all game-theoretic models in EGTtools.
    
    This abstract class defines the required interface for any game to be used in
    evolutionary dynamics models. All concrete games must inherit from this class
    and implement its methods.
    """

    def __init__(self) -> None:
        ...

    def __str__(self) -> str:
        """
        Returns a string representation of the game object.
        
        Returns
        -------
        str
            A string describing the game instance.
        """

    def calculate_fitness(self, strategy_index: int, pop_size: int, strategies: NDArray[np.uint64]) -> float:
        """
        Computes the fitness of a given strategy in a population.
        
        Parameters
        ----------
        strategy_index : int
            The index of the strategy whose fitness is being computed.
        pop_size : int
            Total population size.
        strategies : NDArray[np.int64]
            Vector representing the number of individuals using each strategy.
        
        Returns
        -------
        float
            The computed fitness of the specified strategy.
        """

    def calculate_payoffs(self) -> NDArray[np.float64]:
        """
        Calculates and stores all payoffs internally for all possible group compositions.
        
        This method must be called before computing fitness values or using the game in simulations.
        """

    def nb_strategies(self) -> int:
        """
        Returns the number of strategies available in the game.
        
        Returns
        -------
        int
            The total number of strategies.
        """

    def payoff(self, strategy: int, group_composition: list[int]) -> float:
        """
        Returns the expected payoff of a specific strategy in a group.
        
        Parameters
        ----------
        strategy : int
            Index of the focal strategy.
        group_composition : NDArray[np.int64]
            Vector specifying the number of individuals using each strategy.
        
        Returns
        -------
        float
            Expected payoff of the strategy in the given group context.
        """

    def payoffs(self) -> NDArray[np.float64]:
        """
        Returns the current payoff matrix of the game.
        
        Returns
        -------
        NDArray[np.float64]
            The stored payoff matrix used in the game.
        """

    def play(self, group_composition: list[int], game_payoffs: list[float]) -> None:
        """
        Computes the payoff of each strategy for a given group composition.
        
        This method modifies `game_payoffs` in-place to store the payoff of each strategy,
        given a group composed of the specified number of individuals per strategy.
        
        This avoids memory reallocation by reusing the existing array, which can
        significantly speed up simulations.
        
        Parameters
        ----------
        group_composition : NDArray[np.int64]
            A vector of shape (n_strategies,) indicating the number of individuals
            playing each strategy in the group.
        game_payoffs : NDArray[np.float64]
            A pre-allocated vector of shape (n_strategies,) which will be updated
            with the payoffs of each strategy in the group. Modified in-place.
        
        Returns
        -------
        None
            This function modifies `game_payoffs` directly.
        
        Examples
        --------
        >>> group = np.array([1, 2, 0])
        >>> out = np.zeros_like(group, dtype=np.float64)
        >>> game.play(group, out)
        >>> print(out)  # Contains payoffs for each strategy in the group
        """

    def save_payoffs(self, file_name: str) -> None:
        """
        Saves the current payoff matrix to a file.
        
        Parameters
        ----------
        file_name : str
            Name of the file to which the matrix should be saved.
        """

    def type(self) -> str:
        """
        Returns the type of the game as a string.
        
        Returns
        -------
        str
            A label identifying the game type (e.g., "NormalFormGame").
        """


class AbstractNPlayerGame(AbstractGame):
    def __init__(self, nb_strategies: int, group_size: int) -> None:
        """
                 Abstract N-Player Game.
        
                 This abstract base class represents a symmetric N-player game in which each strategy's
                 fitness is computed as the expected payoff over all group compositions in a population.
        
                 Notes
                 -----
                 Subclasses must implement the `play` and `calculate_payoffs` methods.
                 The following attributes are expected:
                 - `self.nb_strategies_` (int): number of strategies.
                 - `self.payoffs_` (NDArray): of shape (nb_strategies, nb_group_configurations).
        
                 Parameters
                 ----------
                 nb_strategies : int
                     Total number of strategies in the game.
                 group_size : int
                     Size of the interacting group.
        """

    def __str__(self) -> str:
        """
                 Returns a string representation of the game.
        
                 Returns
                 -------
                 str
        """

    def calculate_fitness(self, strategy_index: int, pop_size: int, strategies: NDArray[np.uint64]) -> float:
        """
                 Computes the fitness of a given strategy in a population state.
        
                 Parameters
                 ----------
                 strategy_index : int
                     The strategy of the focal player.
                 pop_size : int
                     Total population size (excluding the focal player).
                 strategies : NDArray[uint64]
                     The population state as a strategy count vector.
        
                 Returns
                 -------
                 float
                     Fitness of the focal strategy in the given state.
        """

    def calculate_payoffs(self) -> NDArray[np.float64]:
        """
                 Computes and returns the full payoff matrix.
        
                 Returns
                 -------
                 NDArray
                     A matrix with expected payoffs. Each row represents a strategy,
                     each column a group configuration.
        """

    def group_size(self) -> int:
        """
                 Returns the size of the group.
        
                 Returns
                 -------
                 int
        """

    def nb_group_configurations(self) -> int:
        """
                 Returns the number of distinct group configurations.
        
                 Returns
                 -------
                 int
        """

    def nb_strategies(self) -> int:
        """
                 Returns the number of strategies in the game.
        
                 Returns
                 -------
                 int
        """

    def payoff(self, strategy: int, group_composition: list[int]) -> float:
        """
                 Returns the payoff of a strategy in a given group context.
        
                 Parameters
                 ----------
                 strategy : int
                     The strategy index.
                 group_composition : list[int] or NDArray[int]
                     The group configuration.
        
                 Returns
                 -------
                 float
                     The corresponding payoff.
        """

    def payoffs(self) -> NDArray[np.float64]:
        """
                 Returns the payoff matrix.
        
                 Returns
                 -------
                 NDArray
                     The matrix of shape (nb_strategies, nb_group_configurations).
        """

    def play(self, group_composition: {list[int], NDArray[int]}, game_payoffs: {list[float], NDArray[float]}) -> None:
        """
                 Executes the game for a given group composition and fills the payoff vector.
        
                 Parameters
                 ----------
                 group_composition : List[int] | NDArray[int]
                     The number of players of each strategy in the group.
                 game_payoffs : List[float] | NDArray[float]
                     Output container where the payoff of each player will be written.
        """

    def save_payoffs(self, file_name: str) -> None:
        """
                 Saves the payoff matrix to a text file.
        
                 Parameters
                 ----------
                 file_name : str
                     Destination file path.
        """

    def type(self) -> str:
        """
                 Returns the string identifier for the game.
        
                 Returns
                 -------
                 str
        """

    def update_payoff(self, strategy_index: int, group_configuration_index: int, value: float) -> None:
        """
                 Updates an entry in the payoff matrix.
        
                 Parameters
                 ----------
                 strategy_index : int
                     Index of the strategy (row).
                 group_configuration_index : int
                     Index of the group composition (column).
                 value : float
                     The new payoff value.
        """


class AbstractSpatialGame:
    def __init__(self) -> None:
        """
                Abstract base class for spatially structured games.
        
                This interface supports general spatial interaction models, where the fitness of a strategy
                is computed based on a local context (e.g., neighborhood composition).
        
                This is typically used in network-based or spatial grid environments.
        
                Note
                ----
                This interface is still under active development and may change in future versions.
        """

    def __str__(self) -> str:
        """
        String representation of the spatial game.
        """

    def calculate_fitness(self, strategy_index: int, state: NDArray[np.uint64]) -> float:
        """
                Calculates the fitness of a strategy in a local interaction context.
        
                Parameters
                ----------
                strategy_index : int
                    The strategy of the focal player.
                state : NDArray[int]
                    Vector representing the local configuration (e.g., neighbor counts).
        
                Returns
                -------
                float
                    The computed fitness of the strategy in the given local state.
        """

    def nb_strategies(self) -> int:
        """
        Returns the number of strategies in the spatial game.
        """

    def type(self) -> str:
        """
        Identifier for the type of spatial game.
        """


class CRDGame(AbstractGame):
    def __init__(self, endowment: int, threshold: int, nb_rounds: int, group_size: int, risk: float,
                 enhancement_factor: float, strategies: list) -> None:
        """
                Collective Risk Dilemma (CRD) Game.
        
                This implementation allows defining arbitrary strategies using instances of AbstractCRDStrategy.
        
                Based on:
                Milinski et al. (2008). The collective-risk social dilemma and the prevention of simulated
                dangerous climate change. PNAS, 105(7), 2291–2294.
        
                Parameters
                ----------
                endowment : int
                    Initial endowment of each player.
                threshold : int
                    Collective target the group must achieve to avoid risk.
                nb_rounds : int
                    Number of rounds in the game.
                group_size : int
                    Number of players in each group.
                risk : float
                    Probability of losing remaining endowment if the target is not met.
                enhancement_factor : float
                    Multiplier for successful cooperation (may add surplus).
                strategies : list[AbstractCRDStrategy]
                    List of strategy instances.
        """

    def __str__(self) -> str:
        """
        Returns a string representation of the CRDGame.
        """

    def calculate_fitness(self, player_strategy: int, pop_size: int, population_state: NDArray[np.uint64]) -> float:
        """
                Calculates the fitness of a strategy in a given population state.
        
                Parameters
                ----------
                player_strategy : int
                    Index of the focal strategy.
                pop_size : int
                    Total population size (excluding focal).
                population_state : NDArray[numpy.uint64]
                    Vector of strategy counts.
        
                Returns
                -------
                float
        """

    def calculate_group_achievement(self, population_size: int, stationary_distribution: NDArray[np.float64]) -> float:
        """
        Calculates group achievement given a stationary distribution.
        """

    def calculate_payoffs(self) -> NDArray[np.float64]:
        """
                Computes the expected payoffs for each strategy under all group configurations.
        
                Returns
                -------
                NDArray of shape (nb_strategies, nb_group_configurations)
        """

    def calculate_polarization(self, population_size: int, population_state: NDArray[np.float64]) -> NDArray[
        np.float64]:
        """
        Computes contribution polarization relative to the fair contribution (E/2).
        """

    def calculate_polarization_success(self, population_size: int, population_state: NDArray[np.float64]) -> NDArray[
        np.float64]:
        """
        Computes contribution polarization among successful groups.
        """

    def calculate_population_group_achievement(self, population_size: int,
                                               population_state: NDArray[np.uint64]) -> float:
        """
        Calculates group achievement for the population at a given state.
        """

    def nb_strategies(self) -> int:
        """
        Number of strategies in the game.
        """

    def payoff(self, strategy: int, group_composition: list[int]) -> float:
        """
                Returns the payoff of a strategy in a given group composition.
        
                Parameters
                ----------
                strategy : int
                    Index of the strategy.
                group_composition : list[int]
                    Group composition vector.
        
                Returns
                -------
                float
        """

    def payoffs(self) -> NDArray[np.float64]:
        """
        Returns the payoff matrix for all strategies and group configurations.
        """

    def play(self, group_composition: list[int], game_payoffs: list[float]) -> None:
        """
                Plays a single round of the CRD game for the specified group composition.
        
                Parameters
                ----------
                group_composition : list[int] or NDArray[int]
                    Number of players using each strategy.
                game_payoffs : list[float] or NDArray[float]
                    Output vector to store player payoffs.
        """

    def save_payoffs(self, file_name: str) -> None:
        """
                Saves the payoff matrix to a file.
        
                Parameters
                ----------
                file_name : str
                    Output file path.
        """

    def type(self) -> str:
        """
        Returns the type of the game.
        """

    @property
    def endowment(self) -> int:
        """
        Initial endowment for each player.
        """

    @property
    def enhancement_factor(self) -> float:
        """
        Multiplier applied to payoffs if the target is met.
        """

    @property
    def group_size(self) -> int:
        """
        Number of players per group.
        """

    @property
    def nb_rounds(self) -> int:
        """
        Number of rounds in the game.
        """

    @property
    def nb_states(self) -> int:
        """
        Number of distinct population states.
        """

    @property
    def risk(self) -> float:
        """
        Probability of losing endowment if the target is not met.
        """

    @property
    def strategies(self) -> list[...]:
        """
        List of strategy instances in the game.
        """

    @property
    def target(self) -> int:
        """
        Collective target to avoid risk.
        """


class CRDGameTU(AbstractGame):
    def __init__(self, endowment: int, threshold: int, nb_rounds: int, group_size: int, risk: float,
                 tu: egttools.numerical.numerical_.distributions.TimingUncertainty, strategies: list) -> None:
        """
                Collective Risk Dilemma with Timing Uncertainty (CRDGameTU).
        
                This game extends the CRD setting proposed by Milinski et al. (2008) by introducing
                uncertainty about the number of rounds before a potential disaster. Timing uncertainty
                is captured using a TimingUncertainty object.
        
                In this version, players contribute resources over multiple rounds to a common pool.
                If the collective threshold is reached before the disaster occurs, the group avoids the risk.
                Otherwise, each player faces a predefined probability of losing their remaining endowment.
        
                Parameters
                ----------
                endowment : int
                    Initial endowment of each player.
                threshold : int
                    Collective target required to avoid risk.
                nb_rounds : int
                    Maximum number of rounds (before which a disaster may occur).
                group_size : int
                    Number of players per group.
                risk : float
                    Probability of failure if the target is not met.
                tu : TimingUncertainty
                    An object modeling timing uncertainty.
                strategies : list[AbstractCRDStrategy]
                    List of strategy instances.
        """

    def __str__(self) -> str:
        """
        Returns a string representation of the CRDGameTU.
        """

    def calculate_fitness(self, player_strategy: int, pop_size: int, population_state: NDArray[np.uint64]) -> float:
        """
                Computes the fitness of a strategy in a given population state.
        
                Parameters
                ----------
                player_strategy : int
                    Index of the focal strategy.
                pop_size : int
                    Total population size.
                population_state : NDArray
                    Vector of strategy counts.
        
                Returns
                -------
                float
        """

    def calculate_group_achievement(self, population_size: int, stationary_distribution: NDArray[np.float64]) -> float:
        """
        Calculates group achievement based on a stationary distribution.
        """

    def calculate_payoffs(self) -> NDArray[np.float64]:
        """
                Computes the expected payoffs for each strategy across all group configurations.
        
                Returns
                -------
                NDArray
                    Matrix of shape (nb_strategies, nb_group_configurations).
        """

    def calculate_polarization(self, population_size: int, population_state: NDArray[np.float64]) -> NDArray[
        np.float64]:
        """
        Computes contribution polarization in a given population state.
        """

    def calculate_polarization_success(self, population_size: int, population_state: NDArray[np.float64]) -> NDArray[
        np.float64]:
        """
        Computes contribution polarization among successful groups.
        """

    def calculate_population_group_achievement(self, population_size: int,
                                               population_state: NDArray[np.uint64]) -> float:
        """
        Calculates group achievement for a given population state.
        """

    def nb_strategies(self) -> int:
        """
        Number of strategies in the game.
        """

    def payoff(self, strategy: int, group_composition: list[int]) -> float:
        """
                Returns the payoff of a strategy given a group composition.
        
                Parameters
                ----------
                strategy : int
                    Strategy index.
                group_composition : list[int]
                    Group composition vector.
        
                Returns
                -------
                float
        """

    def payoffs(self) -> NDArray[np.float64]:
        """
        Returns the matrix of expected payoffs.
        """

    def play(self, group_composition: list[int], game_payoffs: list[float]) -> None:
        """
                Executes one iteration of the CRD game using a specific group composition.
        
                This method calculates the payoffs for each player based on their strategy
                and the current group composition under timing uncertainty.
        
                Parameters
                ----------
                group_composition : list[int] or NDArray[int]
                    Number of players per strategy in the group.
                game_payoffs : list[float] or NDArray[float]
                    Output vector for player payoffs.
        """

    def save_payoffs(self, file_name: str) -> None:
        """
                Saves the payoff matrix to a text file.
        
                Parameters
                ----------
                file_name : str
                    Path to the output file.
        """

    def type(self) -> str:
        """
        Returns the type identifier of the game.
        """

    @property
    def endowment(self) -> int:
        """
        Initial endowment per player.
        """

    @property
    def group_size(self) -> int:
        """
        Size of the group.
        """

    @property
    def min_rounds(self) -> int:
        """
        Minimum number of rounds the game will run.
        """

    @property
    def nb_states(self) -> int:
        """
        Number of possible population states.
        """

    @property
    def risk(self) -> float:
        """
        Probability of losing endowment if the target is not met.
        """

    @property
    def strategies(self) -> list[...]:
        """
        List of strategy objects participating in the game.
        """

    @property
    def target(self) -> int:
        """
        Target that the group must reach.
        """


class Matrix2PlayerGameHolder(AbstractGame):
    @staticmethod
    def payoff(*args, **kwargs) -> float:
        """
        Returns the payoff for a given strategy pair.
        """

    def __init__(self, nb_strategies: int, payoff_matrix: NDArray[np.float64, NDArray.flags.c_contiguous]) -> None:
        """
                Matrix-based 2-Player Game Holder.
        
                Stores the expected payoffs between strategies in a 2-player game.
                This class is useful for simulations where the payoff matrix is externally computed
                and fixed, enabling fast fitness calculations without recomputation.
        
                Parameters
                ----------
                nb_strategies : int
                    Number of strategies used in the game.
                payoff_matrix : NDArray[float64[m, m]]
                    Matrix containing the payoff of each strategy against all others.
        """

    def __str__(self) -> str:
        ...

    def calculate_fitness(self, player_type: int, pop_size: int, population_state: NDArray[np.uint64]) -> float:
        """
                Computes the fitness of a strategy given the population configuration.
        
                Assumes the focal player is not included in the population state.
        
                Parameters
                ----------
                player_type : int
                    Index of the focal strategy.
                pop_size : int
                    Size of the population.
                population_state : NDArray
                    Vector of counts of each strategy in the population.
        
                Returns
                -------
                float
        """

    def calculate_payoffs(self) -> NDArray[np.float64]:
        """
                Returns the stored payoff matrix.
        
                Returns
                -------
                NDArray
                    Payoff matrix of shape (nb_strategies, nb_strategies).
        """

    def nb_strategies(self) -> int:
        """
        Returns the number of strategies in the game.
        """

    def payoffs(self) -> NDArray[np.float64]:
        """
        Returns the expected payoff matrix.
        """

    def play(self, group_composition: list[int], game_payoffs: list[float]) -> None:
        """
                Executes a match given a group composition and stores the resulting payoffs.
        
                Parameters
                ----------
                group_composition : list[int] or NDArray[int]
                    Count of each strategy in the group (typically 2 players).
                game_payoffs : list[float] or NDArray[float]
                    Output vector to be filled with each player's payoff.
        """

    def save_payoffs(self, arg0: str) -> None:
        """
        Saves the current payoff matrix to a text file.
        """

    def type(self) -> str:
        ...

    def update_payoff_matrix(self, payoff_matrix: NDArray[np.float64]) -> None:
        """
        Replaces the internal payoff matrix with a new one.
        """


class MatrixNPlayerGameHolder(AbstractGame):
    @staticmethod
    def payoff(*args, **kwargs) -> float:
        """
        Returns the payoff for a strategy given a specific group configuration.
        """

    def __init__(self, nb_strategies: int, group_size: int,
                 payoff_matrix: NDArray[np.float64, NDArray.flags.c_contiguous]) -> None:
        """
                Matrix-based N-Player Game Holder.
        
                Stores a fixed matrix of expected payoffs for N-player games, where each entry corresponds
                to the expected payoff for a strategy across possible group configurations.
        
                This class enables efficient fitness evaluations for evolutionary simulations without
                re-computing payoffs.
        
                Parameters
                ----------
                nb_strategies : int
                    Number of strategies in the game.
                group_size : int
                    Size of the interacting group.
                payoff_matrix : NDArray[float64[m, n]]
                    Matrix of shape (nb_strategies, nb_group_configurations) encoding payoffs for all strategy-group pairs.
        """

    def __str__(self) -> str:
        ...

    def calculate_fitness(self, player_strategy: int, pop_size: int, population_state: NDArray[np.uint64]) -> float:
        """
                Computes the fitness of a strategy based on the current population state.
        
                Parameters
                ----------
                player_strategy : int
                    Index of the strategy used by the focal player.
                pop_size : int
                    Population size (excluding focal player).
                population_state : NDArray
                    Vector of strategy counts in the population.
        
                Returns
                -------
                float
                    Fitness of the focal strategy.
        """

    def calculate_payoffs(self) -> NDArray[np.float64]:
        """
                Returns the internal matrix of precomputed payoffs.
        
                Returns
                -------
                NDArray
                    Matrix of shape (nb_strategies, nb_group_configurations).
        """

    def group_size(self) -> int:
        """
        Size of the player group.
        """

    def nb_group_configurations(self) -> int:
        """
        Number of distinct group configurations supported by the matrix.
        """

    def nb_strategies(self) -> int:
        """
        Number of strategies defined in the game.
        """

    def payoffs(self) -> NDArray[np.float64]:
        """
        Returns the full payoff matrix.
        """

    def play(self, group_composition: list[int], game_payoffs: list[float]) -> None:
        """
                Simulates the game based on a predefined payoff matrix.
        
                Parameters
                ----------
                group_composition : list[int] or NDArray[int]
                    Number of players using each strategy in the group.
                game_payoffs : list[float] or NDArray[float]
                    Output vector for storing player payoffs.
        """

    def save_payoffs(self, arg0: str) -> None:
        """
        Saves the payoff matrix to a text file.
        """

    def type(self) -> str:
        ...

    def update_payoff_matrix(self, payoff_matrix: NDArray[np.float64]) -> None:
        """
        Replaces the stored payoff matrix with a new one.
        """


class NPlayerStagHunt(AbstractGame):
    @staticmethod
    def payoff(*args, **kwargs) -> float:
        """
        Returns the payoff of a strategy given a group composition.
        """

    def __init__(self, group_size: int, cooperation_threshold: int, enhancement_factor: float, cost: float) -> None:
        """
                N-Player Stag Hunt (NPSH).
        
                This game models a public goods scenario where individuals may choose to cooperate (hunt stag)
                or defect (hunt hare). Only if a sufficient number of players cooperate is the public good provided.
        
                Based on the model described in:
                Pacheco, J. M., Vasconcelos, V. V., & Santos, F. C. (2014).
                "Evolutionary dynamics of collective action in N-person stag hunt dilemmas."
                Journal of Theoretical Biology, 350, 61–68.
        
                Parameters
                ----------
                group_size : int
                    Number of players in the group (N).
                cooperation_threshold : int
                    Minimum number of cooperators (M) required to produce the collective benefit.
                enhancement_factor : float
                    Multiplicative factor applied to the benefit when the public good is provided.
                cost : float
                    Cost of cooperation.
        """

    def __str__(self) -> str:
        ...

    def calculate_fitness(self, player_strategy: int, pop_size: int, population_state: NDArray[np.uint64]) -> float:
        """
                Computes the fitness of a strategy given a population state.
        
                Parameters
                ----------
                player_strategy : int
                    Index of the focal strategy.
                pop_size : int
                    Total number of individuals in the population.
                population_state : NDArray
                    Vector of strategy counts (excluding the focal individual).
        
                Returns
                -------
                float
        """

    def calculate_group_achievement(self, population_size: int, stationary_distribution: NDArray[np.float64]) -> float:
        """
                Computes the expected collective success weighted by a stationary distribution.
        
                Parameters
                ----------
                population_size : int
                    Total population size.
                stationary_distribution : NDArray[float]
                    Stationary distribution over population states.
        
                Returns
                -------
                float
                    Average group success probability.
        """

    def calculate_payoffs(self) -> NDArray[np.float64]:
        """
                Computes and stores the expected payoff matrix for all strategy-group combinations.
        
                Also updates internal cooperation level metrics for use in diagnostics and analysis.
        """

    def calculate_population_group_achievement(self, population_size: int,
                                               population_state: NDArray[np.uint64]) -> float:
        """
                Estimates the likelihood that a random group from the population meets the cooperation threshold.
        
                This value can serve as a proxy for expected success of collective actions.
        
                Parameters
                ----------
                population_size : int
                    Total number of individuals in the population.
                population_state : NDArray
                    Vector of counts for each strategy.
        
                Returns
                -------
                float
        """

    def nb_group_configurations(self) -> int:
        """
        Number of unique group compositions.
        """

    def nb_strategies(self) -> int:
        """
        Number of strategies involved in the game.
        """

    def payoffs(self) -> NDArray[np.float64]:
        """
        Returns the expected payoff matrix for all strategy combinations.
        """

    def play(self, group_composition: list[int], game_payoffs: list[float]) -> None:
        """
                Simulates the game and fills in the payoff vector for a given group composition.
        
                Parameters
                ----------
                group_composition : list[int] or NDArray[int]
                    Number of players of each strategy in the group.
                game_payoffs : list[float] or NDArray[float]
                    Output vector to store the resulting payoff for each player.
        """

    def save_payoffs(self, arg0: str) -> None:
        """
        Saves the payoff matrix to a text file.
        """

    def strategies(self) -> list[str]:
        """
        Returns the list of strategy names used in the game.
        """

    def type(self) -> str:
        ...

    @property
    def cooperation_threshold(self) -> int:
        """
        Minimum number of cooperators required to succeed.
        """

    @property
    def cost(self) -> float:
        """
        Cost paid by each cooperator.
        """

    @property
    def enhancement_factor(self) -> float:
        """
        Factor by which collective benefit is multiplied when successful.
        """

    @property
    def group_achievement_per_group(self) -> NDArray[np.int64]:
        ...

    @property
    def group_size(self) -> int:
        """
        Size of the player group in each game round.
        """


class NormalFormGame(AbstractGame):
    @typing.overload
    def __init__(self, nb_rounds: int, payoff_matrix: NDArray[np.float64, NDArray.flags.c_contiguous]) -> None:
        """
                Normal Form Game with two actions.
        
                Implements a repeated symmetric 2-player game based on a payoff matrix.
        
                Parameters
                ----------
                nb_rounds : int
                    Number of rounds played by each strategy pair.
                payoff_matrix : NDArray of shape (2, 2)
                    Payoff matrix where entry (i, j) gives the payoff of strategy i against j.
        """

    @typing.overload
    def __init__(self, nb_rounds: int, payoff_matrix: NDArray[np.float64, NDArray.flags.c_contiguous],
                 strategies: list) -> None:
        """
                Normal Form Game initialized with custom strategy classes.
        
                This constructor allows using any number of strategies, defined in Python as subclasses
                of AbstractNFGStrategy.
        
                Parameters
                ----------
                nb_rounds : int
                    Number of rounds in the repeated game.
                payoff_matrix : NDArray
                    Payoff matrix of shape (nb_actions, nb_actions).
                strategies : list[AbstractNFGStrategy]
                    List of strategy instances.
        """

    def __str__(self) -> str:
        """
                Returns a string representation of the NormalFormGame instance.
        
                Returns
                -------
                str
        """

    def calculate_fitness(self, player_strategy: int, population_size: int,
                          population_state: NDArray[np.uint64]) -> float:
        """
                Computes the fitness of a strategy in a given population state.
        
                Parameters
                ----------
                player_strategy : int
                    Index of the focal strategy.
                population_size : int
                    Total number of individuals (excluding the focal player).
                population_state : NDArray[numpy.uint64]
                    Strategy counts in the population.
        
                Returns
                -------
                float
                    Fitness of the focal strategy.
        """

    def calculate_payoffs(self) -> NDArray[np.float64]:
        """
                Calculates the expected payoff matrix for all strategy pairs.
        
                Returns
                -------
                NDArray of shape (nb_strategies, nb_strategies)
                    Matrix of expected payoffs between strategies.
        """

    def expected_payoffs(self) -> NDArray[np.float64]:
        """
        Returns the matrix of expected payoffs between strategies.
        """

    def nb_strategies(self) -> int:
        """
        Number of strategies in the game.
        """

    def payoff(self, strategy: int, strategy_pair: list[int]) -> float:
        """
                Returns the payoff for a given strategy in a specific match-up.
        
                Parameters
                ----------
                strategy : int
                    Index of the strategy used by the player.
                strategy_pair : list[int]
                    List of two integers representing the strategies in the match-up.
        
                Returns
                -------
                float
        """

    def payoffs(self) -> NDArray[np.float64]:
        """
                Returns the payoff matrix.
        
                Returns
                -------
                NDArray of shape (nb_strategies, nb_strategies)
        """

    def play(self, group_composition: list[int], game_payoffs: list[float]) -> None:
        """
                Executes a game round and stores the resulting payoffs.
        
                Parameters
                ----------
                group_composition : list[int] or NDArray[int]
                    Composition of the pairwise game (usually 2 players).
                game_payoffs : list[float] or NDArray[float]
                    Output array to store individual payoffs.
        """

    def save_payoffs(self, file_name: str) -> None:
        """
                Saves the payoff matrix to a text file.
        
                Parameters
                ----------
                file_name : str
                    File path where the matrix will be saved.
        """

    def type(self) -> str:
        """
                Returns the type identifier of the game.
        
                Returns
                -------
                str
        """

    @property
    def nb_rounds(self) -> int:
        """
        Number of rounds per interaction.
        """

    @property
    def nb_states(self) -> int:
        """
        Number of unique pairwise strategy combinations.
        """

    @property
    def strategies(self) -> list[...]:
        """
        List of strategies participating in the game.
        """


class NormalFormNetworkGame(AbstractSpatialGame):
    @typing.overload
    def __init__(self, nb_rounds: int, payoff_matrix: NDArray[np.float64, NDArray.flags.c_contiguous]) -> None:
        """
                Normal Form Network Game.
        
                This class implements a networked version of a standard two-player normal form game.
                Agents are placed in a network or spatial structure and interact only with their neighbors,
                using a fixed payoff matrix to compute outcomes.
        
                Each agent selects an action (or strategy), and receives payoffs based on pairwise interactions
                with all neighbors. Repeated interactions are supported, allowing for multi-round strategies
                and dynamic behaviors such as conditional cooperation.
        
                Parameters
                ----------
                nb_rounds : int
                    Number of repeated rounds for each pairwise encounter.
                payoff_matrix : NDArray[float64[m, m]]
                    Payoff matrix specifying the outcomes for all strategy pairs.
        """

    @typing.overload
    def __init__(self, nb_rounds: int, payoff_matrix: NDArray[np.float64, NDArray.flags.c_contiguous],
                 strategies: list) -> None:
        """
                Normal Form Game with custom strategy list.
        
                This constructor initializes a network game with a custom list of strategies.
                Each strategy must be a pointer to a subclass of AbstractNFGStrategy.
        
                Parameters
                ----------
                nb_rounds : int
                    Number of rounds of interaction.
                payoff_matrix : NDArray[float64[m, m]]
                    Payoff matrix used for each pairwise encounter.
                strategies : List[egttools.behaviors.AbstractNFGStrategy]
                    A list of strategy pointers.
        """

    def __str__(self) -> str:
        """
        String representation of the game object.
        """

    def calculate_cooperation_level_neighborhood(self, strategy_index: int, state: NDArray[np.uint64]) -> float:
        """
                Calculates the level of cooperation in a given neighborhood.
        
                Useful when using multi-round or conditional strategies where past actions
                may affect behavior.
        
                Parameters
                ----------
                strategy_index : int
                    Focal strategy.
                state : NDArray[int]
                    Neighbor strategy counts.
        
                Returns
                -------
                float
                    Level of cooperation.
        """

    def calculate_fitness(self, strategy_index: int, state: NDArray[np.uint64]) -> float:
        """
                Computes fitness of a strategy given a neighborhood configuration.
        
                Parameters
                ----------
                strategy_index : int
                    Strategy whose fitness will be computed.
                state : NDArray[int]
                    Vector representing neighborhood strategy counts.
        
                Returns
                -------
                float
                    Fitness value.
        """

    def calculate_payoffs(self) -> None:
        """
        Recalculates the expected payoff matrix based on current strategies.
        """

    def expected_payoffs(self) -> NDArray[np.float64]:
        """
        Returns the expected payoffs for each strategy.
        """

    def nb_rounds(self) -> int:
        """
        Number of repeated rounds per encounter.
        """

    def nb_strategies(self) -> int:
        """
        Number of strategies available.
        """

    def strategies(self) -> list[...]:
        """
        List of strategies currently active in the game.
        """

    def type(self) -> str:
        """
        Returns the type identifier of the game.
        """


class OneShotCRD(AbstractGame):
    @staticmethod
    def payoff(*args, **kwargs) -> float:
        """
        Returns the payoff for a given strategy and group composition.
        """

    def __init__(self, endowment: float, cost: float, risk: float, group_size: int, min_nb_cooperators: int) -> None:
        """
                One-Shot Collective Risk Dilemma (CRD).
        
                This implementation models the one-shot version of the CRD introduced in:
                Santos, F. C., & Pacheco, J. M. (2011).
                "Risk of collective failure provides an escape from the tragedy of the commons."
                PNAS, 108(26), 10421–10425.
        
                A single group of `group_size` players must decide whether to contribute to a public good.
                Cooperators (Cs) contribute a fraction `cost` of their `endowment`, while Defectors (Ds) contribute nothing.
                If the number of cooperators is greater than or equal to `min_nb_cooperators`, all players avoid the risk
                and receive their remaining endowment. Otherwise, each player loses their remaining endowment with
                probability `risk`.
        
                Parameters
                ----------
                endowment : float
                    The initial endowment received by all players.
                cost : float
                    The fraction of the endowment that Cooperators contribute (in [0, 1]).
                risk : float
                    Probability of collective loss if the group fails to reach the threshold.
                group_size : int
                    Number of players in the group.
                min_nb_cooperators : int
                    Minimum number of cooperators needed to avoid risk.
        """

    def __str__(self) -> str:
        ...

    def calculate_fitness(self, player_strategy: int, pop_size: int, population_state: NDArray[np.uint64]) -> float:
        """
                Calculates the fitness of a strategy given a population state.
        
                Assumes the focal player is excluded from the population state.
        
                Parameters
                ----------
                player_strategy : int
                    Index of the focal strategy.
                pop_size : int
                    Population size.
                population_state : NDArray
                    Vector of strategy counts in the population.
        
                Returns
                -------
                float
        """

    def calculate_group_achievement(self, population_size: int, stationary_distribution: NDArray[np.float64]) -> float:
        """
                Computes group achievement from a stationary distribution.
        
                This method evaluates the probability that a group avoids risk across all states
                weighted by their probability in the stationary distribution.
        
                Parameters
                ----------
                population_size : int
                    Total population size.
                stationary_distribution : NDArray[float]
                    Stationary distribution over population states.
        
                Returns
                -------
                float
                    Weighted average group success probability.
        """

    def calculate_payoffs(self) -> NDArray[np.float64]:
        """
                Updates the payoff matrix and cooperation level matrix for all strategy pairs.
        
                This method precomputes and stores the expected payoff for each strategy given every
                possible group composition, allowing for faster access in subsequent simulations.
        """

    def calculate_population_group_achievement(self, population_size: int,
                                               population_state: NDArray[np.uint64]) -> float:
        """
                Computes the group achievement for the given population state.
        
                This metric captures the expected probability that a randomly formed group from the population
                will reach the collective contribution threshold.
        
                Parameters
                ----------
                population_size : int
                    Total number of individuals.
                population_state : NDArray
                    Vector of counts of each strategy in the population.
        
                Returns
                -------
                float
                    The probability that a randomly formed group avoids the collective risk.
        """

    def nb_strategies(self) -> int:
        """
        Number of strategies in the game.
        """

    def payoffs(self) -> NDArray[np.float64]:
        """
        Returns the expected payoff matrix.
        """

    def play(self, group_composition: list[int], game_payoffs: list[float]) -> None:
        """
                Executes a one-shot CRD round and updates payoffs for the given group composition.
        
                Assumes two strategies: Defectors (index 0) and Cooperators (index 1).
        
                The payoffs are computed as follows:
        
                .. math::
                    \\Pi_{D}(k) = b\\{\\theta(k-M)+ (1-r)[1 - \\theta(k-M)]\\}
        
                    \\Pi_{C}(k) = \\Pi_{D}(k) - cb
        
                    \\text{where } \\theta(x) = \\begin{cases} 0 & x < 0 \\\\ 1 & x \\geq 0 \\end{cases}
        
                Parameters
                ----------
                group_composition : list[int] or NDArray[int]
                    Number of players per strategy.
                game_payoffs : list[float] or NDArray[float]
                    Output vector to store payoffs for each player.
        """

    def save_payoffs(self, arg0: str) -> None:
        """
        Saves the payoff matrix to a text file.
        """

    def type(self) -> str:
        ...

    @property
    def cost(self) -> float:
        """
        Fraction of endowment contributed by cooperators.
        """

    @property
    def endowment(self) -> float:
        """
        Initial endowment per player.
        """

    @property
    def group_achievement_per_group(self) -> NDArray[np.int64]:
        ...

    @property
    def group_size(self) -> int:
        """
        Number of players per group.
        """

    @property
    def min_nb_cooperators(self) -> int:
        """
        Minimum number of cooperators required to avoid risk.
        """

    @property
    def nb_states(self) -> int:
        """
        Number of distinct group compositions.
        """

    @property
    def risk(self) -> float:
        """
        Probability of collective failure.
        """


class OneShotCRDNetworkGame(AbstractSpatialGame):
    def __init__(self, endowment: float, cost: float, risk: float, min_nb_cooperators: int) -> None:
        """
                One-Shot Collective Risk Dilemma in Networks.
        
                This game implements the one-shot version of the Collective Risk Dilemma in spatial or networked settings,
                following the formulation in:
        
                Santos, F. C., & Pacheco, J. M. (2011).
                "Risk of collective failure provides an escape from the tragedy of the commons."
                Proceedings of the National Academy of Sciences, 108(26), 10421–10425.
        
                The game is played once by each group. Cooperation is costly, and a threshold number of
                cooperators is required to avoid collective loss. Spatial interaction allows strategies to
                propagate based on local fitness.
        
                Parameters
                ----------
                endowment : float
                    Initial endowment received by each individual.
                cost : float
                    Cost of contributing to the public good.
                risk : float
                    Probability of collective loss if the threshold is not met.
                min_nb_cooperators : int
                    Minimum number of cooperators required to avoid risk.
        """

    def __str__(self) -> str:
        """
        String representation of the game.
        """

    def calculate_fitness(self, strategy_index: int, state: NDArray[np.uint64]) -> float:
        """
                Computes the fitness of a strategy in a local neighborhood.
        
                The neighborhood composition determines whether the group reaches the threshold required
                to avoid risk. If successful, players keep their endowment; otherwise, they lose it with probability `risk`.
        
                Parameters
                ----------
                strategy_index : int
                    The focal strategy being evaluated.
                state : NDArray[int]
                    Vector representing the number of neighbors using each strategy.
        
                Returns
                -------
                float
                    The fitness of the strategy given the local state.
        """

    def cost(self) -> float:
        """
        Returns the cost of cooperation.
        """

    def endowment(self) -> float:
        """
        Returns the initial endowment for each player.
        """

    def min_nb_cooperators(self) -> int:
        """
        Returns the minimum number of cooperators required to prevent collective loss.
        """

    def nb_strategies(self) -> int:
        """
        Returns the number of strategies available in the game.
        """

    def risk(self) -> float:
        """
        Returns the probability of losing endowment if cooperation fails.
        """

    def type(self) -> str:
        """
        Returns the identifier for the game type.
        """


__init__: str = 'The `egttools.numerical.games` submodule contains the available games.'

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
import numpy
import typing
__all__: list[str] = ['AbstractGame', 'AbstractNPlayerGame', 'AbstractReplicatorGame', 'AbstractSpatialGame', 'CRDGame', 'CRDGameTU', 'Matrix2PlayerGameHolder', 'MatrixNPlayerGameHolder', 'NPlayerStagHunt', 'NormalFormGame', 'NormalFormNetworkGame', 'OneShotCRD', 'OneShotCRDNetworkGame']
class AbstractGame:
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
    def calculate_fitness(self, strategy_index: int, pop_size: int, strategies: numpy.ndarray[numpy.uint64[m, 1]]) -> float:
        """
        Computes the fitness of a given strategy in a population.
        
        Parameters
        ----------
        strategy_index : int
            The index of the strategy whose fitness is being computed.
        pop_size : int
            Total population size.
        strategies : numpy.ndarray
            One-dimensional array representing the number of individuals using each strategy.
        
        Returns
        -------
        float
            The computed fitness of the specified strategy.
        """
    def calculate_payoffs(self) -> numpy.ndarray[numpy.float64[m, n]]:
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
        group_composition : numpy.ndarray
            One-dimensional array specifying the number of individuals using each strategy.
        
        Returns
        -------
        float
            Expected payoff of the strategy in the given group context.
        """
    def payoffs(self) -> numpy.ndarray[numpy.float64[m, n]]:
        """
        Returns the current payoff matrix of the game.
        
        Returns
        -------
        numpy.ndarray
            The stored payoff matrix used in the game.
        """
    def play(self, group_composition: list[int], game_payoffs: list[float]) -> None:
        """
        Computes the payoff of each strategy for a given group composition.
        
        This method modifies `game_payoffs` in-place to store the payoff of each strategy,
        given a group composed of the specified number of individuals per strategy.
        
        Parameters
        ----------
        group_composition : numpy.ndarray
            One-dimensional array indicating the number of individuals playing each strategy in the group.
        game_payoffs : numpy.ndarray
            Pre-allocated one-dimensional array that will be updated with the payoffs of each strategy.
        
        Returns
        -------
        None
            This function modifies `game_payoffs` directly.
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
            A label identifying the game type.
        """
class AbstractNPlayerGame(AbstractGame):
    def __init__(self, nb_strategies: int, group_size: int) -> None:
        """
        Abstract N-player game.
        
        This abstract base class represents a symmetric N-player game in which each strategy's
        fitness is computed as the expected payoff over all group compositions in a population.
        
        Parameters
        ----------
        nb_strategies : int
            Total number of strategies in the game.
        group_size : int
            Size of the interacting group.
        """
    def __str__(self) -> str:
        ...
    def calculate_fitness(self, strategy_index: int, pop_size: int, strategies: numpy.ndarray[numpy.uint64[m, 1]]) -> float:
        """
        Computes the fitness of a given strategy in a population state.
        
        Parameters
        ----------
        strategy_index : int
            The strategy of the focal player.
        pop_size : int
            Total population size.
        strategies : numpy.ndarray
            Population state as a strategy count vector.
        
        Returns
        -------
        float
            Fitness of the focal strategy in the given state.
        """
    def calculate_payoffs(self) -> numpy.ndarray[numpy.float64[m, n]]:
        """
        Computes and returns the full payoff matrix.
        
        Returns
        -------
        numpy.ndarray
            A matrix with expected payoffs. Each row represents a strategy,
            and each column a group configuration.
        """
    def group_size(self) -> int:
        ...
    def nb_group_configurations(self) -> int:
        ...
    def nb_strategies(self) -> int:
        ...
    def payoff(self, strategy: int, group_composition: list[int]) -> float:
        """
        Returns the payoff of a strategy in a given group context.
        
        Parameters
        ----------
        strategy : int
            The strategy index.
        group_composition : numpy.ndarray
            The group configuration.
        
        Returns
        -------
        float
            The corresponding payoff.
        """
    def payoffs(self) -> numpy.ndarray[numpy.float64[m, n]]:
        """
        Returns the payoff matrix.
        
        Returns
        -------
        numpy.ndarray
            Matrix of shape (nb_strategies, nb_group_configurations).
        """
    def play(self, group_composition: list[int], game_payoffs: list[float]) -> None:
        """
        Executes the game for a given group composition and fills the payoff vector.
        
        Parameters
        ----------
        group_composition : numpy.ndarray
            One-dimensional array containing the number of players of each strategy in the group.
        game_payoffs : numpy.ndarray
            Output container where the payoff of each strategy will be written.
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
        ...
    def update_payoff(self, strategy_index: int, group_configuration_index: int, value: float) -> None:
        """
        Updates an entry in the payoff matrix.
        
        Parameters
        ----------
        strategy_index : int
            Index of the strategy.
        group_configuration_index : int
            Index of the group composition.
        value : float
            The new payoff value.
        """
class AbstractReplicatorGame:
    """
    
    Base class for games that define fitness in the infinite-population limit.
    
    This abstract class defines the interface required for a game to be used with
    replicator dynamics in EGTtools. Concrete implementations must provide the
    number of strategies, the group size, a method to compute the fitness vector at
    a given population state, and access to the corresponding payoff table when
    available.
    """
    def __init__(self) -> None:
        ...
    def __str__(self) -> str:
        """
        Returns a string representation of the game object.
        
        Returns
        -------
        str
            A short description of the game.
        """
    def calculate_fitness(self, frequencies: numpy.ndarray[numpy.float64[m, 1]]) -> numpy.ndarray[numpy.float64[m, 1]]:
        """
        Returns the expected fitness of all strategies at a given population state.
        
        Parameters
        ----------
        frequencies : numpy.ndarray
            One-dimensional array containing the frequency of each strategy in the population.
        
        Returns
        -------
        numpy.ndarray
            One-dimensional array containing the expected fitness of each strategy.
        """
    def calculate_payoffs(self) -> numpy.ndarray[numpy.float64[m, n]]:
        """
        Computes or refreshes the payoff table of the game.
        
        Implementations may use this method to lazily compute and cache the payoff
        structure associated with the game.
        
        Returns
        -------
        numpy.ndarray
            The payoff table of the game.
        """
    def group_size(self) -> int:
        """
        Returns the group size of the game.
        
        Returns
        -------
        int
            The number of individuals in each interacting group.
        """
    def nb_strategies(self) -> int:
        """
        Returns the number of strategies available in the game.
        
        Returns
        -------
        int
            The total number of strategies.
        """
    def payoffs(self) -> numpy.ndarray[numpy.float64[m, n]]:
        """
        Returns the currently stored payoff table of the game.
        
        If the payoff table is computed lazily, `calculate_payoffs()` should be called
        first to ensure that the returned table is initialized and up to date.
        
        Returns
        -------
        numpy.ndarray
            The current payoff table of the game.
        """
    def type(self) -> str:
        """
        Returns the type of the game as a string.
        
        Returns
        -------
        str
            A label identifying the game type.
        """
class AbstractSpatialGame:
    def __init__(self) -> None:
        """
        Abstract base class for spatially structured games.
        
        This interface supports general spatial interaction models, where the fitness of a strategy
        is computed based on a local context.
        """
    def __str__(self) -> str:
        ...
    def calculate_fitness(self, strategy_index: int, state: numpy.ndarray[numpy.uint64[m, 1]]) -> float:
        """
        Calculates the fitness of a strategy in a local interaction context.
        
        Parameters
        ----------
        strategy_index : int
            The strategy of the focal player.
        state : numpy.ndarray
            Vector representing the local configuration.
        
        Returns
        -------
        float
            The computed fitness of the strategy in the given local state.
        """
    def nb_strategies(self) -> int:
        ...
    def type(self) -> str:
        ...
class CRDGame(AbstractGame):
    def __init__(self, endowment: int, threshold: int, nb_rounds: int, group_size: int, risk: float, enhancement_factor: float, strategies: list) -> None:
        """
        Collective risk dilemma game.
        
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
            Multiplier for successful cooperation.
        strategies : list[AbstractCRDStrategy]
            List of strategy instances.
        """
    def __str__(self) -> str:
        ...
    def calculate_fitness(self, player_strategy: int, pop_size: int, population_state: numpy.ndarray[numpy.uint64[m, 1]]) -> float:
        """
        Calculates the fitness of a strategy in a given population state.
        
        Parameters
        ----------
        player_strategy : int
            Index of the focal strategy.
        pop_size : int
            Total population size.
        population_state : numpy.ndarray
            Vector of strategy counts.
        
        Returns
        -------
        float
        """
    def calculate_group_achievement(self, population_size: int, stationary_distribution: numpy.ndarray[numpy.float64[m, 1]]) -> float:
        """
        Calculates group achievement given a stationary distribution.
        """
    def calculate_payoffs(self) -> numpy.ndarray[numpy.float64[m, n]]:
        """
        Computes the expected payoffs for each strategy under all group configurations.
        
        Returns
        -------
        numpy.ndarray
        """
    def calculate_polarization(self, population_size: int, population_state: numpy.ndarray[numpy.float64[m, 1]]) -> numpy.ndarray[numpy.float64[3, 1]]:
        """
        Computes contribution polarization relative to the fair contribution.
        """
    def calculate_polarization_success(self, population_size: int, population_state: numpy.ndarray[numpy.float64[m, 1]]) -> numpy.ndarray[numpy.float64[3, 1]]:
        """
        Computes contribution polarization among successful groups.
        """
    def calculate_population_group_achievement(self, population_size: int, population_state: numpy.ndarray[numpy.uint64[m, 1]]) -> float:
        """
        Calculates group achievement for the population at a given state.
        """
    def nb_strategies(self) -> int:
        ...
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
    def payoffs(self) -> numpy.ndarray[numpy.float64[m, n]]:
        """
        Returns the payoff matrix for all strategies and group configurations.
        """
    def play(self, arg0: list[int], arg1: list[float]) -> None:
        """
        Plays a single round of the CRD game for the specified group composition.
        
        Parameters
        ----------
        group_composition : numpy.ndarray
            Number of players using each strategy.
        game_payoffs : numpy.ndarray
            Output vector to store player payoffs.
        """
    def save_payoffs(self, arg0: str) -> None:
        """
        Saves the payoff matrix to a file.
        
        Parameters
        ----------
        file_name : str
            Output file path.
        """
    def type(self) -> str:
        ...
    @property
    def endowment(self) -> int:
        ...
    @property
    def enhancement_factor(self) -> float:
        ...
    @property
    def group_size(self) -> int:
        ...
    @property
    def nb_rounds(self) -> int:
        ...
    @property
    def nb_states(self) -> int:
        ...
    @property
    def risk(self) -> float:
        ...
    @property
    def strategies(self) -> list:
        """
        List of strategy instances in the game.
        """
    @property
    def target(self) -> int:
        ...
class CRDGameTU(AbstractGame):
    def __init__(self, endowment: int, threshold: int, nb_rounds: int, group_size: int, risk: float, tu: egttools.numerical.numerical_.distributions.TimingUncertainty, strategies: list) -> None:
        """
        Collective risk dilemma with timing uncertainty.
        
        Parameters
        ----------
        endowment : int
            Initial endowment of each player.
        threshold : int
            Collective target required to avoid risk.
        nb_rounds : int
            Maximum number of rounds.
        group_size : int
            Number of players per group.
        risk : float
            Probability of failure if the target is not met.
        tu : TimingUncertainty
            Object modeling timing uncertainty.
        strategies : list[AbstractCRDStrategy]
            List of strategy instances.
        """
    def __str__(self) -> str:
        ...
    def calculate_fitness(self, player_strategy: int, pop_size: int, population_state: numpy.ndarray[numpy.uint64[m, 1]]) -> float:
        """
        Computes the fitness of a strategy in a given population state.
        
        Parameters
        ----------
        player_strategy : int
            Index of the focal strategy.
        pop_size : int
            Total population size.
        population_state : numpy.ndarray
            Vector of strategy counts.
        
        Returns
        -------
        float
        """
    def calculate_group_achievement(self, population_size: int, stationary_distribution: numpy.ndarray[numpy.float64[m, 1]]) -> float:
        """
        Calculates group achievement based on a stationary distribution.
        """
    def calculate_payoffs(self) -> numpy.ndarray[numpy.float64[m, n]]:
        """
        Computes the expected payoffs for each strategy across all group configurations.
        
        Returns
        -------
        numpy.ndarray
            Matrix of expected payoffs.
        """
    def calculate_polarization(self, population_size: int, population_state: numpy.ndarray[numpy.float64[m, 1]]) -> numpy.ndarray[numpy.float64[3, 1]]:
        """
        Computes contribution polarization in a given population state.
        """
    def calculate_polarization_success(self, population_size: int, population_state: numpy.ndarray[numpy.float64[m, 1]]) -> numpy.ndarray[numpy.float64[3, 1]]:
        """
        Computes contribution polarization among successful groups.
        """
    def calculate_population_group_achievement(self, population_size: int, population_state: numpy.ndarray[numpy.uint64[m, 1]]) -> float:
        """
        Calculates group achievement for a given population state.
        """
    def nb_strategies(self) -> int:
        ...
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
    def payoffs(self) -> numpy.ndarray[numpy.float64[m, n]]:
        """
        Returns the matrix of expected payoffs.
        """
    def play(self, arg0: list[int], arg1: list[float]) -> None:
        """
        Executes one iteration of the CRD game using a specific group composition.
        
        Parameters
        ----------
        group_composition : numpy.ndarray
            Number of players per strategy in the group.
        game_payoffs : numpy.ndarray
            Output vector for player payoffs.
        """
    def save_payoffs(self, arg0: str) -> None:
        """
        Saves the payoff matrix to a text file.
        
        Parameters
        ----------
        file_name : str
            Path to the output file.
        """
    def type(self) -> str:
        ...
    @property
    def endowment(self) -> int:
        ...
    @property
    def group_size(self) -> int:
        ...
    @property
    def min_rounds(self) -> int:
        ...
    @property
    def nb_states(self) -> int:
        ...
    @property
    def risk(self) -> float:
        ...
    @property
    def strategies(self) -> list:
        """
        List of strategy objects participating in the game.
        """
    @property
    def target(self) -> int:
        ...
class Matrix2PlayerGameHolder(AbstractGame):
    def __init__(self, nb_strategies: int, payoff_matrix: numpy.ndarray[numpy.float64[m, n], numpy.ndarray.flags.c_contiguous]) -> None:
        """
        Matrix-based 2-player game holder.
        
        Parameters
        ----------
        nb_strategies : int
            Number of strategies used in the game.
        payoff_matrix : numpy.ndarray
            Matrix containing the payoff of each strategy against all others.
        """
    def __str__(self) -> str:
        ...
    def calculate_fitness(self, player_strategy: int, pop_size: int, population_state: numpy.ndarray[numpy.uint64[m, 1]]) -> float:
        """
        Computes the fitness of a strategy given the population configuration.
        
        Parameters
        ----------
        player_strategy : int
            Index of the focal strategy.
        pop_size : int
            Size of the population.
        population_state : numpy.ndarray
            Vector of counts of each strategy in the population.
        
        Returns
        -------
        float
        """
    def calculate_payoffs(self) -> numpy.ndarray[numpy.float64[m, n]]:
        """
        Returns the stored payoff matrix.
        
        Returns
        -------
        numpy.ndarray
            Payoff matrix.
        """
    def nb_strategies(self) -> int:
        ...
    def payoff(self, strategy: int, strategy_pair: list[int]) -> float:
        """
        Returns the payoff for a given strategy pair.
        """
    def payoffs(self) -> numpy.ndarray[numpy.float64[m, n]]:
        """
        Returns the expected payoff matrix.
        """
    def play(self, arg0: list[int], arg1: list[float]) -> None:
        """
        Executes a match given a group composition and stores the resulting payoffs.
        
        Parameters
        ----------
        group_composition : numpy.ndarray
            Count of each strategy in the group.
        game_payoffs : numpy.ndarray
            Output vector to be filled with each player's payoff.
        """
    def save_payoffs(self, arg0: str) -> None:
        """
        Saves the current payoff matrix to a text file.
        """
    def type(self) -> str:
        ...
    def update_payoff_matrix(self, payoff_matrix: numpy.ndarray[numpy.float64[m, n]]) -> None:
        """
        Replaces the internal payoff matrix with a new one.
        """
class MatrixNPlayerGameHolder(AbstractGame):
    def __init__(self, nb_strategies: int, group_size: int, payoff_matrix: numpy.ndarray[numpy.float64[m, n], numpy.ndarray.flags.c_contiguous]) -> None:
        """
        Matrix-based N-player game holder.
        
        Parameters
        ----------
        nb_strategies : int
            Number of strategies in the game.
        group_size : int
            Size of the interacting group.
        payoff_matrix : numpy.ndarray
            Matrix encoding payoffs for all strategy-group pairs.
        """
    def __str__(self) -> str:
        ...
    def calculate_fitness(self, player_strategy: int, pop_size: int, population_state: numpy.ndarray[numpy.uint64[m, 1]]) -> float:
        """
        Computes the fitness of a strategy based on the current population state.
        
        Parameters
        ----------
        player_strategy : int
            Index of the strategy used by the focal player.
        pop_size : int
            Population size.
        population_state : numpy.ndarray
            Vector of strategy counts in the population.
        
        Returns
        -------
        float
            Fitness of the focal strategy.
        """
    def calculate_payoffs(self) -> numpy.ndarray[numpy.float64[m, n]]:
        """
        Returns the internal matrix of precomputed payoffs.
        
        Returns
        -------
        numpy.ndarray
        """
    def group_size(self) -> int:
        ...
    def nb_group_configurations(self) -> int:
        ...
    def nb_strategies(self) -> int:
        ...
    def payoff(self, strategy: int, strategy_pair: list[int]) -> float:
        """
        Returns the payoff for a strategy given a specific group configuration.
        """
    def payoffs(self) -> numpy.ndarray[numpy.float64[m, n]]:
        """
        Returns the full payoff matrix.
        """
    def play(self, arg0: list[int], arg1: list[float]) -> None:
        """
        Simulates the game based on a predefined payoff matrix.
        
        Parameters
        ----------
        group_composition : numpy.ndarray
            Number of players using each strategy in the group.
        game_payoffs : numpy.ndarray
            Output vector for storing player payoffs.
        """
    def save_payoffs(self, arg0: str) -> None:
        """
        Saves the payoff matrix to a text file.
        """
    def type(self) -> str:
        ...
    def update_payoff_matrix(self, payoff_matrix: numpy.ndarray[numpy.float64[m, n]]) -> None:
        """
        Replaces the stored payoff matrix with a new one.
        """
class NPlayerStagHunt(AbstractGame):
    def __init__(self, group_size: int, cooperation_threshold: int, enhancement_factor: float, cost: float) -> None:
        """
        N-player stag hunt.
        
        Parameters
        ----------
        group_size : int
            Number of players in the group.
        cooperation_threshold : int
            Minimum number of cooperators required to produce the collective benefit.
        enhancement_factor : float
            Multiplicative factor applied to the benefit when the public good is provided.
        cost : float
            Cost of cooperation.
        """
    def __str__(self) -> str:
        ...
    def calculate_fitness(self, player_strategy: int, pop_size: int, population_state: numpy.ndarray[numpy.uint64[m, 1]]) -> float:
        """
        Computes the fitness of a strategy given a population state.
        
        Parameters
        ----------
        player_strategy : int
            Index of the focal strategy.
        pop_size : int
            Total number of individuals in the population.
        population_state : numpy.ndarray
            Vector of strategy counts.
        
        Returns
        -------
        float
        """
    def calculate_group_achievement(self, population_size: int, stationary_distribution: numpy.ndarray[numpy.float64[m, 1]]) -> float:
        """
        Computes the expected collective success weighted by a stationary distribution.
        """
    def calculate_payoffs(self) -> numpy.ndarray[numpy.float64[m, n]]:
        """
        Computes and stores the expected payoff matrix for all strategy-group combinations.
        """
    def calculate_population_group_achievement(self, population_size: int, population_state: numpy.ndarray[numpy.uint64[m, 1]]) -> float:
        """
        Estimates the likelihood that a random group from the population meets the cooperation threshold.
        """
    def nb_group_configurations(self) -> int:
        ...
    def nb_strategies(self) -> int:
        ...
    def payoff(self, strategy: int, strategy_pair: list[int]) -> float:
        """
        Returns the payoff of a strategy given a group composition.
        """
    def payoffs(self) -> numpy.ndarray[numpy.float64[m, n]]:
        """
        Returns the expected payoff matrix for all strategy combinations.
        """
    def play(self, arg0: list[int], arg1: list[float]) -> None:
        """
        Simulates the game and fills in the payoff vector for a given group composition.
        
        Parameters
        ----------
        group_composition : numpy.ndarray
            Number of players of each strategy in the group.
        game_payoffs : numpy.ndarray
            Output vector to store the resulting payoff for each player.
        """
    def save_payoffs(self, arg0: str) -> None:
        """
        Saves the payoff matrix to a text file.
        """
    def strategies(self) -> list[str]:
        ...
    def type(self) -> str:
        ...
    @property
    def cooperation_threshold(self) -> int:
        ...
    @property
    def cost(self) -> float:
        ...
    @property
    def enhancement_factor(self) -> float:
        ...
    @property
    def group_achievement_per_group(self) -> numpy.ndarray[numpy.int64[m, 1]]:
        ...
    @property
    def group_size(self) -> int:
        ...
class NormalFormGame(AbstractGame):
    @typing.overload
    def __init__(self, nb_rounds: int, payoff_matrix: numpy.ndarray[numpy.float64[m, n], numpy.ndarray.flags.c_contiguous]) -> None:
        """
        Normal-form game with repeated pairwise interactions.
        
        Parameters
        ----------
        nb_rounds : int
            Number of rounds played by each strategy pair.
        payoff_matrix : numpy.ndarray
            Payoff matrix where entry (i, j) gives the payoff of strategy i against j.
        """
    @typing.overload
    def __init__(self, nb_rounds: int, payoff_matrix: numpy.ndarray[numpy.float64[m, n], numpy.ndarray.flags.c_contiguous], strategies: list) -> None:
        """
        Normal-form game initialized with custom strategy classes.
        
        Parameters
        ----------
        nb_rounds : int
            Number of rounds in the repeated game.
        payoff_matrix : numpy.ndarray
            Payoff matrix.
        strategies : list[AbstractNFGStrategy]
            List of strategy instances.
        """
    def __str__(self) -> str:
        ...
    def calculate_fitness(self, player_strategy: int, population_size: int, population_state: numpy.ndarray[numpy.uint64[m, 1]]) -> float:
        """
        Computes the fitness of a strategy in a given population state.
        
        Parameters
        ----------
        player_strategy : int
            Index of the focal strategy.
        population_size : int
            Total number of individuals.
        population_state : numpy.ndarray
            Strategy counts in the population.
        
        Returns
        -------
        float
            Fitness of the focal strategy.
        """
    def calculate_payoffs(self) -> numpy.ndarray[numpy.float64[m, n]]:
        """
        Calculates the expected payoff matrix for all strategy pairs.
        
        Returns
        -------
        numpy.ndarray
            Matrix of expected payoffs between strategies.
        """
    def expected_payoffs(self) -> numpy.ndarray[numpy.float64[m, n]]:
        """
        Returns the matrix of expected payoffs between strategies.
        """
    def nb_strategies(self) -> int:
        ...
    def payoff(self, strategy: int, strategy_pair: list[int]) -> float:
        """
        Returns the payoff for a given strategy in a specific match-up.
        
        Parameters
        ----------
        strategy : int
            Index of the strategy used by the player.
        strategy_pair : list[int]
            Pair of strategy indices in the match-up.
        
        Returns
        -------
        float
        """
    def payoffs(self) -> numpy.ndarray[numpy.float64[m, n]]:
        """
        Returns the payoff matrix.
        
        Returns
        -------
        numpy.ndarray
            Matrix of expected payoffs between strategies.
        """
    def play(self, group_composition: list[int], game_payoffs: list[float]) -> None:
        """
        Executes a game round and stores the resulting payoffs.
        
        Parameters
        ----------
        group_composition : numpy.ndarray
            Composition of the pairwise game.
        game_payoffs : numpy.ndarray
            Output array to store individual payoffs.
        """
    def save_payoffs(self, arg0: str) -> None:
        """
        Saves the payoff matrix to a text file.
        
        Parameters
        ----------
        file_name : str
            File path where the matrix will be saved.
        """
    def type(self) -> str:
        ...
    @property
    def nb_rounds(self) -> int:
        ...
    @property
    def nb_states(self) -> int:
        ...
    @property
    def strategies(self) -> list:
        """
        List of strategies participating in the game.
        """
class NormalFormNetworkGame(AbstractSpatialGame):
    @typing.overload
    def __init__(self, nb_rounds: int, payoff_matrix: numpy.ndarray[numpy.float64[m, n], numpy.ndarray.flags.c_contiguous]) -> None:
        """
        Normal-form network game.
        
        Parameters
        ----------
        nb_rounds : int
            Number of repeated rounds for each pairwise encounter.
        payoff_matrix : numpy.ndarray
            Payoff matrix specifying the outcomes for all strategy pairs.
        """
    @typing.overload
    def __init__(self, nb_rounds: int, payoff_matrix: numpy.ndarray[numpy.float64[m, n], numpy.ndarray.flags.c_contiguous], strategies: list) -> None:
        """
        Normal-form game with custom strategy list.
        
        Parameters
        ----------
        nb_rounds : int
            Number of rounds of interaction.
        payoff_matrix : numpy.ndarray
            Payoff matrix used for each pairwise encounter.
        strategies : list[AbstractNFGStrategy]
            List of strategy instances.
        """
    def __str__(self) -> str:
        ...
    def calculate_cooperation_level_neighborhood(self, strategy_index: int, state: numpy.ndarray[numpy.uint64[m, 1]]) -> float:
        """
        Calculates the level of cooperation in a given neighborhood.
        
        Parameters
        ----------
        strategy_index : int
            Focal strategy.
        state : numpy.ndarray
            Neighbor strategy counts.
        
        Returns
        -------
        float
            Level of cooperation.
        """
    def calculate_fitness(self, strategy_index: int, state: numpy.ndarray[numpy.uint64[m, 1]]) -> float:
        """
        Computes fitness of a strategy given a neighborhood configuration.
        
        Parameters
        ----------
        strategy_index : int
            Strategy whose fitness will be computed.
        state : numpy.ndarray
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
    def expected_payoffs(self) -> numpy.ndarray[numpy.float64[m, n]]:
        """
        Returns the expected payoffs for each strategy.
        """
    def nb_rounds(self) -> int:
        ...
    def nb_strategies(self) -> int:
        ...
    def strategies(self) -> list:
        """
        List of strategies currently active in the game.
        """
    def type(self) -> str:
        ...
class OneShotCRD(AbstractGame):
    def __init__(self, endowment: float, cost: float, risk: float, group_size: int, min_nb_cooperators: int) -> None:
        """
        One-shot collective risk dilemma.
        
        Parameters
        ----------
        endowment : float
            Initial endowment received by all players.
        cost : float
            Fraction of the endowment contributed by cooperators.
        risk : float
            Probability of collective loss if the group fails to reach the threshold.
        group_size : int
            Number of players in the group.
        min_nb_cooperators : int
            Minimum number of cooperators needed to avoid risk.
        """
    def __str__(self) -> str:
        ...
    def calculate_fitness(self, player_strategy: int, pop_size: int, population_state: numpy.ndarray[numpy.uint64[m, 1]]) -> float:
        """
        Calculates the fitness of a strategy given a population state.
        
        Parameters
        ----------
        player_strategy : int
            Index of the focal strategy.
        pop_size : int
            Population size.
        population_state : numpy.ndarray
            Vector of strategy counts in the population.
        
        Returns
        -------
        float
        """
    def calculate_group_achievement(self, population_size: int, stationary_distribution: numpy.ndarray[numpy.float64[m, 1]]) -> float:
        """
        Computes group achievement from a stationary distribution.
        """
    def calculate_payoffs(self) -> numpy.ndarray[numpy.float64[m, n]]:
        """
        Updates the payoff matrix and cooperation level matrix for all strategy pairs.
        """
    def calculate_population_group_achievement(self, population_size: int, population_state: numpy.ndarray[numpy.uint64[m, 1]]) -> float:
        """
        Computes the group achievement for the given population state.
        """
    def nb_strategies(self) -> int:
        ...
    def payoff(self, strategy: int, strategy_pair: list[int]) -> float:
        """
        Returns the payoff for a given strategy and group composition.
        """
    def payoffs(self) -> numpy.ndarray[numpy.float64[m, n]]:
        """
        Returns the expected payoff matrix.
        """
    def play(self, arg0: list[int], arg1: list[float]) -> None:
        """
        Executes a one-shot CRD round and updates payoffs for the given group composition.
        
        Parameters
        ----------
        group_composition : numpy.ndarray
            Number of players per strategy.
        game_payoffs : numpy.ndarray
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
        ...
    @property
    def endowment(self) -> float:
        ...
    @property
    def group_achievement_per_group(self) -> numpy.ndarray[numpy.int64[m, 1]]:
        ...
    @property
    def group_size(self) -> int:
        ...
    @property
    def min_nb_cooperators(self) -> int:
        ...
    @property
    def nb_states(self) -> int:
        ...
    @property
    def risk(self) -> float:
        ...
class OneShotCRDNetworkGame(AbstractSpatialGame):
    def __init__(self, endowment: float, cost: float, risk: float, min_nb_cooperators: int) -> None:
        """
        One-shot collective risk dilemma in networks.
        
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
        ...
    def calculate_fitness(self, strategy_index: int, state: numpy.ndarray[numpy.uint64[m, 1]]) -> float:
        """
        Computes the fitness of a strategy in a local neighborhood.
        
        Parameters
        ----------
        strategy_index : int
            The focal strategy being evaluated.
        state : numpy.ndarray
            Vector representing the number of neighbors using each strategy.
        
        Returns
        -------
        float
            The fitness of the strategy given the local state.
        """
    def cost(self) -> float:
        ...
    def endowment(self) -> float:
        ...
    def min_nb_cooperators(self) -> int:
        ...
    def nb_strategies(self) -> int:
        ...
    def risk(self) -> float:
        ...
    def type(self) -> str:
        ...
__init__: str = 'The `egttools.numerical.games` submodule contains the available games.'

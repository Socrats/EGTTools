from __future__ import annotations
__all__: list[str] = ['AbstractNetworkStructure', 'AbstractStructure', 'Network', 'NetworkGroup', 'NetworkGroupSync', 'NetworkSync']
class AbstractNetworkStructure(AbstractStructure):
    """
    
    Abstract base class for network-structured populations.
    
    This class extends `AbstractStructure` for populations represented as nodes in a
    network, where edges define who interacts with whom.
    
    Subclasses must implement at least the following methods:
    
    - `initialize()`
    - `initialize_state(state)`
    - `update_population()`
    - `calculate_average_gradient_of_selection()`
    - `mean_population_state()`
    - `nb_strategies()`
    - `population_size()`
    """
    @staticmethod
    def __init__(*args, **kwargs):
        ...
    @staticmethod
    def calculate_average_gradient_of_selection(*args, **kwargs):
        """
        
        Calculate the average gradient of selection at the current network state.
        
        Returns
        -------
        numpy.ndarray
            Averaged gradient of selection for each strategy.
        """
    @staticmethod
    def calculate_average_gradient_of_selection_and_update_population(*args, **kwargs):
        """
        
        Calculate the average gradient of selection and update the population.
        
        Returns
        -------
        numpy.ndarray
            Averaged gradient of selection for each strategy.
        """
    @staticmethod
    def initialize_state(*args, **kwargs):
        """
        
        Initialize the population at a specified aggregate state.
        
        Parameters
        ----------
        state : numpy.ndarray
            One-dimensional array containing the counts of each strategy in the population.
        """
    @staticmethod
    def network(*args, **kwargs):
        """
        
        Return the network adjacency structure.
        
        Returns
        -------
        dict[int, list[int]]
            Mapping from each node to its neighbors.
        """
    @staticmethod
    def population_size(*args, **kwargs):
        """
        
        Return the population size.
        
        Returns
        -------
        int
            Number of nodes in the network.
        """
    @staticmethod
    def update_node(*args, **kwargs):
        """
        
        Update the strategy of a given node.
        
        Parameters
        ----------
        node : int
            Index of the node to update.
        """
class AbstractStructure:
    """
    
    Abstract base class for population structures.
    
    This class defines the common interface for structures that contain a population
    and update the behavior of individuals over time.
    
    Subclasses must implement at least the following methods:
    
    - `initialize()`
    - `update_population()`
    - `mean_population_state()`
    - `nb_strategies()`
    """
    @staticmethod
    def __init__(*args, **kwargs):
        ...
    @staticmethod
    def initialize(*args, **kwargs):
        """
        
        Initialize the population.
        
        In evolutionary games, this usually means assigning an initial strategy to each
        individual.
        """
    @staticmethod
    def mean_population_state(*args, **kwargs):
        """
        
        Return the current aggregate population state.
        
        Returns
        -------
        numpy.ndarray
            Total counts of each strategy in the population.
        """
    @staticmethod
    def nb_strategies(*args, **kwargs):
        """
        
        Return the maximum number of strategies that can be present in the population.
        
        Returns
        -------
        int
            Number of strategies.
        """
    @staticmethod
    def update_population(*args, **kwargs):
        """
        
        Update the population by one generation.
        """
class Network(AbstractNetworkStructure):
    """
    
    Asynchronous network population structure with pairwise imitation updates.
    """
    @staticmethod
    def __init__(*args, **kwargs):
        """
        
        Construct a network structure.
        
        Parameters
        ----------
        nb_strategies : int
            Maximum number of strategies in the population.
        beta : float
            Intensity of selection.
        mu : float
            Mutation probability.
        network : dict[int, list[int]]
            Network adjacency dictionary.
        game : egttools.games.AbstractSpatialGame
            Spatial game played by the population.
        cache_size : int, optional
            Cache size used for fitness evaluations.
        """
    @staticmethod
    def calculate_average_gradient_of_selection(*args, **kwargs):
        """
        
        Calculate the average gradient of selection at the current network state.
        
        Returns
        -------
        numpy.ndarray
            Averaged gradient of selection for each strategy.
        """
    @staticmethod
    def calculate_average_gradient_of_selection_and_update_population(*args, **kwargs):
        """
        
        Calculate the average gradient of selection and update the population.
        
        Returns
        -------
        numpy.ndarray
            Averaged gradient of selection for each strategy.
        """
    @staticmethod
    def calculate_fitness(*args, **kwargs):
        """
        
        Calculate the fitness of the individual at a given node.
        
        Parameters
        ----------
        index : int
            Index of the node whose fitness is calculated.
        
        Returns
        -------
        float
            Fitness of the individual at the given node.
        """
    @staticmethod
    def game(*args, **kwargs):
        """
        
        Return the game played by the population.
        
        Returns
        -------
        egttools.games.AbstractSpatialGame
            Game used to evaluate fitness.
        """
    @staticmethod
    def initialize(*args, **kwargs):
        """
        
        Initialize the population.
        
        Each individual adopts one of the available strategies with approximately equal
        probability.
        """
    @staticmethod
    def initialize_state(*args, **kwargs):
        """
        
        Initialize the population at a specified aggregate state.
        
        Parameters
        ----------
        state : numpy.ndarray
            One-dimensional array containing the counts of each strategy in the population.
        """
    @staticmethod
    def mean_population_state(*args, **kwargs):
        """
        
        Return the aggregate population state.
        
        Returns
        -------
        numpy.ndarray
            Total counts of each strategy in the population.
        """
    @staticmethod
    def nb_strategies(*args, **kwargs):
        ...
    @staticmethod
    def network(*args, **kwargs):
        """
        
        Return the network adjacency structure.
        
        Returns
        -------
        dict[int, list[int]]
            Mapping from each node to its neighbors.
        """
    @staticmethod
    def population_size(*args, **kwargs):
        ...
    @staticmethod
    def population_strategies(*args, **kwargs):
        """
        
        Return the strategy currently adopted by each node.
        
        Returns
        -------
        list[int]
            Strategy index for each node.
        """
    @staticmethod
    def update_node(*args, **kwargs):
        """
        
        Update the strategy of a given node.
        
        Parameters
        ----------
        node : int
            Index of the node to update.
        """
    @staticmethod
    def update_population(*args, **kwargs):
        """
        
        Update the population by one generation.
        """
class NetworkGroup(AbstractNetworkStructure):
    """
    
    Asynchronous network-group population structure with group interactions.
    """
    @staticmethod
    def __init__(*args, **kwargs):
        """
        
        Construct a network-group structure.
        
        Parameters
        ----------
        nb_strategies : int
            Maximum number of strategies in the population.
        beta : float
            Intensity of selection.
        mu : float
            Mutation probability.
        network : dict[int, list[int]]
            Network adjacency dictionary.
        game : egttools.games.AbstractSpatialGame
            Spatial game played by the population.
        cache_size : int, optional
            Cache size used for fitness evaluations.
        """
    @staticmethod
    def calculate_average_gradient_of_selection(*args, **kwargs):
        ...
    @staticmethod
    def calculate_average_gradient_of_selection_and_update_population(*args, **kwargs):
        ...
    @staticmethod
    def calculate_fitness(*args, **kwargs):
        """
        
        Calculate the fitness of the individual at a given node.
        
        The fitness is the accumulated payoff over the focal interaction and the
        interactions centered on neighboring nodes.
        
        Parameters
        ----------
        index : int
            Index of the node whose fitness is calculated.
        
        Returns
        -------
        float
            Fitness of the individual at the given node.
        """
    @staticmethod
    def calculate_game_payoff(*args, **kwargs):
        """
        
        Calculate the game payoff of the individual at a given node.
        
        Parameters
        ----------
        index : int
            Index of the node whose payoff is calculated.
        
        Returns
        -------
        float
            Payoff of the individual at the given node.
        """
    @staticmethod
    def game(*args, **kwargs):
        ...
    @staticmethod
    def initialize(*args, **kwargs):
        ...
    @staticmethod
    def initialize_state(*args, **kwargs):
        ...
    @staticmethod
    def mean_population_state(*args, **kwargs):
        ...
    @staticmethod
    def nb_strategies(*args, **kwargs):
        ...
    @staticmethod
    def network(*args, **kwargs):
        ...
    @staticmethod
    def population_size(*args, **kwargs):
        ...
    @staticmethod
    def population_strategies(*args, **kwargs):
        ...
    @staticmethod
    def update_node(*args, **kwargs):
        ...
    @staticmethod
    def update_population(*args, **kwargs):
        ...
class NetworkGroupSync(AbstractNetworkStructure):
    """
    
    Synchronous network-group population structure with group interactions.
    """
    @staticmethod
    def __init__(*args, **kwargs):
        ...
    @staticmethod
    def calculate_average_gradient_of_selection(*args, **kwargs):
        ...
    @staticmethod
    def calculate_average_gradient_of_selection_and_update_population(*args, **kwargs):
        ...
    @staticmethod
    def calculate_fitness(*args, **kwargs):
        """
        
        Calculate the fitness of the individual at a given node.
        
        The fitness is the accumulated payoff over the focal interaction and the
        interactions centered on neighboring nodes.
        
        Parameters
        ----------
        index : int
            Index of the node whose fitness is calculated.
        
        Returns
        -------
        float
            Fitness of the individual at the given node.
        """
    @staticmethod
    def calculate_game_payoff(*args, **kwargs):
        """
        
        Calculate the game payoff of the individual at a given node.
        
        Parameters
        ----------
        index : int
            Index of the node whose payoff is calculated.
        
        Returns
        -------
        float
            Payoff of the individual at the given node.
        """
    @staticmethod
    def game(*args, **kwargs):
        ...
    @staticmethod
    def initialize(*args, **kwargs):
        ...
    @staticmethod
    def initialize_state(*args, **kwargs):
        ...
    @staticmethod
    def mean_population_state(*args, **kwargs):
        ...
    @staticmethod
    def nb_strategies(*args, **kwargs):
        ...
    @staticmethod
    def network(*args, **kwargs):
        ...
    @staticmethod
    def population_size(*args, **kwargs):
        ...
    @staticmethod
    def population_strategies(*args, **kwargs):
        ...
    @staticmethod
    def update_node(*args, **kwargs):
        ...
    @staticmethod
    def update_population(*args, **kwargs):
        ...
class NetworkSync(AbstractNetworkStructure):
    """
    
    Synchronous network population structure with pairwise imitation updates.
    """
    @staticmethod
    def __init__(*args, **kwargs):
        ...
    @staticmethod
    def calculate_average_gradient_of_selection(*args, **kwargs):
        ...
    @staticmethod
    def calculate_average_gradient_of_selection_and_update_population(*args, **kwargs):
        ...
    @staticmethod
    def calculate_fitness(*args, **kwargs):
        """
        
        Calculate the fitness of the individual at a given node.
        
        Parameters
        ----------
        index : int
            Index of the node whose fitness is calculated.
        
        Returns
        -------
        float
            Fitness of the individual at the given node.
        """
    @staticmethod
    def game(*args, **kwargs):
        ...
    @staticmethod
    def initialize(*args, **kwargs):
        ...
    @staticmethod
    def initialize_state(*args, **kwargs):
        ...
    @staticmethod
    def mean_population_state(*args, **kwargs):
        ...
    @staticmethod
    def nb_strategies(*args, **kwargs):
        ...
    @staticmethod
    def network(*args, **kwargs):
        ...
    @staticmethod
    def population_size(*args, **kwargs):
        ...
    @staticmethod
    def population_strategies(*args, **kwargs):
        ...
    @staticmethod
    def update_node(*args, **kwargs):
        ...
    @staticmethod
    def update_population(*args, **kwargs):
        ...
__init__: str = 'The `egttools.numerical.structure` submodule contains population structures.'

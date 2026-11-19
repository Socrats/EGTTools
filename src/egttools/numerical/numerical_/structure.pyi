from __future__ import annotations
import collections.abc
import egttools.numerical.numerical_.games
import numpy
import numpy.typing
import typing
__all__ = ['AbstractNetworkStructure', 'AbstractStructure', 'NetworkCoEvolutionaryPC', 'NetworkCoEvolutionaryPCHomophilic', 'NetworkMCEstimatorBD', 'NetworkMCEstimatorDB', 'NetworkMCEstimatorLP', 'NetworkMCEstimatorPC', 'NetworkMCEstimatorTDPC', 'run_network_sweep']
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
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    def __init__(self) -> None:
        ...
    def calculate_average_gradient_of_selection(self) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"]:
        """
        Calculate the average gradient of selection at the current network state.
        
        Returns
        -------
        numpy.ndarray
            Averaged gradient of selection for each strategy.
        """
    def calculate_average_gradient_of_selection_and_update_population(self) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"]:
        """
        Calculate the average gradient of selection and update the population.
        
        Returns
        -------
        numpy.ndarray
            Averaged gradient of selection for each strategy.
        """
    def initialize_state(self, state: typing.Annotated[numpy.typing.ArrayLike, numpy.uint64, "[m, 1]"]) -> None:
        """
        Initialize the population at a specified aggregate state.
        
        Parameters
        ----------
        state : numpy.ndarray
            One-dimensional array containing the counts of each strategy in the population.
        """
    def network(self) -> list[list[int]]:
        """
        Return the network adjacency structure.
        
        Returns
        -------
        list[list[int]]
            Adjacency list: entry i contains the neighbor node indices of node i.
        """
    def population_size(self) -> int:
        """
        Return the population size.
        
        Returns
        -------
        int
            Number of nodes in the network.
        """
    def update_node(self, node: typing.SupportsInt | typing.SupportsIndex) -> None:
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
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    def __init__(self) -> None:
        ...
    def initialize(self) -> None:
        """
        Initialize the population.
        
        In evolutionary games, this usually means assigning an initial strategy to each
        individual.
        """
    def mean_population_state(self) -> typing.Annotated[numpy.typing.NDArray[numpy.uint64], "[m, 1]"]:
        """
        Return the current aggregate population state.
        
        Returns
        -------
        numpy.ndarray
            Total counts of each strategy in the population.
        """
    def nb_strategies(self) -> int:
        """
        Return the maximum number of strategies that can be present in the population.
        
        Returns
        -------
        int
            Number of strategies.
        """
    def update_population(self) -> None:
        """
        Update the population by one generation.
        """
class NetworkCoEvolutionaryPC:
    """
    
    Co-evolutionary network estimator: Pairwise Comparison updates + Random rewiring.
    
    Each time step either rewires an edge (with probability ``rewiring_probability``) or
    performs a pairwise-comparison strategy update. This is the Santos & Pacheco (2006) model.
    """
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    def __init__(self, game: egttools.numerical.numerical_.games.AbstractSpatialGame, topology: collections.abc.Mapping[typing.SupportsInt | typing.SupportsIndex, collections.abc.Sequence[typing.SupportsInt | typing.SupportsIndex]], nb_strategies: typing.SupportsInt | typing.SupportsIndex, beta: typing.SupportsFloat | typing.SupportsIndex, mu: typing.SupportsFloat | typing.SupportsIndex, rewiring_probability: typing.SupportsFloat | typing.SupportsIndex, cache_size: typing.SupportsInt | typing.SupportsIndex = 100000) -> None:
        """
        Construct a NetworkCoEvolutionary estimator.
        
        Parameters
        ----------
        game : egttools.games.AbstractSpatialGame
        topology : dict[int, list[int]]
        nb_strategies : int
        beta : float
        mu : float
            Mutation probability.
        rewiring_probability : float
            Probability that a time step is a rewiring event rather than a strategy update.
        cache_size : int, optional
        """
    def beta(self) -> float:
        ...
    def estimate_fixation_probability(self, invader: typing.SupportsInt | typing.SupportsIndex, resident: typing.SupportsInt | typing.SupportsIndex, nb_runs: typing.SupportsInt | typing.SupportsIndex, nb_generations: typing.SupportsInt | typing.SupportsIndex) -> float:
        """
        Estimate the fixation probability of a single invader in an otherwise resident population.
        """
    def estimate_strategy_distribution(self, nb_runs: typing.SupportsInt | typing.SupportsIndex, nb_generations: typing.SupportsInt | typing.SupportsIndex, transitory: typing.SupportsInt | typing.SupportsIndex, tolerance: typing.SupportsFloat | typing.SupportsIndex = 0.0, check_every: typing.SupportsInt | typing.SupportsIndex = 0) -> tuple[typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"], typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"], float, float]:
        """
        Estimate time-averaged strategy frequencies and edge homophily after the transitory period.
        
        Returns
        -------
        tuple[numpy.ndarray, numpy.ndarray, float, float]
            (mean_frequencies, se_frequencies, mean_homophily, se_homophily)
        """
    def initial_topology(self) -> list[list[int]]:
        ...
    def mu(self) -> float:
        ...
    def nb_strategies(self) -> int:
        ...
    def population_size(self) -> int:
        ...
    def rewiring_probability(self) -> float:
        ...
    def run(self, nb_generations: typing.SupportsInt | typing.SupportsIndex, transitory: typing.SupportsInt | typing.SupportsIndex, init_state: typing.Annotated[numpy.typing.ArrayLike, numpy.uint64, "[m, 1]"]) -> typing.Annotated[numpy.typing.NDArray[numpy.uint64], "[m, n]"]:
        """
        Run a single co-evolutionary trajectory and return aggregate strategy counts per generation.
        """
    def run_snapshots(self, nb_generations: typing.SupportsInt | typing.SupportsIndex, transitory: typing.SupportsInt | typing.SupportsIndex, snapshot_interval: typing.SupportsInt | typing.SupportsIndex, init_state: typing.Annotated[numpy.typing.ArrayLike, numpy.uint64, "[m, 1]"], strategy_callback: typing.Any, topology_interval: typing.SupportsInt | typing.SupportsIndex = 0, topology_callback: typing.Any = None) -> None:
        """
        Run a co-evolutionary trajectory, calling callbacks at each snapshot.
        
        Parameters
        ----------
        strategy_callback : callable
            Called as ``strategy_callback(generation, population)`` every ``snapshot_interval`` gens.
        topology_interval : int, optional
            Call topology_callback every this many generations (0 = never).
        topology_callback : callable, optional
            Called as ``topology_callback(generation, adjacency_list)``.
        """
    def set_beta(self, beta: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    def set_mu(self, mu: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    def set_rewiring_probability(self, p: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
class NetworkCoEvolutionaryPCHomophilic:
    """
    
    Co-evolutionary network estimator: Pairwise Comparison updates + Homophilic rewiring.
    
    When a rewiring event occurs, the focal node severs a link to a different-strategy neighbour
    and preferentially reconnects to a same-strategy non-neighbour. Models social polarisation
    (Borges et al. 2023).
    """
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    def __init__(self, game: egttools.numerical.numerical_.games.AbstractSpatialGame, topology: collections.abc.Mapping[typing.SupportsInt | typing.SupportsIndex, collections.abc.Sequence[typing.SupportsInt | typing.SupportsIndex]], nb_strategies: typing.SupportsInt | typing.SupportsIndex, beta: typing.SupportsFloat | typing.SupportsIndex, mu: typing.SupportsFloat | typing.SupportsIndex, rewiring_probability: typing.SupportsFloat | typing.SupportsIndex, cache_size: typing.SupportsInt | typing.SupportsIndex = 100000) -> None:
        """
        Construct a NetworkCoEvolutionary estimator.
        
        Parameters
        ----------
        game : egttools.games.AbstractSpatialGame
        topology : dict[int, list[int]]
        nb_strategies : int
        beta : float
        mu : float
            Mutation probability.
        rewiring_probability : float
            Probability that a time step is a rewiring event rather than a strategy update.
        cache_size : int, optional
        """
    def beta(self) -> float:
        ...
    def estimate_fixation_probability(self, invader: typing.SupportsInt | typing.SupportsIndex, resident: typing.SupportsInt | typing.SupportsIndex, nb_runs: typing.SupportsInt | typing.SupportsIndex, nb_generations: typing.SupportsInt | typing.SupportsIndex) -> float:
        """
        Estimate the fixation probability of a single invader in an otherwise resident population.
        """
    def estimate_strategy_distribution(self, nb_runs: typing.SupportsInt | typing.SupportsIndex, nb_generations: typing.SupportsInt | typing.SupportsIndex, transitory: typing.SupportsInt | typing.SupportsIndex, tolerance: typing.SupportsFloat | typing.SupportsIndex = 0.0, check_every: typing.SupportsInt | typing.SupportsIndex = 0) -> tuple[typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"], typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"], float, float]:
        """
        Estimate time-averaged strategy frequencies and edge homophily after the transitory period.
        
        Returns
        -------
        tuple[numpy.ndarray, numpy.ndarray, float, float]
            (mean_frequencies, se_frequencies, mean_homophily, se_homophily)
        """
    def initial_topology(self) -> list[list[int]]:
        ...
    def mu(self) -> float:
        ...
    def nb_strategies(self) -> int:
        ...
    def population_size(self) -> int:
        ...
    def rewiring_probability(self) -> float:
        ...
    def run(self, nb_generations: typing.SupportsInt | typing.SupportsIndex, transitory: typing.SupportsInt | typing.SupportsIndex, init_state: typing.Annotated[numpy.typing.ArrayLike, numpy.uint64, "[m, 1]"]) -> typing.Annotated[numpy.typing.NDArray[numpy.uint64], "[m, n]"]:
        """
        Run a single co-evolutionary trajectory and return aggregate strategy counts per generation.
        """
    def run_snapshots(self, nb_generations: typing.SupportsInt | typing.SupportsIndex, transitory: typing.SupportsInt | typing.SupportsIndex, snapshot_interval: typing.SupportsInt | typing.SupportsIndex, init_state: typing.Annotated[numpy.typing.ArrayLike, numpy.uint64, "[m, 1]"], strategy_callback: typing.Any, topology_interval: typing.SupportsInt | typing.SupportsIndex = 0, topology_callback: typing.Any = None) -> None:
        """
        Run a co-evolutionary trajectory, calling callbacks at each snapshot.
        
        Parameters
        ----------
        strategy_callback : callable
            Called as ``strategy_callback(generation, population)`` every ``snapshot_interval`` gens.
        topology_interval : int, optional
            Call topology_callback every this many generations (0 = never).
        topology_callback : callable, optional
            Called as ``topology_callback(generation, adjacency_list)``.
        """
    def set_beta(self, beta: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    def set_mu(self, mu: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    def set_rewiring_probability(self, p: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
class NetworkMCEstimatorBD:
    """
    
    Monte Carlo estimator for evolutionary games on networks using the Birth-Death update rule.
    
    In each step a node is selected to reproduce proportional to fitness, then replaces a
    random neighbour.
    """
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    def __init__(self, game: egttools.numerical.numerical_.games.AbstractSpatialGame, topology: collections.abc.Mapping[typing.SupportsInt | typing.SupportsIndex, collections.abc.Sequence[typing.SupportsInt | typing.SupportsIndex]], nb_strategies: typing.SupportsInt | typing.SupportsIndex, beta: typing.SupportsFloat | typing.SupportsIndex, mu: typing.SupportsFloat | typing.SupportsIndex, cache_size: typing.SupportsInt | typing.SupportsIndex = 100000, seed: typing.SupportsInt | typing.SupportsIndex | None = None) -> None:
        """
        Construct a NetworkMCEstimator.
        
        Parameters
        ----------
        game : egttools.games.AbstractSpatialGame
            Spatial game used to evaluate node fitness.
        topology : dict[int, list[int]]
            Network adjacency dictionary (e.g. from NetworkX ``G.adjacency()``).
        nb_strategies : int
            Number of distinct strategies.
        beta : float
            Selection intensity.
        mu : float
            Mutation probability per time step.
        cache_size : int, optional
            Per-thread LRU fitness cache size (default 100000).
        seed : int, optional
            If given, seeds the step-by-step session generator (used by
            ``initialize``/``step``) independently of the process-global
            ``egttools.Random`` singleton. Does not affect the OpenMP-parallel
            batched ``estimate_*`` methods. Omit (default) for prior behavior.
        """
    def beta(self) -> float:
        ...
    def calculate_gradient_of_selection(self, population: collections.abc.Sequence[typing.SupportsInt | typing.SupportsIndex]) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"]:
        """
        Numerically exact gradient of selection for the given per-node strategy assignment.
        
        Parameters
        ----------
        population : list[int]
            Strategy index for each node (length = population_size).
        
        Returns
        -------
        numpy.ndarray
            Gradient vector of length nb_strategies.
        """
    def estimate_agos(self, nb_runs: typing.SupportsInt | typing.SupportsIndex, nb_generations: typing.SupportsInt | typing.SupportsIndex, transitory: typing.SupportsInt | typing.SupportsIndex = 0, runs_per_j: typing.SupportsInt | typing.SupportsIndex = 0) -> tuple[typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]"], typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]"]]:
        """
        Estimate the time-independent Average Gradient of Selection G^A(j).
        
        Runs independent trajectories; at each post-transitory generation computes
        the numerically exact gradient and bins the result by cooperator count j.
        OpenMP-parallelised over runs; per-thread caches prevent contention.
        
        Parameters
        ----------
        nb_runs : int
            Total trajectories when ``runs_per_j == 0`` (random starts).
        nb_generations : int
            Generations (each = N elementary steps) per trajectory.
        transitory : int, optional
            Burn-in generations not counted (default 0).
        runs_per_j : int, optional
            When > 0, uses the paper's sampling scheme: for each initial cooperator
            count j0 ∈ {1, …, N-1} exactly ``runs_per_j`` trajectories are started
            with j0 cooperators on random nodes.  Total runs = runs_per_j × (N-1);
            ``nb_runs`` is ignored.  This gives uniform coverage of all j values.
        
        Returns
        -------
        tuple[numpy.ndarray, numpy.ndarray]
            (mean_G, se_G) each of shape (N+1, nb_strategies).
            Row j holds the average/SE gradient at cooperator count j.
            Rows 0 and N are zero (absorbing states).
        """
    def estimate_agos_time_dependent(self, nb_runs: typing.SupportsInt | typing.SupportsIndex, nb_generations: typing.SupportsInt | typing.SupportsIndex) -> tuple[numpy.typing.NDArray[numpy.float64], numpy.typing.NDArray[numpy.float64]]:
        """
        Estimate the time-dependent Average Gradient of Selection G^A(j, t).
        
        Like estimate_agos but preserves the generation index, letting you study how
        the gradient landscape evolves from the initial transient to the stationary
        regime (e.g. Fig. 2 of Pinheiro et al. 2012).
        
        Returns
        -------
        tuple[numpy.ndarray, numpy.ndarray]
            (mean_G_t, se_G_t) each of shape (nb_generations, N+1, nb_strategies).
            ``mean_G_t[t, j, k]`` is the mean gradient of strategy k at generation t
            and cooperator count j.
        """
    def estimate_fixation_probability(self, invader: typing.SupportsInt | typing.SupportsIndex, resident: typing.SupportsInt | typing.SupportsIndex, nb_runs: typing.SupportsInt | typing.SupportsIndex, nb_generations: typing.SupportsInt | typing.SupportsIndex) -> float:
        """
        Estimate the fixation probability of a single invader in an otherwise resident population.
        
        Parameters
        ----------
        invader : int
        resident : int
        nb_runs : int
        nb_generations : int
            Maximum time-steps per trial.
        
        Returns
        -------
        float
            Estimated fixation probability in [0, 1].
        """
    def estimate_strategy_distribution(self, nb_runs: typing.SupportsInt | typing.SupportsIndex, nb_generations: typing.SupportsInt | typing.SupportsIndex, transitory: typing.SupportsInt | typing.SupportsIndex, tolerance: typing.SupportsFloat | typing.SupportsIndex = 0.0, check_every: typing.SupportsInt | typing.SupportsIndex = 0) -> tuple[typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"], typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"]]:
        """
        Estimate time-averaged strategy frequencies after the transitory period.
        
        Returns
        -------
        tuple[numpy.ndarray, numpy.ndarray]
            (mean_frequencies, standard_errors), each of length nb_strategies.
        """
    @typing.overload
    def initialize(self, init_state: typing.Annotated[numpy.typing.ArrayLike, numpy.uint64, "[m, 1]"]) -> None:
        """
        Initialise a manual-stepping session with the given strategy counts.
        
        Parameters
        ----------
        init_state : numpy.ndarray
            Strategy count vector of length nb_strategies; sum must equal population_size.
        """
    @typing.overload
    def initialize(self) -> None:
        """
        Initialise a manual-stepping session with a uniformly random strategy assignment.
        """
    def mean_population_state(self) -> typing.Annotated[numpy.typing.NDArray[numpy.uint64], "[m, 1]"]:
        """
        Return strategy count vector (length = nb_strategies).
        """
    def mu(self) -> float:
        ...
    def nb_strategies(self) -> int:
        ...
    def population_size(self) -> int:
        ...
    def population_strategies(self) -> list[int]:
        """
        Return per-node strategy assignments as a list of integers (length = population_size).
        """
    def run(self, nb_generations: typing.SupportsInt | typing.SupportsIndex, transitory: typing.SupportsInt | typing.SupportsIndex, init_state: typing.Annotated[numpy.typing.ArrayLike, numpy.uint64, "[m, 1]"]) -> typing.Annotated[numpy.typing.NDArray[numpy.uint64], "[m, n]"]:
        """
        Run a single trajectory and return aggregate strategy counts per generation.
        
        Returns
        -------
        numpy.ndarray
            Matrix of shape (nb_generations - transitory, nb_strategies).
        """
    def run_snapshots(self, nb_generations: typing.SupportsInt | typing.SupportsIndex, transitory: typing.SupportsInt | typing.SupportsIndex, snapshot_interval: typing.SupportsInt | typing.SupportsIndex, init_state: typing.Annotated[numpy.typing.ArrayLike, numpy.uint64, "[m, 1]"], callback: typing.Any) -> None:
        """
        Run a trajectory and call ``callback(generation, population)`` at each snapshot.
        
        Parameters
        ----------
        snapshot_interval : int
            Call callback every this many generations after transitory.
        callback : callable
            Called as ``callback(generation: int, population: list[int])``.
        """
    def set_beta(self, beta: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    def set_mu(self, mu: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    def step(self) -> None:
        """
        Advance the session by one generation.
        
        For asynchronous rules: N individual update steps (one per node on average).
        For synchronous rules (e.g. LinearProportional): one full simultaneous sweep.
        Raises RuntimeError if initialize() has not been called first.
        """
    def topology(self) -> list[list[int]]:
        ...
class NetworkMCEstimatorDB:
    """
    
    Monte Carlo estimator for evolutionary games on networks using the Death-Birth update rule.
    
    In each step a random node dies, and a neighbour is selected to reproduce proportional to
    fitness.
    """
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    def __init__(self, game: egttools.numerical.numerical_.games.AbstractSpatialGame, topology: collections.abc.Mapping[typing.SupportsInt | typing.SupportsIndex, collections.abc.Sequence[typing.SupportsInt | typing.SupportsIndex]], nb_strategies: typing.SupportsInt | typing.SupportsIndex, beta: typing.SupportsFloat | typing.SupportsIndex, mu: typing.SupportsFloat | typing.SupportsIndex, cache_size: typing.SupportsInt | typing.SupportsIndex = 100000, seed: typing.SupportsInt | typing.SupportsIndex | None = None) -> None:
        """
        Construct a NetworkMCEstimator.
        
        Parameters
        ----------
        game : egttools.games.AbstractSpatialGame
            Spatial game used to evaluate node fitness.
        topology : dict[int, list[int]]
            Network adjacency dictionary (e.g. from NetworkX ``G.adjacency()``).
        nb_strategies : int
            Number of distinct strategies.
        beta : float
            Selection intensity.
        mu : float
            Mutation probability per time step.
        cache_size : int, optional
            Per-thread LRU fitness cache size (default 100000).
        seed : int, optional
            If given, seeds the step-by-step session generator (used by
            ``initialize``/``step``) independently of the process-global
            ``egttools.Random`` singleton. Does not affect the OpenMP-parallel
            batched ``estimate_*`` methods. Omit (default) for prior behavior.
        """
    def beta(self) -> float:
        ...
    def calculate_gradient_of_selection(self, population: collections.abc.Sequence[typing.SupportsInt | typing.SupportsIndex]) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"]:
        """
        Numerically exact gradient of selection for the given per-node strategy assignment.
        
        Parameters
        ----------
        population : list[int]
            Strategy index for each node (length = population_size).
        
        Returns
        -------
        numpy.ndarray
            Gradient vector of length nb_strategies.
        """
    def estimate_agos(self, nb_runs: typing.SupportsInt | typing.SupportsIndex, nb_generations: typing.SupportsInt | typing.SupportsIndex, transitory: typing.SupportsInt | typing.SupportsIndex = 0, runs_per_j: typing.SupportsInt | typing.SupportsIndex = 0) -> tuple[typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]"], typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]"]]:
        """
        Estimate the time-independent Average Gradient of Selection G^A(j).
        
        Runs independent trajectories; at each post-transitory generation computes
        the numerically exact gradient and bins the result by cooperator count j.
        OpenMP-parallelised over runs; per-thread caches prevent contention.
        
        Parameters
        ----------
        nb_runs : int
            Total trajectories when ``runs_per_j == 0`` (random starts).
        nb_generations : int
            Generations (each = N elementary steps) per trajectory.
        transitory : int, optional
            Burn-in generations not counted (default 0).
        runs_per_j : int, optional
            When > 0, uses the paper's sampling scheme: for each initial cooperator
            count j0 ∈ {1, …, N-1} exactly ``runs_per_j`` trajectories are started
            with j0 cooperators on random nodes.  Total runs = runs_per_j × (N-1);
            ``nb_runs`` is ignored.  This gives uniform coverage of all j values.
        
        Returns
        -------
        tuple[numpy.ndarray, numpy.ndarray]
            (mean_G, se_G) each of shape (N+1, nb_strategies).
            Row j holds the average/SE gradient at cooperator count j.
            Rows 0 and N are zero (absorbing states).
        """
    def estimate_agos_time_dependent(self, nb_runs: typing.SupportsInt | typing.SupportsIndex, nb_generations: typing.SupportsInt | typing.SupportsIndex) -> tuple[numpy.typing.NDArray[numpy.float64], numpy.typing.NDArray[numpy.float64]]:
        """
        Estimate the time-dependent Average Gradient of Selection G^A(j, t).
        
        Like estimate_agos but preserves the generation index, letting you study how
        the gradient landscape evolves from the initial transient to the stationary
        regime (e.g. Fig. 2 of Pinheiro et al. 2012).
        
        Returns
        -------
        tuple[numpy.ndarray, numpy.ndarray]
            (mean_G_t, se_G_t) each of shape (nb_generations, N+1, nb_strategies).
            ``mean_G_t[t, j, k]`` is the mean gradient of strategy k at generation t
            and cooperator count j.
        """
    def estimate_fixation_probability(self, invader: typing.SupportsInt | typing.SupportsIndex, resident: typing.SupportsInt | typing.SupportsIndex, nb_runs: typing.SupportsInt | typing.SupportsIndex, nb_generations: typing.SupportsInt | typing.SupportsIndex) -> float:
        """
        Estimate the fixation probability of a single invader in an otherwise resident population.
        
        Parameters
        ----------
        invader : int
        resident : int
        nb_runs : int
        nb_generations : int
            Maximum time-steps per trial.
        
        Returns
        -------
        float
            Estimated fixation probability in [0, 1].
        """
    def estimate_strategy_distribution(self, nb_runs: typing.SupportsInt | typing.SupportsIndex, nb_generations: typing.SupportsInt | typing.SupportsIndex, transitory: typing.SupportsInt | typing.SupportsIndex, tolerance: typing.SupportsFloat | typing.SupportsIndex = 0.0, check_every: typing.SupportsInt | typing.SupportsIndex = 0) -> tuple[typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"], typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"]]:
        """
        Estimate time-averaged strategy frequencies after the transitory period.
        
        Returns
        -------
        tuple[numpy.ndarray, numpy.ndarray]
            (mean_frequencies, standard_errors), each of length nb_strategies.
        """
    @typing.overload
    def initialize(self, init_state: typing.Annotated[numpy.typing.ArrayLike, numpy.uint64, "[m, 1]"]) -> None:
        """
        Initialise a manual-stepping session with the given strategy counts.
        
        Parameters
        ----------
        init_state : numpy.ndarray
            Strategy count vector of length nb_strategies; sum must equal population_size.
        """
    @typing.overload
    def initialize(self) -> None:
        """
        Initialise a manual-stepping session with a uniformly random strategy assignment.
        """
    def mean_population_state(self) -> typing.Annotated[numpy.typing.NDArray[numpy.uint64], "[m, 1]"]:
        """
        Return strategy count vector (length = nb_strategies).
        """
    def mu(self) -> float:
        ...
    def nb_strategies(self) -> int:
        ...
    def population_size(self) -> int:
        ...
    def population_strategies(self) -> list[int]:
        """
        Return per-node strategy assignments as a list of integers (length = population_size).
        """
    def run(self, nb_generations: typing.SupportsInt | typing.SupportsIndex, transitory: typing.SupportsInt | typing.SupportsIndex, init_state: typing.Annotated[numpy.typing.ArrayLike, numpy.uint64, "[m, 1]"]) -> typing.Annotated[numpy.typing.NDArray[numpy.uint64], "[m, n]"]:
        """
        Run a single trajectory and return aggregate strategy counts per generation.
        
        Returns
        -------
        numpy.ndarray
            Matrix of shape (nb_generations - transitory, nb_strategies).
        """
    def run_snapshots(self, nb_generations: typing.SupportsInt | typing.SupportsIndex, transitory: typing.SupportsInt | typing.SupportsIndex, snapshot_interval: typing.SupportsInt | typing.SupportsIndex, init_state: typing.Annotated[numpy.typing.ArrayLike, numpy.uint64, "[m, 1]"], callback: typing.Any) -> None:
        """
        Run a trajectory and call ``callback(generation, population)`` at each snapshot.
        
        Parameters
        ----------
        snapshot_interval : int
            Call callback every this many generations after transitory.
        callback : callable
            Called as ``callback(generation: int, population: list[int])``.
        """
    def set_beta(self, beta: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    def set_mu(self, mu: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    def step(self) -> None:
        """
        Advance the session by one generation.
        
        For asynchronous rules: N individual update steps (one per node on average).
        For synchronous rules (e.g. LinearProportional): one full simultaneous sweep.
        Raises RuntimeError if initialize() has not been called first.
        """
    def topology(self) -> list[list[int]]:
        ...
class NetworkMCEstimatorLP:
    """
    
    Monte Carlo estimator for evolutionary games on networks using the Linear-Proportional synchronous update rule.
    
    Implements the update rule from Santos, Pacheco & Lenaerts (2006) PNAS.
    Each generation all nodes simultaneously accumulate payoffs against all neighbours,
    then simultaneously copy a neighbour's strategy with probability proportional to
    the payoff advantage, normalised by max(degree) * D_>, where D_> = max(T,1) - min(S,0).
    
    Pass ``beta = max(T, 1.0) - min(S, 0.0)`` as the selection intensity.
    """
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    def __init__(self, game: egttools.numerical.numerical_.games.AbstractSpatialGame, topology: collections.abc.Mapping[typing.SupportsInt | typing.SupportsIndex, collections.abc.Sequence[typing.SupportsInt | typing.SupportsIndex]], nb_strategies: typing.SupportsInt | typing.SupportsIndex, beta: typing.SupportsFloat | typing.SupportsIndex, mu: typing.SupportsFloat | typing.SupportsIndex, cache_size: typing.SupportsInt | typing.SupportsIndex = 100000, seed: typing.SupportsInt | typing.SupportsIndex | None = None) -> None:
        """
        Construct a NetworkMCEstimator.
        
        Parameters
        ----------
        game : egttools.games.AbstractSpatialGame
            Spatial game used to evaluate node fitness.
        topology : dict[int, list[int]]
            Network adjacency dictionary (e.g. from NetworkX ``G.adjacency()``).
        nb_strategies : int
            Number of distinct strategies.
        beta : float
            Selection intensity.
        mu : float
            Mutation probability per time step.
        cache_size : int, optional
            Per-thread LRU fitness cache size (default 100000).
        seed : int, optional
            If given, seeds the step-by-step session generator (used by
            ``initialize``/``step``) independently of the process-global
            ``egttools.Random`` singleton. Does not affect the OpenMP-parallel
            batched ``estimate_*`` methods. Omit (default) for prior behavior.
        """
    def beta(self) -> float:
        ...
    def calculate_gradient_of_selection(self, population: collections.abc.Sequence[typing.SupportsInt | typing.SupportsIndex]) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"]:
        """
        Numerically exact gradient of selection for the given per-node strategy assignment.
        
        Parameters
        ----------
        population : list[int]
            Strategy index for each node (length = population_size).
        
        Returns
        -------
        numpy.ndarray
            Gradient vector of length nb_strategies.
        """
    def estimate_agos(self, nb_runs: typing.SupportsInt | typing.SupportsIndex, nb_generations: typing.SupportsInt | typing.SupportsIndex, transitory: typing.SupportsInt | typing.SupportsIndex = 0, runs_per_j: typing.SupportsInt | typing.SupportsIndex = 0) -> tuple[typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]"], typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]"]]:
        """
        Estimate the time-independent Average Gradient of Selection G^A(j).
        
        Runs independent trajectories; at each post-transitory generation computes
        the numerically exact gradient and bins the result by cooperator count j.
        OpenMP-parallelised over runs; per-thread caches prevent contention.
        
        Parameters
        ----------
        nb_runs : int
            Total trajectories when ``runs_per_j == 0`` (random starts).
        nb_generations : int
            Generations (each = N elementary steps) per trajectory.
        transitory : int, optional
            Burn-in generations not counted (default 0).
        runs_per_j : int, optional
            When > 0, uses the paper's sampling scheme: for each initial cooperator
            count j0 ∈ {1, …, N-1} exactly ``runs_per_j`` trajectories are started
            with j0 cooperators on random nodes.  Total runs = runs_per_j × (N-1);
            ``nb_runs`` is ignored.  This gives uniform coverage of all j values.
        
        Returns
        -------
        tuple[numpy.ndarray, numpy.ndarray]
            (mean_G, se_G) each of shape (N+1, nb_strategies).
            Row j holds the average/SE gradient at cooperator count j.
            Rows 0 and N are zero (absorbing states).
        """
    def estimate_agos_time_dependent(self, nb_runs: typing.SupportsInt | typing.SupportsIndex, nb_generations: typing.SupportsInt | typing.SupportsIndex) -> tuple[numpy.typing.NDArray[numpy.float64], numpy.typing.NDArray[numpy.float64]]:
        """
        Estimate the time-dependent Average Gradient of Selection G^A(j, t).
        
        Like estimate_agos but preserves the generation index, letting you study how
        the gradient landscape evolves from the initial transient to the stationary
        regime (e.g. Fig. 2 of Pinheiro et al. 2012).
        
        Returns
        -------
        tuple[numpy.ndarray, numpy.ndarray]
            (mean_G_t, se_G_t) each of shape (nb_generations, N+1, nb_strategies).
            ``mean_G_t[t, j, k]`` is the mean gradient of strategy k at generation t
            and cooperator count j.
        """
    def estimate_fixation_probability(self, invader: typing.SupportsInt | typing.SupportsIndex, resident: typing.SupportsInt | typing.SupportsIndex, nb_runs: typing.SupportsInt | typing.SupportsIndex, nb_generations: typing.SupportsInt | typing.SupportsIndex) -> float:
        """
        Estimate the fixation probability of a single invader in an otherwise resident population.
        
        Parameters
        ----------
        invader : int
        resident : int
        nb_runs : int
        nb_generations : int
            Maximum time-steps per trial.
        
        Returns
        -------
        float
            Estimated fixation probability in [0, 1].
        """
    def estimate_strategy_distribution(self, nb_runs: typing.SupportsInt | typing.SupportsIndex, nb_generations: typing.SupportsInt | typing.SupportsIndex, transitory: typing.SupportsInt | typing.SupportsIndex, tolerance: typing.SupportsFloat | typing.SupportsIndex = 0.0, check_every: typing.SupportsInt | typing.SupportsIndex = 0) -> tuple[typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"], typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"]]:
        """
        Estimate time-averaged strategy frequencies after the transitory period.
        
        Returns
        -------
        tuple[numpy.ndarray, numpy.ndarray]
            (mean_frequencies, standard_errors), each of length nb_strategies.
        """
    @typing.overload
    def initialize(self, init_state: typing.Annotated[numpy.typing.ArrayLike, numpy.uint64, "[m, 1]"]) -> None:
        """
        Initialise a manual-stepping session with the given strategy counts.
        
        Parameters
        ----------
        init_state : numpy.ndarray
            Strategy count vector of length nb_strategies; sum must equal population_size.
        """
    @typing.overload
    def initialize(self) -> None:
        """
        Initialise a manual-stepping session with a uniformly random strategy assignment.
        """
    def mean_population_state(self) -> typing.Annotated[numpy.typing.NDArray[numpy.uint64], "[m, 1]"]:
        """
        Return strategy count vector (length = nb_strategies).
        """
    def mu(self) -> float:
        ...
    def nb_strategies(self) -> int:
        ...
    def population_size(self) -> int:
        ...
    def population_strategies(self) -> list[int]:
        """
        Return per-node strategy assignments as a list of integers (length = population_size).
        """
    def run(self, nb_generations: typing.SupportsInt | typing.SupportsIndex, transitory: typing.SupportsInt | typing.SupportsIndex, init_state: typing.Annotated[numpy.typing.ArrayLike, numpy.uint64, "[m, 1]"]) -> typing.Annotated[numpy.typing.NDArray[numpy.uint64], "[m, n]"]:
        """
        Run a single trajectory and return aggregate strategy counts per generation.
        
        Returns
        -------
        numpy.ndarray
            Matrix of shape (nb_generations - transitory, nb_strategies).
        """
    def run_snapshots(self, nb_generations: typing.SupportsInt | typing.SupportsIndex, transitory: typing.SupportsInt | typing.SupportsIndex, snapshot_interval: typing.SupportsInt | typing.SupportsIndex, init_state: typing.Annotated[numpy.typing.ArrayLike, numpy.uint64, "[m, 1]"], callback: typing.Any) -> None:
        """
        Run a trajectory and call ``callback(generation, population)`` at each snapshot.
        
        Parameters
        ----------
        snapshot_interval : int
            Call callback every this many generations after transitory.
        callback : callable
            Called as ``callback(generation: int, population: list[int])``.
        """
    def set_beta(self, beta: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    def set_mu(self, mu: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    def step(self) -> None:
        """
        Advance the session by one generation.
        
        For asynchronous rules: N individual update steps (one per node on average).
        For synchronous rules (e.g. LinearProportional): one full simultaneous sweep.
        Raises RuntimeError if initialize() has not been called first.
        """
    def topology(self) -> list[list[int]]:
        ...
class NetworkMCEstimatorPC:
    """
    
    Monte Carlo estimator for evolutionary games on networks using the Pairwise Comparison update rule.
    
    Provides gradient of selection, fixation probability, strategy distribution estimation,
    and trajectory generation. Uses OpenMP for parallel runs and an LRU cache for fitness
    memoization.
    
    The network topology is stored as a contiguous adjacency list for O(1) neighbour lookup.
    """
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    def __init__(self, game: egttools.numerical.numerical_.games.AbstractSpatialGame, topology: collections.abc.Mapping[typing.SupportsInt | typing.SupportsIndex, collections.abc.Sequence[typing.SupportsInt | typing.SupportsIndex]], nb_strategies: typing.SupportsInt | typing.SupportsIndex, beta: typing.SupportsFloat | typing.SupportsIndex, mu: typing.SupportsFloat | typing.SupportsIndex, cache_size: typing.SupportsInt | typing.SupportsIndex = 100000, seed: typing.SupportsInt | typing.SupportsIndex | None = None) -> None:
        """
        Construct a NetworkMCEstimator.
        
        Parameters
        ----------
        game : egttools.games.AbstractSpatialGame
            Spatial game used to evaluate node fitness.
        topology : dict[int, list[int]]
            Network adjacency dictionary (e.g. from NetworkX ``G.adjacency()``).
        nb_strategies : int
            Number of distinct strategies.
        beta : float
            Selection intensity.
        mu : float
            Mutation probability per time step.
        cache_size : int, optional
            Per-thread LRU fitness cache size (default 100000).
        seed : int, optional
            If given, seeds the step-by-step session generator (used by
            ``initialize``/``step``) independently of the process-global
            ``egttools.Random`` singleton. Does not affect the OpenMP-parallel
            batched ``estimate_*`` methods. Omit (default) for prior behavior.
        """
    def beta(self) -> float:
        ...
    def calculate_gradient_of_selection(self, population: collections.abc.Sequence[typing.SupportsInt | typing.SupportsIndex]) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"]:
        """
        Numerically exact gradient of selection for the given per-node strategy assignment.
        
        Parameters
        ----------
        population : list[int]
            Strategy index for each node (length = population_size).
        
        Returns
        -------
        numpy.ndarray
            Gradient vector of length nb_strategies.
        """
    def estimate_agos(self, nb_runs: typing.SupportsInt | typing.SupportsIndex, nb_generations: typing.SupportsInt | typing.SupportsIndex, transitory: typing.SupportsInt | typing.SupportsIndex = 0, runs_per_j: typing.SupportsInt | typing.SupportsIndex = 0) -> tuple[typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]"], typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]"]]:
        """
        Estimate the time-independent Average Gradient of Selection G^A(j).
        
        Runs independent trajectories; at each post-transitory generation computes
        the numerically exact gradient and bins the result by cooperator count j.
        OpenMP-parallelised over runs; per-thread caches prevent contention.
        
        Parameters
        ----------
        nb_runs : int
            Total trajectories when ``runs_per_j == 0`` (random starts).
        nb_generations : int
            Generations (each = N elementary steps) per trajectory.
        transitory : int, optional
            Burn-in generations not counted (default 0).
        runs_per_j : int, optional
            When > 0, uses the paper's sampling scheme: for each initial cooperator
            count j0 ∈ {1, …, N-1} exactly ``runs_per_j`` trajectories are started
            with j0 cooperators on random nodes.  Total runs = runs_per_j × (N-1);
            ``nb_runs`` is ignored.  This gives uniform coverage of all j values.
        
        Returns
        -------
        tuple[numpy.ndarray, numpy.ndarray]
            (mean_G, se_G) each of shape (N+1, nb_strategies).
            Row j holds the average/SE gradient at cooperator count j.
            Rows 0 and N are zero (absorbing states).
        """
    def estimate_agos_time_dependent(self, nb_runs: typing.SupportsInt | typing.SupportsIndex, nb_generations: typing.SupportsInt | typing.SupportsIndex) -> tuple[numpy.typing.NDArray[numpy.float64], numpy.typing.NDArray[numpy.float64]]:
        """
        Estimate the time-dependent Average Gradient of Selection G^A(j, t).
        
        Like estimate_agos but preserves the generation index, letting you study how
        the gradient landscape evolves from the initial transient to the stationary
        regime (e.g. Fig. 2 of Pinheiro et al. 2012).
        
        Returns
        -------
        tuple[numpy.ndarray, numpy.ndarray]
            (mean_G_t, se_G_t) each of shape (nb_generations, N+1, nb_strategies).
            ``mean_G_t[t, j, k]`` is the mean gradient of strategy k at generation t
            and cooperator count j.
        """
    def estimate_fixation_probability(self, invader: typing.SupportsInt | typing.SupportsIndex, resident: typing.SupportsInt | typing.SupportsIndex, nb_runs: typing.SupportsInt | typing.SupportsIndex, nb_generations: typing.SupportsInt | typing.SupportsIndex) -> float:
        """
        Estimate the fixation probability of a single invader in an otherwise resident population.
        
        Parameters
        ----------
        invader : int
        resident : int
        nb_runs : int
        nb_generations : int
            Maximum time-steps per trial.
        
        Returns
        -------
        float
            Estimated fixation probability in [0, 1].
        """
    def estimate_strategy_distribution(self, nb_runs: typing.SupportsInt | typing.SupportsIndex, nb_generations: typing.SupportsInt | typing.SupportsIndex, transitory: typing.SupportsInt | typing.SupportsIndex, tolerance: typing.SupportsFloat | typing.SupportsIndex = 0.0, check_every: typing.SupportsInt | typing.SupportsIndex = 0) -> tuple[typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"], typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"]]:
        """
        Estimate time-averaged strategy frequencies after the transitory period.
        
        Returns
        -------
        tuple[numpy.ndarray, numpy.ndarray]
            (mean_frequencies, standard_errors), each of length nb_strategies.
        """
    @typing.overload
    def initialize(self, init_state: typing.Annotated[numpy.typing.ArrayLike, numpy.uint64, "[m, 1]"]) -> None:
        """
        Initialise a manual-stepping session with the given strategy counts.
        
        Parameters
        ----------
        init_state : numpy.ndarray
            Strategy count vector of length nb_strategies; sum must equal population_size.
        """
    @typing.overload
    def initialize(self) -> None:
        """
        Initialise a manual-stepping session with a uniformly random strategy assignment.
        """
    def mean_population_state(self) -> typing.Annotated[numpy.typing.NDArray[numpy.uint64], "[m, 1]"]:
        """
        Return strategy count vector (length = nb_strategies).
        """
    def mu(self) -> float:
        ...
    def nb_strategies(self) -> int:
        ...
    def population_size(self) -> int:
        ...
    def population_strategies(self) -> list[int]:
        """
        Return per-node strategy assignments as a list of integers (length = population_size).
        """
    def run(self, nb_generations: typing.SupportsInt | typing.SupportsIndex, transitory: typing.SupportsInt | typing.SupportsIndex, init_state: typing.Annotated[numpy.typing.ArrayLike, numpy.uint64, "[m, 1]"]) -> typing.Annotated[numpy.typing.NDArray[numpy.uint64], "[m, n]"]:
        """
        Run a single trajectory and return aggregate strategy counts per generation.
        
        Returns
        -------
        numpy.ndarray
            Matrix of shape (nb_generations - transitory, nb_strategies).
        """
    def run_snapshots(self, nb_generations: typing.SupportsInt | typing.SupportsIndex, transitory: typing.SupportsInt | typing.SupportsIndex, snapshot_interval: typing.SupportsInt | typing.SupportsIndex, init_state: typing.Annotated[numpy.typing.ArrayLike, numpy.uint64, "[m, 1]"], callback: typing.Any) -> None:
        """
        Run a trajectory and call ``callback(generation, population)`` at each snapshot.
        
        Parameters
        ----------
        snapshot_interval : int
            Call callback every this many generations after transitory.
        callback : callable
            Called as ``callback(generation: int, population: list[int])``.
        """
    def set_beta(self, beta: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    def set_mu(self, mu: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    def step(self) -> None:
        """
        Advance the session by one generation.
        
        For asynchronous rules: N individual update steps (one per node on average).
        For synchronous rules (e.g. LinearProportional): one full simultaneous sweep.
        Raises RuntimeError if initialize() has not been called first.
        """
    def topology(self) -> list[list[int]]:
        ...
class NetworkMCEstimatorTDPC:
    """
    
    Monte Carlo estimator for evolutionary games on networks with time-dependent beta/mu schedules.
    
    Uses the Pairwise Comparison update rule with piecewise-constant C++ schedules for the
    selection intensity beta(t) and mutation rate mu(t).
    """
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    def __init__(self, game: egttools.numerical.numerical_.games.AbstractSpatialGame, topology: collections.abc.Mapping[typing.SupportsInt | typing.SupportsIndex, collections.abc.Sequence[typing.SupportsInt | typing.SupportsIndex]], nb_strategies: typing.SupportsInt | typing.SupportsIndex, beta: typing.SupportsFloat | typing.SupportsIndex, mu: typing.SupportsFloat | typing.SupportsIndex, cache_size: typing.SupportsInt | typing.SupportsIndex = 100000, seed: typing.SupportsInt | typing.SupportsIndex | None = None) -> None:
        """
        Construct a NetworkMCEstimator.
        
        Parameters
        ----------
        game : egttools.games.AbstractSpatialGame
            Spatial game used to evaluate node fitness.
        topology : dict[int, list[int]]
            Network adjacency dictionary (e.g. from NetworkX ``G.adjacency()``).
        nb_strategies : int
            Number of distinct strategies.
        beta : float
            Selection intensity.
        mu : float
            Mutation probability per time step.
        cache_size : int, optional
            Per-thread LRU fitness cache size (default 100000).
        seed : int, optional
            If given, seeds the step-by-step session generator (used by
            ``initialize``/``step``) independently of the process-global
            ``egttools.Random`` singleton. Does not affect the OpenMP-parallel
            batched ``estimate_*`` methods. Omit (default) for prior behavior.
        """
    def beta(self) -> float:
        ...
    def calculate_gradient_of_selection(self, population: collections.abc.Sequence[typing.SupportsInt | typing.SupportsIndex]) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"]:
        """
        Numerically exact gradient of selection for the given per-node strategy assignment.
        
        Parameters
        ----------
        population : list[int]
            Strategy index for each node (length = population_size).
        
        Returns
        -------
        numpy.ndarray
            Gradient vector of length nb_strategies.
        """
    def estimate_agos(self, nb_runs: typing.SupportsInt | typing.SupportsIndex, nb_generations: typing.SupportsInt | typing.SupportsIndex, transitory: typing.SupportsInt | typing.SupportsIndex = 0, runs_per_j: typing.SupportsInt | typing.SupportsIndex = 0) -> tuple[typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]"], typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]"]]:
        """
        Estimate the time-independent Average Gradient of Selection G^A(j).
        
        Runs independent trajectories; at each post-transitory generation computes
        the numerically exact gradient and bins the result by cooperator count j.
        OpenMP-parallelised over runs; per-thread caches prevent contention.
        
        Parameters
        ----------
        nb_runs : int
            Total trajectories when ``runs_per_j == 0`` (random starts).
        nb_generations : int
            Generations (each = N elementary steps) per trajectory.
        transitory : int, optional
            Burn-in generations not counted (default 0).
        runs_per_j : int, optional
            When > 0, uses the paper's sampling scheme: for each initial cooperator
            count j0 ∈ {1, …, N-1} exactly ``runs_per_j`` trajectories are started
            with j0 cooperators on random nodes.  Total runs = runs_per_j × (N-1);
            ``nb_runs`` is ignored.  This gives uniform coverage of all j values.
        
        Returns
        -------
        tuple[numpy.ndarray, numpy.ndarray]
            (mean_G, se_G) each of shape (N+1, nb_strategies).
            Row j holds the average/SE gradient at cooperator count j.
            Rows 0 and N are zero (absorbing states).
        """
    def estimate_agos_time_dependent(self, nb_runs: typing.SupportsInt | typing.SupportsIndex, nb_generations: typing.SupportsInt | typing.SupportsIndex) -> tuple[numpy.typing.NDArray[numpy.float64], numpy.typing.NDArray[numpy.float64]]:
        """
        Estimate the time-dependent Average Gradient of Selection G^A(j, t).
        
        Like estimate_agos but preserves the generation index, letting you study how
        the gradient landscape evolves from the initial transient to the stationary
        regime (e.g. Fig. 2 of Pinheiro et al. 2012).
        
        Returns
        -------
        tuple[numpy.ndarray, numpy.ndarray]
            (mean_G_t, se_G_t) each of shape (nb_generations, N+1, nb_strategies).
            ``mean_G_t[t, j, k]`` is the mean gradient of strategy k at generation t
            and cooperator count j.
        """
    def estimate_fixation_probability(self, invader: typing.SupportsInt | typing.SupportsIndex, resident: typing.SupportsInt | typing.SupportsIndex, nb_runs: typing.SupportsInt | typing.SupportsIndex, nb_generations: typing.SupportsInt | typing.SupportsIndex) -> float:
        """
        Estimate the fixation probability of a single invader in an otherwise resident population.
        
        Parameters
        ----------
        invader : int
        resident : int
        nb_runs : int
        nb_generations : int
            Maximum time-steps per trial.
        
        Returns
        -------
        float
            Estimated fixation probability in [0, 1].
        """
    def estimate_strategy_distribution(self, nb_runs: typing.SupportsInt | typing.SupportsIndex, nb_generations: typing.SupportsInt | typing.SupportsIndex, transitory: typing.SupportsInt | typing.SupportsIndex, tolerance: typing.SupportsFloat | typing.SupportsIndex = 0.0, check_every: typing.SupportsInt | typing.SupportsIndex = 0) -> tuple[typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"], typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"]]:
        """
        Estimate time-averaged strategy frequencies after the transitory period.
        
        Returns
        -------
        tuple[numpy.ndarray, numpy.ndarray]
            (mean_frequencies, standard_errors), each of length nb_strategies.
        """
    @typing.overload
    def initialize(self, init_state: typing.Annotated[numpy.typing.ArrayLike, numpy.uint64, "[m, 1]"]) -> None:
        """
        Initialise a manual-stepping session with the given strategy counts.
        
        Parameters
        ----------
        init_state : numpy.ndarray
            Strategy count vector of length nb_strategies; sum must equal population_size.
        """
    @typing.overload
    def initialize(self) -> None:
        """
        Initialise a manual-stepping session with a uniformly random strategy assignment.
        """
    def mean_population_state(self) -> typing.Annotated[numpy.typing.NDArray[numpy.uint64], "[m, 1]"]:
        """
        Return strategy count vector (length = nb_strategies).
        """
    def mu(self) -> float:
        ...
    def nb_strategies(self) -> int:
        ...
    def population_size(self) -> int:
        ...
    def population_strategies(self) -> list[int]:
        """
        Return per-node strategy assignments as a list of integers (length = population_size).
        """
    def run(self, nb_generations: typing.SupportsInt | typing.SupportsIndex, transitory: typing.SupportsInt | typing.SupportsIndex, init_state: typing.Annotated[numpy.typing.ArrayLike, numpy.uint64, "[m, 1]"]) -> typing.Annotated[numpy.typing.NDArray[numpy.uint64], "[m, n]"]:
        """
        Run a single trajectory and return aggregate strategy counts per generation.
        
        Returns
        -------
        numpy.ndarray
            Matrix of shape (nb_generations - transitory, nb_strategies).
        """
    def run_snapshots(self, nb_generations: typing.SupportsInt | typing.SupportsIndex, transitory: typing.SupportsInt | typing.SupportsIndex, snapshot_interval: typing.SupportsInt | typing.SupportsIndex, init_state: typing.Annotated[numpy.typing.ArrayLike, numpy.uint64, "[m, 1]"], callback: typing.Any) -> None:
        """
        Run a trajectory and call ``callback(generation, population)`` at each snapshot.
        
        Parameters
        ----------
        snapshot_interval : int
            Call callback every this many generations after transitory.
        callback : callable
            Called as ``callback(generation: int, population: list[int])``.
        """
    def set_beta(self, beta: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    def set_mu(self, mu: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
    def step(self) -> None:
        """
        Advance the session by one generation.
        
        For asynchronous rules: N individual update steps (one per node on average).
        For synchronous rules (e.g. LinearProportional): one full simultaneous sweep.
        Raises RuntimeError if initialize() has not been called first.
        """
    def topology(self) -> list[list[int]]:
        ...
def run_network_sweep(games: collections.abc.Sequence[typing.Any], betas: typing.Annotated[numpy.typing.ArrayLike, numpy.float64], topologies: collections.abc.Sequence[collections.abc.Mapping[typing.SupportsInt | typing.SupportsIndex, collections.abc.Sequence[typing.SupportsInt | typing.SupportsIndex]]], nb_strategies: typing.SupportsInt | typing.SupportsIndex, mu: typing.SupportsFloat | typing.SupportsIndex, nb_runs: typing.SupportsInt | typing.SupportsIndex, avg_gens: typing.SupportsInt | typing.SupportsIndex, transitory: typing.SupportsInt | typing.SupportsIndex, init_state: typing.Annotated[numpy.typing.ArrayLike, numpy.uint64], update_rule: str = 'PC', cache_size: typing.SupportsInt | typing.SupportsIndex = 100000) -> numpy.typing.NDArray[numpy.float64]:
    """
    Run a parallel parameter sweep over multiple games and topologies using OpenMP.
    
    Each (game, topology) combination is simulated independently with ``nb_runs`` runs.
    Topologies are shared read-only across threads — the adjacency list is never copied —
    so even a complete graph at large N uses only one copy of memory.
    
    Parameters
    ----------
    games : list[AbstractSpatialGame]
        One C++ game object per parameter combination (e.g. one per (T, S) grid point).
        **Must be C++ classes** such as ``NormalFormNetworkGame`` or
        ``OneShotCRDNetworkGame``.  Python subclasses of ``AbstractSpatialGame`` raise
        ``TypeError`` because ``calculate_fitness`` would be called without the GIL.
    betas : np.ndarray, shape (n_games,)
        Selection-intensity / payoff-normalisation constant, one per game.
        For the LP rule pass ``max(T, 1.0) - min(S, 0.0)`` per (T, S) point.
    topologies : list[dict[int, list[int]]]
        Network adjacency dicts (NetworkX format).  Shared read-only across threads.
    nb_strategies : int
        Number of strategies.
    mu : float
        Mutation probability per time step.
    nb_runs : int
        Independent simulation runs per (game, topology) combination.
    avg_gens : int
        Generations to record after the transitory.
    transitory : int
        Burn-in generations (discarded before recording).
    init_state : np.ndarray[uint64], shape (nb_strategies,)
        Initial strategy counts; must sum to population_size.
    update_rule : str, optional
        ``"LP"`` / ``"LinearProportional"`` (default for Santos 2006),
        ``"PC"`` / ``"PairwiseComparison"``, ``"BD"`` / ``"BirthDeath"``,
        ``"DB"`` / ``"DeathBirth"``.
    cache_size : int, optional
        Per-thread LRU fitness cache capacity (default 100 000).
    
    Returns
    -------
    np.ndarray, shape (n_games, n_topologies)
        Mean cooperation fraction (strategy-0 count / N) for each (game, topology)
        pair, averaged over ``nb_runs`` runs and ``avg_gens`` recording generations.
        Average over the topology axis to get a single cooperation frequency per
        parameter combination.
    """
__init__: str = 'The `egttools.numerical.structure` submodule contains population structures.'

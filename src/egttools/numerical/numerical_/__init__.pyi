"""
The `numerical` module contains optimized functions and classes to simulate evolutionary dynamics in large populations. This module is written in C++.
"""
from __future__ import annotations
from . import DataStructures
from . import behaviors
from . import distributions
from . import games
from . import random
from . import structure
__all__: list[str] = ['DataStructures', 'GeneralPopulationEvolver', 'NetworkEvolver', 'PairwiseComparison', 'PairwiseComparisonNumerical', 'USES_BOOST', 'VERSION', 'behaviors', 'calculate_nb_states', 'calculate_state', 'calculate_strategies_distribution', 'distributions', 'games', 'is_blas_lapack_enabled', 'is_openmp_enabled', 'random', 'replicator_equation', 'replicator_equation_n_player', 'sample_simplex', 'sample_simplex_directly', 'sample_unit_simplex', 'structure', 'vectorized_replicator_equation', 'vectorized_replicator_equation_n_player']
class GeneralPopulationEvolver:
    """
    
    Evolver for a general population structure.
    """
    @staticmethod
    def __init__(*args, **kwargs):
        """
        
        Construct an evolver for a general population structure.
        
        Parameters
        ----------
        structure : egttools.numerical.structure.AbstractStructure
            Structure defining how individuals interact and update their strategies.
        """
    @staticmethod
    def evolve(*args, **kwargs):
        """
        
        Evolve the population and return the final state.
        """
    @staticmethod
    def run(*args, **kwargs):
        """
        
        Run the population dynamics and return the result after discarding the transitory phase.
        """
    @staticmethod
    def structure(*args, **kwargs):
        """
        
        Return the structure used by the evolver.
        """
class NetworkEvolver:
    """
    
    Utility class for evolving network-structured populations.
    """
    @staticmethod
    def estimate_time_dependent_average_gradients_of_selection(*args, **kwargs):
        """
        
        Estimate time-dependent average gradients of selection for a set of states.
        
        
        Estimate time-dependent average gradients of selection across multiple networks.
        """
    @staticmethod
    def estimate_time_independent_average_gradients_of_selection(*args, **kwargs):
        """
        
        Estimate time-independent average gradients of selection for a set of states.
        
        
        Estimate time-independent average gradients of selection across multiple networks.
        """
    @staticmethod
    def evolve(*args, **kwargs):
        """
        
        Evolve the network population and return the final state.
        
        
        Evolve the network population from a given initial state and return the final state.
        """
    @staticmethod
    def run(*args, **kwargs):
        """
        
        Simulate the full trajectory of the population states.
        
        
        Run the simulation from a custom initial state and return the trajectory.
        """
class PairwiseComparison:
    """
    
    Analytical pairwise-comparison process for finite populations.
    
    This class studies evolutionary dynamics in a well-mixed population of fixed size
    :math:`Z`, whose state is represented by a vector of strategy counts
    :math:`x = (x_1, \\dots, x_n)` satisfying :math:`\\sum_{i=1}^n x_i = Z`.
    
    Under the pairwise comparison rule, strategy updates are driven by pairwise imitation,
    typically through the Fermi kernel
    
    .. math::
    
        p_{i \\to j}(x) =
        \\frac{1}{1 + \\exp[-\\beta (f_j(x) - f_i(x))]},
    
    where :math:`\\beta \\ge 0` is the intensity of selection and :math:`f_i(x)` is the
    fitness of strategy :math:`i` in state :math:`x`.
    
    The class provides tools to construct the full Markov transition matrix, compute
    gradients of selection, fixation probabilities, and the reduced small-mutation-limit
    (SML) dynamics.
    """
    @staticmethod
    def __init__(*args, **kwargs):
        """
        
        Construct an analytical pairwise-comparison process.
        
        Parameters
        ----------
        population_size : int
            Population size :math:`Z`.
        game : egttools.games.AbstractGame
            Game defining the fitness of each strategy as a function of the current
            population state.
        
        Notes
        -----
        The number of population states is
        
        .. math::
        
            |\\mathcal{S}| = \\binom{Z + n - 1}{n - 1},
        
        where :math:`n` is the number of strategies.
        
        
        Construct an analytical pairwise-comparison process with a configurable fitness cache.
        
        Parameters
        ----------
        population_size : int
            Population size :math:`Z`.
        game : egttools.games.AbstractGame
            Game defining the fitness of each strategy as a function of the current
            population state.
        cache_size : int
            Maximum number of cached fitness evaluations.
        """
    @staticmethod
    def calculate_fixation_probability(*args, **kwargs):
        """
        
        Compute the fixation probability of one mutant in a monomorphic resident population.
        
        This method restricts the dynamics to the one-dimensional edge involving the
        invading and resident strategies only. It returns the probability that a single
        invader eventually takes over the whole population.
        
        Parameters
        ----------
        invading_strategy_index : int
            Index of the invading strategy.
        resident_strategy_index : int
            Index of the resident strategy.
        beta : float
            Intensity of selection :math:`\\beta`.
        
        Returns
        -------
        float
            Probability that one invader fixates in a population of residents.
        """
    @staticmethod
    def calculate_gradient_of_selection(*args, **kwargs):
        """
        
        Compute the gradient of selection without mutation at a given population state.
        
        Let :math:`x = (x_1,\\dots,x_n)` be the current state. This method returns the
        expected one-step drift induced only by selection. For each strategy :math:`i`,
        
        .. math::
        
            g_i(x)
            =
            \\frac{1}{n}
            \\sum_{j \\ne i}
            \\left[
            T^{\\mathrm{sel}}_{j \\to i}(x) - T^{\\mathrm{sel}}_{i \\to j}(x)
            \\right],
        
        where :math:`T^{\\mathrm{sel}}_{j \\to i}(x)` is the probability that one
        individual of strategy :math:`j` is replaced by one individual of strategy
        :math:`i` under pairwise comparison alone.
        
        Under the Fermi rule, the local net flux can be written as
        
        .. math::
        
            T^{\\mathrm{sel}}_{j \\to i}(x) - T^{\\mathrm{sel}}_{i \\to j}(x)
            =
            \\frac{x_i x_j}{Z(Z-1)}
            \\tanh\\!\\left(\\frac{\\beta}{2}(f_i(x)-f_j(x))\\right).
        
        The resulting vector is tangent to the simplex, so
        
        .. math::
        
            \\sum_{i=1}^n g_i(x) = 0.
        
        Parameters
        ----------
        beta : float
            Intensity of selection :math:`\\beta`.
        state : numpy.ndarray
            One-dimensional integer array of shape `(nb_strategies,)` containing the
            current population state.
        
        Returns
        -------
        numpy.ndarray
            One-dimensional array of shape `(nb_strategies,)` containing the mutation-free
            gradient of selection.
        """
    @staticmethod
    def calculate_gradient_of_selection_with_mutation(*args, **kwargs):
        """
        
        Compute the gradient of selection with mutation at a given population state.
        
        Let :math:`x = (x_1,\\dots,x_n)` be the current state, with population size
        :math:`Z` and :math:`n` strategies. This method returns the expected one-step
        drift when both pairwise comparison and mutation are active:
        
        .. math::
        
            g_i^{(\\mu)}(x)
            =
            (1-\\mu)\\, g_i(x)
            +
            \\frac{\\mu_{\\mathrm{eff}}}{nZ}\\left(Z - n x_i\\right),
        
        where :math:`g_i(x)` is the mutation-free gradient returned by
        :meth:`calculate_gradient_of_selection`, and :math:`\\mu_{\\mathrm{eff}}` is the
        effective mutation probability towards one specific alternative strategy:
        
        .. math::
        
            \\mu_{\\mathrm{eff}} =
            \\begin{cases}
            \\mu, & n = 2, \\\\
            \\mu/(n-1), & n > 2.
            \\end{cases}
        
        The first term is the selection contribution scaled by :math:`(1-\\mu)`, and the
        second term is the mutation drift induced by uniform mutation towards the other
        strategies.
        
        As in the mutation-free case, the resulting vector is tangent to the simplex:
        
        .. math::
        
            \\sum_{i=1}^n g_i^{(\\mu)}(x) = 0.
        
        Parameters
        ----------
        beta : float
            Intensity of selection :math:`\\beta`.
        mu : float
            Mutation probability :math:`\\mu`.
        state : numpy.ndarray
            One-dimensional integer array of shape `(nb_strategies,)` containing the
            current population state.
        
        Returns
        -------
        numpy.ndarray
            One-dimensional array of shape `(nb_strategies,)` containing the gradient
            with mutation.
        """
    @staticmethod
    def calculate_transition_and_fixation_matrix_sml(*args, **kwargs):
        """
        
        Return the reduced transition matrix and fixation matrix in the small-mutation limit.
        
        In the Small Mutation Limit (SML), mutations are assumed sufficiently rare that
        the population is almost always monomorphic before the next mutation occurs.
        The resulting reduced Markov chain acts only on the monomorphic states.
        
        If the current monomorphic state is strategy :math:`i`, then for :math:`i \\ne j`
        
        .. math::
        
            T_{ij}^{\\mathrm{SML}} = \\frac{\\rho_{ij}}{n-1},
        
        where :math:`\\rho_{ij}` is the fixation probability of one mutant of strategy
        :math:`j` in a resident population of strategy :math:`i`. The diagonal entries
        are set so that each row sums to one.
        
        Parameters
        ----------
        beta : float
            Intensity of selection :math:`\\beta`.
        
        Returns
        -------
        tuple[numpy.ndarray, numpy.ndarray]
            A tuple `(transition_matrix, fixation_probabilities)` where:
        
            - `transition_matrix` is the reduced SML transition matrix of shape
              `(nb_strategies, nb_strategies)`;
            - `fixation_probabilities[i, j]` is the probability that one mutant of
              strategy `j` fixates in a population of strategy `i`.
        """
    @staticmethod
    def calculate_transition_matrix(*args, **kwargs):
        """
        
        Compute the full transition matrix of the finite-population Markov chain.
        
        The chain evolves on the set of all population states
        
        .. math::
        
            \\mathcal{S} = \\left\\{x \\in \\mathbb{N}^n : \\sum_{i=1}^n x_i = Z \\right\\}.
        
        Each off-diagonal transition changes the state by replacing one individual of one
        strategy by one individual of another strategy. Mutation is incorporated directly
        into the transition probabilities, and diagonal entries are set so that each row
        sums to one.
        
        Parameters
        ----------
        beta : float
            Intensity of selection :math:`\\beta`.
        mu : float
            Mutation probability :math:`\\mu`.
        
        Returns
        -------
        scipy.sparse.csr_matrix
            Sparse transition matrix of shape `(nb_states, nb_states)`.
        
        Notes
        -----
        For large state spaces, explicitly constructing this matrix may require a large
        amount of memory.
        """
    @staticmethod
    def game(*args, **kwargs):
        """
        
        Return the underlying game.
        
        Returns
        -------
        egttools.games.AbstractGame
            Reference to the game used to evaluate fitness.
        """
    @staticmethod
    def nb_states(*args, **kwargs):
        """
        
        Return the total number of population states.
        
        Returns
        -------
        int
            Number of states in the full Markov chain.
        """
    @staticmethod
    def nb_strategies(*args, **kwargs):
        """
        
        Return the number of strategies.
        
        Returns
        -------
        int
            Number of strategies.
        """
    @staticmethod
    def population_size(*args, **kwargs):
        """
        
        Return the population size.
        
        Returns
        -------
        int
            Population size :math:`Z`.
        """
    @staticmethod
    def pre_calculate_edge_fitnesses(*args, **kwargs):
        """
        
        Precompute fitness values along all edges of the simplex.
        
        This is particularly useful for repeated pairwise fixation calculations, since
        fixation probabilities only depend on states involving two strategies at a time.
        """
    @staticmethod
    def update_population_size(*args, **kwargs):
        """
        
        Update the population size.
        
        Parameters
        ----------
        population_size : int
            New population size :math:`Z`.
        """
class PairwiseComparisonNumerical:
    """
    
    Numerical solver for evolutionary dynamics under the pairwise comparison rule.
    """
    @staticmethod
    def __init__(*args, **kwargs):
        """
        
        Construct a numerical solver for a finite-population game.
        
        Parameters
        ----------
        pop_size : int
            Number of individuals in the population.
        game : egttools.games.AbstractGame
            Game object implementing the payoff and fitness structure.
        cache_size : int
            Maximum cache size for fitness computations.
        """
    @staticmethod
    def estimate_fixation_probability(*args, **kwargs):
        """
        
        Estimate the fixation probability of an invading strategy in a resident population.
        """
    @staticmethod
    def estimate_stationary_distribution(*args, **kwargs):
        """
        
        Estimate the stationary distribution of population states.
        
        Returns
        -------
        numpy.ndarray
            Estimated stationary distribution.
        """
    @staticmethod
    def estimate_stationary_distribution_sparse(*args, **kwargs):
        """
        
        Estimate the stationary distribution in sparse format.
        
        Returns
        -------
        scipy.sparse.csr_matrix
            Estimated stationary distribution in sparse format.
        """
    @staticmethod
    def estimate_strategy_distribution(*args, **kwargs):
        """
        
        Estimate the average frequency of each strategy over time.
        
        Returns
        -------
        numpy.ndarray
            Average frequency of each strategy.
        """
    @staticmethod
    def evolve(*args, **kwargs):
        """
        
        Simulate the pairwise comparison process with mutation.
        
        Parameters
        ----------
        nb_generations : int
            Number of generations to simulate.
        beta : float
            Intensity of selection.
        mu : float
            Mutation rate.
        init_state : numpy.ndarray
            Initial population state.
        
        Returns
        -------
        numpy.ndarray
            Final population state.
        """
    @staticmethod
    def run(*args, **kwargs):
        ...
    @staticmethod
    def run_with_mutation(*args, **kwargs):
        """
        
        Simulate the stochastic dynamics with mutation.
        
        Returns
        -------
        numpy.ndarray
            Matrix containing all intermediate population states.
        
        
        Simulate the stochastic dynamics with mutation, skipping the transient phase.
        
        Returns
        -------
        numpy.ndarray
            Matrix containing the population states after the transient period.
        """
    @staticmethod
    def run_without_mutation(*args, **kwargs):
        """
        
        Simulate the stochastic dynamics without mutation.
        
        Returns
        -------
        numpy.ndarray
            Matrix containing all intermediate population states.
        
        
        Simulate the stochastic dynamics without mutation, skipping the transient phase.
        
        Returns
        -------
        numpy.ndarray
            Matrix containing the population states after the transient period.
        """
    @property
    def cache_size(*args, **kwargs):
        """
        Maximum number of cached fitness values.
        """
    @cache_size.setter
    def cache_size(*args, **kwargs):
        ...
    @property
    def nb_states(*args, **kwargs):
        """
        Number of discrete states in the population.
        """
    @property
    def nb_strategies(*args, **kwargs):
        """
        Number of strategies in the population.
        """
    @property
    def payoffs(*args, **kwargs):
        """
        Payoff matrix used for selection dynamics.
        """
    @property
    def pop_size(*args, **kwargs):
        """
        Current population size.
        """
    @pop_size.setter
    def pop_size(*args, **kwargs):
        ...
def calculate_nb_states(*args, **kwargs):
    """
    
    Calculate the number of possible states in a discrete simplex.
    
    This is the number of integer compositions of `group_size` individuals into
    `nb_strategies` categories.
    
    Parameters
    ----------
    group_size : int
        Total number of individuals.
    nb_strategies : int
        Number of available strategies.
    
    Returns
    -------
    int
        Number of possible discrete states.
    
    See Also
    --------
    egttools.sample_simplex
    egttools.calculate_state
    
    Examples
    --------
    >>> calculate_nb_states(4, 3)
    15
    >>> calculate_nb_states(10, 2)
    11
    """
def calculate_state(*args, **kwargs):
    """
    
    Convert a discrete population configuration into a unique index.
    
    Parameters
    ----------
    group_size : int
        Total number of individuals.
    group_composition : list[int]
        Number of individuals using each strategy.
    
    Returns
    -------
    int
        Unique index corresponding to the group composition.
    
    See Also
    --------
    egttools.sample_simplex
    egttools.calculate_nb_states
    
    Examples
    --------
    >>> calculate_state(3, [1, 1, 1])
    3
    >>> calculate_state(2, [2, 0, 0])
    0
    
    
    Convert a discrete population configuration into a unique index.
    
    Parameters
    ----------
    group_size : int
        Total number of individuals.
    group_composition : numpy.ndarray
        One-dimensional integer array containing the number of individuals using each strategy.
    
    Returns
    -------
    int
        Unique index corresponding to the group composition.
    
    See Also
    --------
    egttools.sample_simplex
    egttools.calculate_nb_states
    
    Examples
    --------
    >>> calculate_state(3, np.array([1, 1, 1]))
    3
    >>> calculate_state(4, np.array([2, 2, 0]))
    6
    """
def calculate_strategies_distribution(*args, **kwargs):
    """
    
    Calculate the average frequency of each strategy given a stationary distribution.
    
    Parameters
    ----------
    pop_size : int
        Total number of individuals in the population.
    nb_strategies : int
        Number of strategies available in the population.
    stationary_distribution : scipy.sparse.csr_matrix
        Sparse matrix representing the stationary distribution over population states.
    
    Returns
    -------
    numpy.ndarray
        One-dimensional array containing the average frequency of each strategy.
    
    See Also
    --------
    egttools.calculate_state
    egttools.sample_simplex
    egttools.calculate_nb_states
    egttools.numerical.PairwiseComparisonNumerical.estimate_stationary_distribution_sparse
    
    Examples
    --------
    >>> from scipy.sparse import csr_matrix
    >>> freq = calculate_strategies_distribution(10, 3, csr_matrix(...))
    >>> freq.shape
    (3,)
    """
def calculate_expected_payoff(pop_size: int, group_size: int, nb_strategies: int,
                              stationary_distribution, payoff_matrix) -> float:
    """
    Calculate the expected payoff averaged over the stationary distribution.

    E[payoff] = sum_s sd(s) * sum_g P(g|s) * avg_payoff(g)

    where avg_payoff(g) = sum_j (g[j] / group_size) * payoff_matrix(j, g_index).
    Each strategy's payoff is weighted by its frequency inside the sampled group.

    Parameters
    ----------
    pop_size : int
        Total number of individuals in the population.
    group_size : int
        Number of individuals sampled per group interaction.
    nb_strategies : int
        Number of strategies available in the population.
    stationary_distribution : scipy.sparse.csr_matrix
        Sparse matrix representing the stationary distribution over population states.
    payoff_matrix : numpy.ndarray
        Matrix of shape (nb_strategies, nb_group_compositions); entry (j, g) is the
        payoff of strategy j when the group composition index is g.

    Returns
    -------
    float
        Expected payoff scalar.
    """
def calculate_expected_indicator(pop_size: int, group_size: int, nb_strategies: int,
                                 stationary_distribution, indicator) -> float:
    """
    Calculate E[f] = sum_s sd(s) * sum_g P(g|s) * f(g) for an arbitrary indicator f.

    The callable ``indicator`` receives a list of ints representing the count of each
    strategy in the sampled group (length nb_strategies, sums to group_size) and must
    return a float.

    Parameters
    ----------
    pop_size : int
        Total number of individuals in the population.
    group_size : int
        Number of individuals sampled per group interaction.
    nb_strategies : int
        Number of strategies available in the population.
    stationary_distribution : scipy.sparse.csr_matrix
        Sparse matrix representing the stationary distribution over population states.
    indicator : callable
        Function with signature ``f(group_config: list[int]) -> float``.

    Returns
    -------
    float
        Expected value of the indicator.

    Examples
    --------
    >>> eta_G = calculate_expected_indicator(
    ...     pop_size, group_size, nb_strategies, sd,
    ...     lambda g: float(g[0] >= 3)
    ... )
    """
def calculate_expected_group_success(pop_size: int, group_size: int, nb_strategies: int,
                                     stationary_distribution,
                                     threshold: int,
                                     contributing_strategies: list) -> float:
    """
    Calculate the expected group success eta_G under the stationary distribution.

    eta_G = sum_s sd(s) * sum_g P(g|s) * I(sum_{k in contributing_strategies} g[k] >= threshold)

    Parameters
    ----------
    pop_size : int
        Total number of individuals in the population.
    group_size : int
        Number of individuals sampled per group interaction.
    nb_strategies : int
        Number of strategies available in the population.
    stationary_distribution : scipy.sparse.csr_matrix
        Sparse matrix representing the stationary distribution over population states.
    threshold : int
        Minimum total count of contributing strategies required for the group to succeed.
    contributing_strategies : list[int]
        Indices of the strategies that count towards the threshold.

    Returns
    -------
    float
        Expected group success in [0, 1].

    Examples
    --------
    >>> eta_G = calculate_expected_group_success(
    ...     pop_size, group_size, nb_strategies, sd,
    ...     threshold=3, contributing_strategies=[0]
    ... )
    >>> eta_G = calculate_expected_group_success(
    ...     pop_size, group_size, nb_strategies, sd,
    ...     threshold=3, contributing_strategies=[0, 2]
    ... )
    """
def is_blas_lapack_enabled() -> bool:
    """
    Check if EGTtools was compiled with BLAS/LAPACK acceleration.
    """
def is_openmp_enabled() -> bool:
    """
    Check if EGTtools was compiled with OpenMP support.
    """
def replicator_equation(*args, **kwargs):
    """
    
    Compute the replicator dynamics gradient for a two-player matrix game.
    
    Parameters
    ----------
    frequencies : numpy.ndarray
        One-dimensional array containing the frequency of each strategy.
    payoff_matrix : numpy.ndarray
        Two-dimensional payoff matrix.
    
    Returns
    -------
    numpy.ndarray
        One-dimensional array containing the replicator gradient for each strategy.
    
    See Also
    --------
    egttools.replicator_equation_n_player
    egttools.games.AbstractReplicatorGame
    
    
    Compute the replicator dynamics gradient for a two-player game object.
    
    Parameters
    ----------
    frequencies : numpy.ndarray
        One-dimensional array containing the frequency of each strategy.
    game : egttools.games.AbstractReplicatorGame
        Game object used to compute expected fitness values.
    
    Returns
    -------
    numpy.ndarray
        One-dimensional array containing the replicator gradient for each strategy.
    
    See Also
    --------
    egttools.replicator_equation_n_player
    egttools.games.AbstractReplicatorGame
    """
def replicator_equation_n_player(*args, **kwargs):
    """
    
    Compute the replicator dynamics gradient for an N-player game defined by a payoff table.
    
    Parameters
    ----------
    frequencies : numpy.ndarray
        One-dimensional array containing the frequency of each strategy.
    payoff_matrix : numpy.ndarray
        Two-dimensional payoff table. Rows correspond to strategies and columns to
        group configurations.
    group_size : int
        Number of players interacting simultaneously.
    
    Returns
    -------
    numpy.ndarray
        One-dimensional array containing the replicator gradient for each strategy.
    
    See Also
    --------
    egttools.replicator_equation
    egttools.games.AbstractReplicatorGame
    
    
    Compute the replicator dynamics gradient for an N-player game object.
    
    Parameters
    ----------
    frequencies : numpy.ndarray
        One-dimensional array containing the frequency of each strategy.
    game : egttools.games.AbstractReplicatorGame
        Game object used to compute expected fitness values.
    
    Returns
    -------
    numpy.ndarray
        One-dimensional array containing the replicator gradient for each strategy.
    
    See Also
    --------
    egttools.replicator_equation
    egttools.games.AbstractReplicatorGame
    """
def sample_simplex(*args, **kwargs):
    """
    
    Convert a state index into a group composition vector.
    
    This function is the inverse of `calculate_state`.
    
    Parameters
    ----------
    index : int
        Index of the population state.
    pop_size : int
        Total number of individuals.
    nb_strategies : int
        Number of available strategies.
    
    Returns
    -------
    numpy.ndarray
        One-dimensional integer array containing the number of individuals using each strategy.
    
    See Also
    --------
    egttools.calculate_state
    egttools.calculate_nb_states
    
    Examples
    --------
    >>> sample_simplex(0, 3, 3)
    array([3, 0, 0])
    >>> sample_simplex(3, 3, 3)
    array([1, 1, 1])
    """
def sample_simplex_directly(*args, **kwargs):
    """
    
    Sample a discrete population state uniformly at random from the simplex.
    
    Parameters
    ----------
    nb_strategies : int
        Number of available strategies.
    pop_size : int
        Total number of individuals in the population.
    
    Returns
    -------
    numpy.ndarray
        One-dimensional integer array containing the number of individuals using each strategy.
    
    See Also
    --------
    egttools.calculate_state
    egttools.calculate_nb_states
    egttools.sample_simplex
    
    Examples
    --------
    >>> sample_simplex_directly(3, 10)
    array([3, 4, 3])
    >>> sample_simplex_directly(2, 5)
    array([2, 3])
    """
def sample_unit_simplex(*args, **kwargs):
    """
    
    Sample a point uniformly at random from the unit simplex.
    
    Parameters
    ----------
    nb_strategies : int
        Number of strategies in the population.
    
    Returns
    -------
    numpy.ndarray
        One-dimensional floating-point array representing a valid probability distribution.
    
    See Also
    --------
    egttools.sample_simplex
    egttools.sample_simplex_directly
    
    Examples
    --------
    >>> sample_unit_simplex(3)
    array([0.25, 0.57, 0.18])
    >>> sample_unit_simplex(2)
    array([0.70, 0.30])
    """
def vectorized_replicator_equation(*args, **kwargs):
    """
    
    Vectorized computation of the replicator dynamics for three-strategy two-player games.
    
    Parameters
    ----------
    x1 : numpy.ndarray
        Two-dimensional array containing the frequency of strategy 1 at each grid point.
    x2 : numpy.ndarray
        Two-dimensional array containing the frequency of strategy 2 at each grid point.
    x3 : numpy.ndarray
        Two-dimensional array containing the frequency of strategy 3 at each grid point.
    game : egttools.games.AbstractReplicatorGame
        Two-player game object used to compute expected fitness values.
    
    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]
        Replicator gradients for the three strategies over the grid.
    
    See Also
    --------
    egttools.replicator_equation
    egttools.vectorized_replicator_equation_n_player
    """
def vectorized_replicator_equation_n_player(*args, **kwargs):
    """
    
    Vectorized computation of the replicator dynamics for three-strategy N-player games.
    
    Parameters
    ----------
    x1 : numpy.ndarray
        Two-dimensional array containing the frequency of strategy 1 at each grid point.
    x2 : numpy.ndarray
        Two-dimensional array containing the frequency of strategy 2 at each grid point.
    x3 : numpy.ndarray
        Two-dimensional array containing the frequency of strategy 3 at each grid point.
    payoff_matrix : numpy.ndarray
        Two-dimensional payoff table.
    group_size : int
        Number of players in each interacting group.
    
    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]
        Replicator gradients for the three strategies over the grid.
    
    See Also
    --------
    egttools.replicator_equation_n_player
    egttools.vectorized_replicator_equation
    
    
    Vectorized computation of the replicator dynamics for three-strategy N-player game objects.
    
    Parameters
    ----------
    x1 : numpy.ndarray
        Two-dimensional array containing the frequency of strategy 1 at each grid point.
    x2 : numpy.ndarray
        Two-dimensional array containing the frequency of strategy 2 at each grid point.
    x3 : numpy.ndarray
        Two-dimensional array containing the frequency of strategy 3 at each grid point.
    game : egttools.games.AbstractReplicatorGame
        Game object used to compute expected fitness values.
    
    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]
        Replicator gradients for the three strategies over the grid.
    
    See Also
    --------
    egttools.replicator_equation_n_player
    egttools.vectorized_replicator_equation
    """
USES_BOOST: bool = True
VERSION: str = '"0.1.15"'
__init__: str = 'The `numerical` module contains optimized functions and classes to simulate evolutionary dynamics in large populations.'
__version__: str = '"0.1.15"'

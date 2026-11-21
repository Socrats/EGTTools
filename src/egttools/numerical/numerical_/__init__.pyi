"""
The `numerical` module contains optimized functions and classes to simulate evolutionary dynamics in large populations. This module is written in C++.
"""
from __future__ import annotations
import collections.abc
import numpy
import numpy.typing
import scipy.sparse
import typing
from . import DataStructures
from . import behaviors
from . import distributions
from . import games
from . import random
from . import structure
__all__ = ['DataStructures', 'GeneralPopulationEvolver', 'MLSGarcia', 'MLSTraulsen', 'PairwiseComparison', 'PairwiseComparisonNumerical', 'PairwiseComparisonTransitionOperator', 'USES_BOOST', 'behaviors', 'calculate_expected_group_success', 'calculate_expected_indicator', 'calculate_expected_indicators', 'calculate_expected_indicators_precomputed', 'calculate_expected_payoff', 'calculate_expected_state_indicator', 'calculate_expected_state_indicators', 'calculate_expected_state_indicators_precomputed', 'calculate_hypergeometric_expected_value', 'calculate_hypergeometric_fitness', 'calculate_nb_states', 'calculate_state', 'calculate_strategies_distribution', 'distributions', 'games', 'is_arpack_enabled', 'is_blas_lapack_enabled', 'is_boost_enabled', 'is_openmp_enabled', 'is_petsc_enabled', 'precompute_group_to_state_indicator_matrix', 'random', 'replicator_equation', 'replicator_equation_n_player', 'sample_simplex', 'sample_simplex_directly', 'sample_unit_simplex', 'structure', 'vectorized_replicator_equation', 'vectorized_replicator_equation_n_player']
class GeneralPopulationEvolver:
    """
    
    Evolver for a general population structure.
    """
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    def __init__(self, structure: structure.AbstractStructure) -> None:
        """
        Construct an evolver for a general population structure.
        
        Parameters
        ----------
        structure : egttools.numerical.structure.AbstractStructure
            Structure defining how individuals interact and update their strategies.
        """
    def evolve(self, nb_generations: typing.SupportsInt | typing.SupportsIndex) -> typing.Annotated[numpy.typing.NDArray[numpy.uint64], "[m, 1]"]:
        """
        Evolve the population and return the final state.
        """
    def run(self, nb_generations: typing.SupportsInt | typing.SupportsIndex, transitory: typing.SupportsInt | typing.SupportsIndex) -> typing.Annotated[numpy.typing.NDArray[numpy.uint64], "[m, n]"]:
        """
        Run the population dynamics and return the result after discarding the transitory phase.
        """
    def structure(self) -> structure.AbstractStructure:
        """
        Return the structure used by the evolver.
        """
class MLSGarcia:
    """
    
            Multi-level selection following Garcia & van den Bergh (2011).
    
            Extends the Traulsen & Nowak model with direct group conflict (kappa)
            and separate in-group / out-group payoff matrices (alpha weighting).
    
            Parameters
            ----------
            nb_generations : int
                Maximum number of Moran steps per run.
            nb_strategies : int
                Number of distinct strategies.
            group_size : int
                Maximum group capacity n (must be >= 4).
            nb_groups : int
                Number of groups m.
            
    """
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    def __init__(self, nb_generations: typing.SupportsInt | typing.SupportsIndex, nb_strategies: typing.SupportsInt | typing.SupportsIndex, group_size: typing.SupportsInt | typing.SupportsIndex, nb_groups: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    def __repr__(self) -> str:
        ...
    def fixation_probability(self, invader: typing.SupportsInt | typing.SupportsIndex, resident: typing.SupportsInt | typing.SupportsIndex, nb_runs: typing.SupportsInt | typing.SupportsIndex, q: typing.SupportsFloat | typing.SupportsIndex, lambda: typing.SupportsFloat | typing.SupportsIndex, w: typing.SupportsFloat | typing.SupportsIndex, alpha: typing.SupportsFloat | typing.SupportsIndex, kappa: typing.SupportsFloat | typing.SupportsIndex, z: typing.SupportsFloat | typing.SupportsIndex, payoff_matrix_in: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]", "flags.c_contiguous"], payoff_matrix_out: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]", "flags.c_contiguous"]) -> float:
        """
                     Estimate fixation probability of invader over resident.
        
                     Parameters
                     ----------
                     invader : int
                     resident : int
                     nb_runs : int
                     q : float      splitting probability
                     lambda : float  migration probability
                     w : float       intensity of selection
                     alpha : float   fraction of interactions within the group
                     kappa : float   average fraction of groups involved in conflict
                     z : float       importance of payoffs in conflict (0 = deterministic)
                     payoff_matrix_in : np.ndarray   (nb_strategies x nb_strategies) in-group payoff
                     payoff_matrix_out : np.ndarray  (nb_strategies x nb_strategies) out-group payoff
        """
    @property
    def generations(self) -> int:
        ...
    @generations.setter
    def generations(self, arg1: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def group_size(self) -> int:
        ...
    @group_size.setter
    def group_size(self, arg1: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def max_pop_size(self) -> int:
        ...
    @property
    def nb_groups(self) -> int:
        ...
    @nb_groups.setter
    def nb_groups(self, arg1: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def nb_strategies(self) -> int:
        ...
class MLSTraulsen:
    """
    
            Multi-level selection following Traulsen & Nowak (2006).
    
            A population of m groups each of maximum size n. In each step an
            individual is selected proportional to its fitness and reproduces; if
            the group exceeds n it either splits (probability q) or ejects a random
            member (probability 1-q). Migration probability lambda moves offspring to
            a randomly chosen other group.
    
            Parameters
            ----------
            nb_generations : int
                Maximum number of Moran steps per run.
            nb_strategies : int
                Number of distinct strategies.
            group_size : int
                Maximum group capacity n (must be >= 4).
            nb_groups : int
                Number of groups m.
            w : float
                Intensity of selection.
            strategies_freq : np.ndarray
                Initial frequency of each strategy (must sum to 1).
            payoff_matrix : np.ndarray
                Square (nb_strategies x nb_strategies) payoff matrix.
            
    """
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    def __init__(self, nb_generations: typing.SupportsInt | typing.SupportsIndex, nb_strategies: typing.SupportsInt | typing.SupportsIndex, group_size: typing.SupportsInt | typing.SupportsIndex, nb_groups: typing.SupportsInt | typing.SupportsIndex, w: typing.SupportsFloat | typing.SupportsIndex, strategies_freq: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"], payoff_matrix: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]", "flags.c_contiguous"]) -> None:
        ...
    def __repr__(self) -> str:
        ...
    @typing.overload
    def evolve(self, q: typing.SupportsFloat | typing.SupportsIndex, w: typing.SupportsFloat | typing.SupportsIndex, init_state: typing.Annotated[numpy.typing.NDArray[numpy.uint64], "[m, 1]"]) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"]:
        """
        Run one Moran simulation and return the final strategy frequencies.
        """
    @typing.overload
    def evolve(self, q: typing.SupportsFloat | typing.SupportsIndex, w: typing.SupportsFloat | typing.SupportsIndex, lambda: typing.SupportsFloat | typing.SupportsIndex, init_state: typing.Annotated[numpy.typing.NDArray[numpy.uint64], "[m, 1]"]) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"]:
        """
        Run one Moran simulation with migration and return the final strategy frequencies.
        """
    @typing.overload
    def fixation_probability(self, invader: typing.SupportsInt | typing.SupportsIndex, resident: typing.SupportsInt | typing.SupportsIndex, nb_runs: typing.SupportsInt | typing.SupportsIndex, q: typing.SupportsFloat | typing.SupportsIndex, w: typing.SupportsFloat | typing.SupportsIndex) -> float:
        """
        Estimate fixation probability of invader over resident (no migration).
        """
    @typing.overload
    def fixation_probability(self, invader: typing.SupportsInt | typing.SupportsIndex, resident: typing.SupportsInt | typing.SupportsIndex, nb_runs: typing.SupportsInt | typing.SupportsIndex, q: typing.SupportsFloat | typing.SupportsIndex, lambda: typing.SupportsFloat | typing.SupportsIndex, w: typing.SupportsFloat | typing.SupportsIndex) -> float:
        """
        Estimate fixation probability of invader over resident (with migration).
        """
    def gradient_of_selection(self, invader: typing.SupportsInt | typing.SupportsIndex, resident: typing.SupportsInt | typing.SupportsIndex, nb_runs: typing.SupportsInt | typing.SupportsIndex, w: typing.SupportsFloat | typing.SupportsIndex, q: typing.SupportsFloat | typing.SupportsIndex = 0.0) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"]:
        """
        Estimate gradient of selection (T+ - T-) for each population configuration.
        """
    @property
    def generations(self) -> int:
        ...
    @generations.setter
    def generations(self, arg1: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def group_size(self) -> int:
        ...
    @group_size.setter
    def group_size(self, arg1: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def max_pop_size(self) -> int:
        ...
    @property
    def nb_groups(self) -> int:
        ...
    @nb_groups.setter
    def nb_groups(self, arg1: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def nb_strategies(self) -> int:
        ...
    @property
    def payoff_matrix(self) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]"]:
        ...
    @property
    def selection_intensity(self) -> float:
        ...
    @selection_intensity.setter
    def selection_intensity(self, arg1: typing.SupportsFloat | typing.SupportsIndex) -> None:
        ...
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
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    @typing.overload
    def __init__(self, population_size: typing.SupportsInt | typing.SupportsIndex, game: games.AbstractGame) -> None:
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
        """
    @typing.overload
    def __init__(self, population_size: typing.SupportsInt | typing.SupportsIndex, game: games.AbstractGame, cache_size: typing.SupportsInt | typing.SupportsIndex) -> None:
        """
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
    def calculate_fixation_probability(self, invading_strategy_index: typing.SupportsInt | typing.SupportsIndex, resident_strategy_index: typing.SupportsInt | typing.SupportsIndex, beta: typing.SupportsFloat | typing.SupportsIndex, high_precision: bool = False) -> float:
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
        high_precision : bool, optional
            When *True*, the internal streaming log-sum-exp is evaluated in 50-digit decimal
            arithmetic, extending the representable range of :math:`\\exp(\\log\\phi)`.
            The return value is still a ``float``; values below ``DBL_MIN`` (~2.2e-308)
            are rounded to zero regardless.  Use
            :meth:`calculate_log_fixation_probability` for the full dynamic range.
            Defaults to ``False``.
        
        Returns
        -------
        float
            Probability that one invader fixates in a population of residents.
        
        Raises
        ------
        IndexError
            If a strategy index is not in ``[0, nb_strategies)``.
        ValueError
            If the invading and resident strategies are the same.
        """
    def calculate_gradient_of_selection(self, beta: typing.SupportsFloat | typing.SupportsIndex, state: typing.Annotated[numpy.typing.NDArray[numpy.uint64], "[m, 1]"]) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"]:
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
        
        Raises
        ------
        ValueError
            If ``state`` does not have one entry per strategy, each between 0 and the population
            size, summing to the population size.
        """
    def calculate_gradient_of_selection_with_mutation(self, beta: typing.SupportsFloat | typing.SupportsIndex, mu: typing.SupportsFloat | typing.SupportsIndex, state: typing.Annotated[numpy.typing.NDArray[numpy.uint64], "[m, 1]"]) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"]:
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
        
        Raises
        ------
        ValueError
            If ``state`` does not have one entry per strategy, each between 0 and the population
            size, summing to the population size.
        """
    def calculate_log_fixation_probability(self, invading_strategy_index: typing.SupportsInt | typing.SupportsIndex, resident_strategy_index: typing.SupportsInt | typing.SupportsIndex, beta: typing.SupportsFloat | typing.SupportsIndex) -> float:
        """
        Return the natural log of the fixation probability, :math:`\\log\\rho`.
        
        Unlike :meth:`calculate_fixation_probability`, this method never underflows: the
        result is a finite ``float`` for any combination of :math:`\\beta`, population
        size, and fitness values.  It is computed as
        :math:`-\\mathrm{softplus}(\\log\\phi)`, so values as small as
        :math:`e^{-10^{308}}` are representable.
        
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
            :math:`\\log\\rho(\\text{invader} \\to \\text{resident})`, always finite.
        
        Raises
        ------
        IndexError
            If a strategy index is not in ``[0, nb_strategies)``.
        ValueError
            If the invading and resident strategies are the same.
        """
    def calculate_transition_and_fixation_matrix_sml(self, beta: typing.SupportsFloat | typing.SupportsIndex) -> tuple[typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]"], typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]"]]:
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
    def calculate_transition_and_log_fixation_matrix_sml(self, beta: typing.SupportsFloat | typing.SupportsIndex) -> tuple[typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]"], typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]"]]:
        """
        Return a numerically stable SML transition matrix and the log-fixation matrix.
        
        Like :meth:`calculate_transition_and_fixation_matrix_sml` but stores
        :math:`\\log\\rho_{ij}` and builds the transition matrix by scaling all
        off-diagonal entries by :math:`\\exp(-\\max_{k\\ne l}\\log\\rho_{kl})`.  This global
        rescaling preserves the stationary distribution while guaranteeing a valid
        stochastic matrix even when every :math:`\\rho_{ij}` underflows to zero in
        double precision.
        
        Parameters
        ----------
        beta : float
            Intensity of selection :math:`\\beta`.
        
        Returns
        -------
        tuple[numpy.ndarray, numpy.ndarray]
            A tuple `(transition_matrix, log_fixation_probabilities)` where:
        
            - `transition_matrix` has the same stationary distribution as the standard
              SML matrix but is numerically well-conditioned;
            - `log_fixation_probabilities[i, j]` = :math:`\\log\\rho_{ij}`.
        """
    def calculate_transition_matrix(self, beta: typing.SupportsFloat | typing.SupportsIndex, mu: typing.SupportsFloat | typing.SupportsIndex) -> scipy.sparse.csr_matrix:
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
        
        Implementation note: fitness values are pre-computed in a serial pass (with the
        GIL held, so Python-subclassed games work correctly), then the sparse matrix is
        assembled in a parallel OpenMP pass (with the GIL released).
        """
    def game(self) -> games.AbstractGame:
        """
        Return the underlying game.
        
        Returns
        -------
        egttools.games.AbstractGame
            Reference to the game used to evaluate fitness.
        """
    def nb_states(self) -> int:
        """
        Return the total number of population states.
        
        Returns
        -------
        int
            Number of states in the full Markov chain.
        """
    def nb_strategies(self) -> int:
        """
        Return the number of strategies.
        
        Returns
        -------
        int
            Number of strategies.
        """
    def population_size(self) -> int:
        """
        Return the population size.
        
        Returns
        -------
        int
            Population size :math:`Z`.
        """
    def pre_calculate_edge_fitnesses(self) -> None:
        """
        Precompute fitness values along all edges of the simplex.
        
        This is particularly useful for repeated pairwise fixation calculations, since
        fixation probabilities only depend on states involving two strategies at a time.
        """
    def update_population_size(self, population_size: typing.SupportsInt | typing.SupportsIndex) -> None:
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
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    def __init__(self, pop_size: typing.SupportsInt | typing.SupportsIndex, game: games.AbstractGame, cache_size: typing.SupportsInt | typing.SupportsIndex, seed: typing.SupportsInt | typing.SupportsIndex | None = None) -> None:
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
        seed : int, optional
            If given, seeds this instance's sequential-method generator (used by
            ``evolve``/``run``/``run_with_mutation``/etc.) independently of the
            process-global ``egttools.Random`` singleton, making those methods
            reproducible for this instance regardless of what else is drawing from
            the global generator. Does not affect the OpenMP-parallel batched
            ``estimate_*`` methods, whose per-run generators still draw from the
            global singleton. Omit (default) to keep the previous behavior.
        """
    def change_game(self, game: games.AbstractGame) -> None:
        """
        Replace the game used for fitness computation.
        
        The solver retains a pointer to the new game object; the caller must ensure
        the game stays alive for the lifetime of this solver (enforced automatically
        when called from Python via the keep-alive policy).
        
        Parameters
        ----------
        game : egttools.games.AbstractGame
            New game object. Must have the same number of strategies as the current game.
        """
    def estimate_absorption_probabilities(self, beta: typing.SupportsFloat | typing.SupportsIndex, init_state: typing.Annotated[numpy.typing.NDArray[numpy.uint64], "[m, 1]"], nb_runs: typing.SupportsInt | typing.SupportsIndex) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"]:
        """
        Estimate the absorption probability for each strategy from a given initial state.
        
        Runs independent trajectories of the mutation-free Moran process from ``init_state``
        and records which strategy fixed in each run. Generalises
        ``estimate_fixation_probability`` to k > 2 strategies.
        
        Parameters
        ----------
        beta : float
            Intensity of selection.
        init_state : numpy.ndarray
            Initial population state — array of strategy counts summing to ``pop_size``.
        nb_runs : int
            Number of independent trajectories.
        
        Returns
        -------
        numpy.ndarray
            Array of shape ``(nb_strategies,)`` with the empirical fixation probability
            for each strategy.
        """
    def estimate_fixation_probability(self, index_invading_strategy: typing.SupportsInt | typing.SupportsIndex, index_resident_strategy: typing.SupportsInt | typing.SupportsIndex, nb_runs: typing.SupportsInt | typing.SupportsIndex, nb_generations: typing.SupportsInt | typing.SupportsIndex, beta: typing.SupportsFloat | typing.SupportsIndex) -> float:
        """
        Estimate the fixation probability of an invading strategy in a resident population.
        
        Raises
        ------
        IndexError
            If a strategy index is not in ``[0, nb_strategies)``.
        ValueError
            If the invading and resident strategies are the same.
        """
    def estimate_mean_absorption_time(self, beta: typing.SupportsFloat | typing.SupportsIndex, init_state: typing.Annotated[numpy.typing.NDArray[numpy.uint64], "[m, 1]"], nb_runs: typing.SupportsInt | typing.SupportsIndex) -> dict:
        """
        Estimate the mean absorption time (fixation time) from a given initial state.
        
        Runs independent trajectories of the mutation-free Moran process from ``init_state``
        and counts generations until any strategy reaches ``pop_size``.
        
        Parameters
        ----------
        beta : float
            Intensity of selection.
        init_state : numpy.ndarray
            Initial population state — array of strategy counts summing to ``pop_size``.
        nb_runs : int
            Number of independent trajectories.
        
        Returns
        -------
        dict with keys ``"mean"`` (float), ``"stderr"`` (float), ``"nb_runs"`` (int).
        """
    def estimate_stationary_distribution(self, nb_runs: typing.SupportsInt | typing.SupportsIndex, nb_generations: typing.SupportsInt | typing.SupportsIndex, transitory: typing.SupportsInt | typing.SupportsIndex, beta: typing.SupportsFloat | typing.SupportsIndex, mu: typing.SupportsFloat | typing.SupportsIndex, tolerance: typing.SupportsFloat | typing.SupportsIndex = 0.0, check_every: typing.SupportsInt | typing.SupportsIndex = 0) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"]:
        """
        Estimate the stationary distribution of population states.
        
        When ``tolerance > 0``, runs are processed in batches of ``check_every``
        (default: ``max(1, nb_runs // 10)``). After each batch the L1 norm of the
        change in the normalised estimate is computed; if it falls below ``tolerance``
        the simulation stops early. This can save significant computation when the
        distribution converges before all ``nb_runs`` are exhausted.
        
        .. warning::
           If ``mu * (nb_generations - transitory)`` is much less than 10 (i.e. fewer
           than ~10 mutations are expected in the counting window) a ``UserWarning`` is
           raised. The geometric-skip approximation becomes inaccurate in this regime.
           Increase ``nb_generations``, decrease ``transitory``, or raise ``mu``.
        
        Parameters
        ----------
        nb_runs : int
            Maximum number of independent simulation runs.
        nb_generations : int
            Number of generations per run.
        transitory : int
            Transient period (generations not counted toward the distribution).
        beta : float
            Intensity of selection.
        mu : float
            Mutation probability (must be > 0).
        tolerance : float, optional
            Convergence threshold on the L1 norm of the change between consecutive
            batch estimates. 0.0 (default) disables early stopping.
        check_every : int, optional
            Number of runs per convergence-check batch. 0 (default) uses
            ``max(1, nb_runs // 10)``.
        
        Returns
        -------
        numpy.ndarray
            Estimated stationary distribution.
        """
    def estimate_stationary_distribution_sparse(self, nb_runs: typing.SupportsInt | typing.SupportsIndex, nb_generations: typing.SupportsInt | typing.SupportsIndex, transitory: typing.SupportsInt | typing.SupportsIndex, beta: typing.SupportsFloat | typing.SupportsIndex, mu: typing.SupportsFloat | typing.SupportsIndex, tolerance: typing.SupportsFloat | typing.SupportsIndex = 0.0, check_every: typing.SupportsInt | typing.SupportsIndex = 0) -> scipy.sparse.csr_matrix:
        """
        Estimate the stationary distribution in sparse format.
        
        Identical to ``estimate_stationary_distribution`` but returns a sparse matrix.
        Use this method when the number of population states is very large, since most
        entries of the stationary distribution will be zero.
        
        .. warning::
           If ``mu * (nb_generations - transitory)`` is much less than 10 a
           ``UserWarning`` is raised. See ``estimate_stationary_distribution`` for details.
        
        Parameters
        ----------
        nb_runs : int
            Maximum number of independent simulation runs.
        nb_generations : int
            Number of generations per run.
        transitory : int
            Transient period (generations not counted toward the distribution).
        beta : float
            Intensity of selection.
        mu : float
            Mutation probability (must be > 0).
        tolerance : float, optional
            Convergence threshold on the L1 norm; 0.0 disables early stopping.
        check_every : int, optional
            Runs per convergence-check batch; 0 uses ``max(1, nb_runs // 10)``.
        
        Returns
        -------
        scipy.sparse.csr_matrix
            Estimated stationary distribution in sparse format.
        """
    def estimate_stationary_indicators(self, indicators: typing.Any, nb_runs: typing.SupportsInt | typing.SupportsIndex, nb_generations: typing.SupportsInt | typing.SupportsIndex, transitory: typing.SupportsInt | typing.SupportsIndex, beta: typing.SupportsFloat | typing.SupportsIndex, mu: typing.SupportsFloat | typing.SupportsIndex, indicator_type: str = 'state', group_size: typing.Any = None, tolerance: typing.SupportsFloat | typing.SupportsIndex = 0.0, check_every: typing.SupportsInt | typing.SupportsIndex = 0, confidence: typing.SupportsFloat | typing.SupportsIndex = 0.95, verbose: bool = False, n_bootstrap: typing.SupportsInt | typing.SupportsIndex = 9999, precompute_limit: typing.SupportsInt | typing.SupportsIndex = 20000000) -> typing.Any:
        """
        Estimate expected indicator values under the stationary distribution.
        
        Runs stochastic simulations and accumulates indicator values at each
        post-transitory step.  The time-average converges to the true expectation by
        the ergodic theorem without storing the full stationary distribution.
        
        For ``indicator_type='state'``, when ``nb_states * len(indicators)`` exceeds
        ``precompute_limit`` this method automatically falls back to evaluating the
        indicators directly on the live simulation state at each recorded step,
        instead of precomputing a dense ``(nb_states, nb_indicators)`` matrix
        upfront.  This keeps memory use at ``O(nb_indicators)`` regardless of the
        size of the state space, at the cost of a Python callback per indicator per
        recorded generation (slower per-step than the matrix lookup, but the only
        way to stay within memory when ``nb_states`` is too large to enumerate).
        This fallback is not yet available for ``indicator_type='group'``, which
        raises a clear error instead of attempting the same allocation.
        
        Parameters
        ----------
        indicators : callable or list[callable]
            One or more indicator functions.
        
            - ``indicator_type='state'``: receives a population state as
              ``np.ndarray`` of shape ``(nb_strategies,)`` with integer counts summing
              to ``pop_size``, returns a ``float``.  Use for quantities like the
              fraction of cooperators.
        
            - ``indicator_type='group'``: receives a group configuration as
              ``np.ndarray`` of shape ``(nb_strategies,)`` summing to ``group_size``,
              returns a ``float``.  The expectation is marginalised over group configs
              using the multivariate hypergeometric distribution.  Requires
              ``group_size``.
        
        nb_runs : int
            Maximum number of independent simulation runs.
        nb_generations : int
            Number of generations per run.
        transitory : int
            Transitory period (generations excluded from accumulation).
        beta : float
            Intensity of selection.
        mu : float
            Mutation probability (must be > 0).
        indicator_type : {'state', 'group'}, default 'state'
            Whether indicators operate on population states or group configurations.
        group_size : int, optional
            Required when ``indicator_type='group'``.
        tolerance : float, default 0.0
            L1 convergence threshold on column-means between batches.  0.0 disables
            early stopping.
        check_every : int, default 0
            Batch size for convergence checks.  0 → ``max(1, nb_runs // 10)``.
        confidence : float, default 0.95
            Confidence level for the bootstrap CI.
        verbose : bool, default False
            If ``True``, attach per-run values to the result.
        n_bootstrap : int, default 9999
            Number of bootstrap resamples.
        precompute_limit : int, default 20_000_000
            Maximum number of elements (``nb_states * len(indicators)``) allowed in
            the precomputed indicator matrix.  Above this, ``indicator_type='state'``
            switches automatically to a direct, memory-bounded estimator (see above);
            ``indicator_type='group'`` raises instead.
        
        Returns
        -------
        StationaryIndicatorResult
            ``.mean`` — grand mean, shape ``(nb_indicators,)``.
            ``.confidence_interval`` — ``(low, high)`` non-parametric bootstrap CI.
            ``.nb_runs_used`` — runs actually completed.
            ``.converged`` — ``True`` if tolerance-based early stopping triggered.
            ``.per_run_values`` — per-run means ``(nb_runs_used, nb_indicators)``
            when ``verbose=True``, else ``None``.
        
        See Also
        --------
        estimate_stationary_indicators_precomputed : low-level fast path accepting a
            precomputed indicator matrix directly.
        egttools.precompute_group_to_state_indicator_matrix : build the indicator
            matrix manually for repeated reuse.
        """
    def estimate_stationary_indicators_precomputed(self, nb_runs: typing.SupportsInt | typing.SupportsIndex, nb_generations: typing.SupportsInt | typing.SupportsIndex, transitory: typing.SupportsInt | typing.SupportsIndex, beta: typing.SupportsFloat | typing.SupportsIndex, mu: typing.SupportsFloat | typing.SupportsIndex, indicator_values: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]", "flags.c_contiguous"], tolerance: typing.SupportsFloat | typing.SupportsIndex = 0.0, check_every: typing.SupportsInt | typing.SupportsIndex = 0) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]"]:
        """
        Estimate expected indicator values under the stationary distribution without
        computing the full distribution first.
        
        At each post-transitory simulation step the method looks up the precomputed
        indicator values for the current population state and accumulates them.  The
        per-run time-average converges to ``E[f_k] = sum_s sd(s) * indicator_values(s, k)``
        by the ergodic theorem.
        
        ``indicator_values`` must be a dense matrix of shape
        ``(nb_states, nb_indicators)`` where row ``s`` contains the values of all
        indicators for the population state at index ``s``.
        
        - For **state-level indicators** ``f(state)``: build ``indicator_values``
          by evaluating ``f`` on ``egttools.sample_simplex(s, pop_size, nb_strategies)``
          for each state index ``s``.
        - For **group-level indicators** ``f(group_config)``: use
          ``egttools.precompute_group_to_state_indicator_matrix`` to marginalise over
          group configurations first, then pass the resulting matrix here.
        
        Returns a matrix of shape ``(nb_runs_used, nb_indicators)`` — one row per
        completed run.  Prefer the high-level method
        ``estimate_stationary_indicators`` (same class) which accepts Python callables,
        builds the indicator matrix automatically, and returns a
        ``StationaryIndicatorResult`` with mean and bootstrap CI.
        
        .. warning::
           If ``mu * (nb_generations - transitory)`` is much less than 10 a
           ``UserWarning`` is raised.  See ``estimate_stationary_distribution`` for
           details.
        
        Parameters
        ----------
        nb_runs : int
            Maximum number of independent simulation runs.
        nb_generations : int
            Number of generations per run.
        transitory : int
            Transitory period (not counted toward indicator accumulation).
        beta : float
            Intensity of selection.
        mu : float
            Mutation probability (must be > 0).
        indicator_values : numpy.ndarray
            Precomputed matrix of shape (nb_states, nb_indicators).
        tolerance : float, optional
            L1 convergence threshold on column-means between batches.
            0.0 (default) disables early stopping.
        check_every : int, optional
            Batch size for convergence checks; 0 uses max(1, nb_runs // 10).
        
        Returns
        -------
        numpy.ndarray
            Per-run means of shape (nb_runs_used, nb_indicators).
        """
    def estimate_strategy_distribution(self, nb_runs: typing.SupportsInt | typing.SupportsIndex, nb_generations: typing.SupportsInt | typing.SupportsIndex, transitory: typing.SupportsInt | typing.SupportsIndex, beta: typing.SupportsFloat | typing.SupportsIndex, mu: typing.SupportsFloat | typing.SupportsIndex, tolerance: typing.SupportsFloat | typing.SupportsIndex = 0.0, check_every: typing.SupportsInt | typing.SupportsIndex = 0) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"]:
        """
        Estimate the average frequency of each strategy over time.
        
        This method bypasses state indexing and is safe when the total number of
        population states exceeds ``MAX_LONG_INT``.
        
        .. warning::
           If ``mu * (nb_generations - transitory)`` is much less than 10 a
           ``UserWarning`` is raised. See ``estimate_stationary_distribution`` for details.
        
        Parameters
        ----------
        nb_runs : int
            Maximum number of independent simulation runs.
        nb_generations : int
            Number of generations per run.
        transitory : int
            Transient period (generations not counted toward the distribution).
        beta : float
            Intensity of selection.
        mu : float
            Mutation probability (must be > 0).
        tolerance : float, optional
            Convergence threshold on the L1 norm; 0.0 disables early stopping.
        check_every : int, optional
            Runs per convergence-check batch; 0 uses ``max(1, nb_runs // 10)``.
        
        Returns
        -------
        numpy.ndarray
            Average frequency of each strategy.
        """
    def evolve(self, nb_generations: typing.SupportsInt | typing.SupportsIndex, beta: typing.SupportsFloat | typing.SupportsIndex, mu: typing.SupportsFloat | typing.SupportsIndex, init_state: typing.Annotated[numpy.typing.NDArray[numpy.uint64], "[m, 1]"]) -> typing.Annotated[numpy.typing.NDArray[numpy.uint64], "[m, 1]"]:
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
    def run(self, *args, **kwargs) -> typing.Any:
        """
        Removed. Raises NotImplementedError; use ``run_with_mutation`` or ``run_without_mutation``.
        """
    @typing.overload
    def run_with_mutation(self, nb_generations: typing.SupportsInt | typing.SupportsIndex, beta: typing.SupportsFloat | typing.SupportsIndex, mu: typing.SupportsFloat | typing.SupportsIndex, init_state: typing.Annotated[numpy.typing.NDArray[numpy.uint64], "[m, 1]"]) -> typing.Annotated[numpy.typing.NDArray[numpy.uint64], "[m, n]"]:
        """
        Simulate the stochastic dynamics with mutation.
        
        Returns
        -------
        numpy.ndarray
            Matrix containing all intermediate population states.
        """
    @typing.overload
    def run_with_mutation(self, nb_generations: typing.SupportsInt | typing.SupportsIndex, transient: typing.SupportsInt | typing.SupportsIndex, beta: typing.SupportsFloat | typing.SupportsIndex, mu: typing.SupportsFloat | typing.SupportsIndex, init_state: typing.Annotated[numpy.typing.NDArray[numpy.uint64], "[m, 1]"]) -> typing.Annotated[numpy.typing.NDArray[numpy.uint64], "[m, n]"]:
        """
        Simulate the stochastic dynamics with mutation, skipping the transient phase.
        
        Returns
        -------
        numpy.ndarray
            Matrix containing the population states after the transient period.
        """
    @typing.overload
    def run_without_mutation(self, nb_generations: typing.SupportsInt | typing.SupportsIndex, beta: typing.SupportsFloat | typing.SupportsIndex, init_state: typing.Annotated[numpy.typing.NDArray[numpy.uint64], "[m, 1]"]) -> typing.Annotated[numpy.typing.NDArray[numpy.uint64], "[m, n]"]:
        """
        Simulate the stochastic dynamics without mutation.
        
        Returns
        -------
        numpy.ndarray
            Matrix containing all intermediate population states.
        """
    @typing.overload
    def run_without_mutation(self, nb_generations: typing.SupportsInt | typing.SupportsIndex, transient: typing.SupportsInt | typing.SupportsIndex, beta: typing.SupportsFloat | typing.SupportsIndex, init_state: typing.Annotated[numpy.typing.NDArray[numpy.uint64], "[m, 1]"]) -> typing.Annotated[numpy.typing.NDArray[numpy.uint64], "[m, n]"]:
        """
        Simulate the stochastic dynamics without mutation, skipping the transient phase.
        
        Returns
        -------
        numpy.ndarray
            Matrix containing the population states after the transient period.
        """
    def set_mutation_matrix(self, mutation_matrix: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]", "flags.c_contiguous"]) -> None:
        """
        Set a full source-strategy-dependent mutation bias.
        
        Row ``i`` of ``mutation_matrix`` is the (unnormalized) distribution over
        target strategies when an individual currently playing strategy ``i``
        mutates. The diagonal is always ignored (mutation always changes strategy).
        Each row must have at least one strictly positive entry among the other
        strategies.
        
        Parameters
        ----------
        mutation_matrix : numpy.ndarray
            Shape ``(nb_strategies, nb_strategies)``, non-negative entries.
        
        See Also
        --------
        set_mutation_weights : convenience method accepting a single vector (bias
            shared by all source strategies) or a full matrix.
        """
    def set_mutation_weights(self, weights: typing.Any) -> None:
        """
        Set the mutation bias. Accepts either shape:
        
        - 1D, length ``nb_strategies``: a target-strategy bias shared by all source
          strategies (broadcast into every row of the mutation matrix; equivalent to
          ``set_mutation_matrix`` with every row equal to this vector and the
          diagonal zeroed).
        - 2D, shape ``(nb_strategies, nb_strategies)``: row ``i`` is the bias over
          target strategies when mutating away from strategy ``i`` (equivalent to
          calling ``set_mutation_matrix`` directly).
        
        Diagonal entries are always ignored (mutation always changes strategy).
        Each row must have at least one positive entry among the other strategies,
        otherwise mutating away from that strategy would have no valid target and
        this raises. A solver's mutation is uniform by default; call this to bias it.
        
        Parameters
        ----------
        weights : array_like
            1D (length ``nb_strategies``) or 2D (``nb_strategies x nb_strategies``).
        """
    @property
    def cache_size(self) -> int:
        """
        Maximum number of cached fitness values.
        """
    @cache_size.setter
    def cache_size(self, arg1: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    @property
    def mutation_matrix(self) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]"]:
        """
        Current (nb_strategies, nb_strategies) mutation matrix. Row i is the (unnormalized) distribution over target strategies when mutating away from strategy i; the diagonal is always 0. Defaults to uniform (all-ones off-diagonal) until set_mutation_weights is called.
        """
    @property
    def nb_states(self) -> int:
        """
        Number of discrete states in the population.
        """
    @property
    def nb_strategies(self) -> int:
        """
        Number of strategies in the population.
        """
    @property
    def payoffs(self) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]"]:
        """
        Payoff matrix used for selection dynamics.
        """
    @property
    def pop_size(self) -> int:
        """
        Current population size.
        """
    @pop_size.setter
    def pop_size(self, arg1: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
class PairwiseComparisonTransitionOperator:
    """
    
    Matrix-free transition operator for the pairwise comparison process.
    
    Computes matrix-vector products ``y = P x``, ``y = P^T x``, and
    ``y = (I - P^T) x`` without ever assembling the transition matrix P.
    Designed for iterative eigensolvers (``scipy.sparse.linalg``, petsc4py)
    and as the basis for future MPI-distributed computation.
    
    The stationary distribution π satisfies ``P^T π = π``. Use
    ``apply_transpose`` or wrap this object with
    ``egttools.numerical.linear_operator.make_transition_operator`` to
    obtain a ``scipy.sparse.linalg.LinearOperator``.
    """
    @staticmethod
    def _pybind11_conduit_v1_(*args, **kwargs):
        ...
    def __init__(self, population_size: typing.SupportsInt | typing.SupportsIndex, game: games.AbstractGame, beta: typing.SupportsFloat | typing.SupportsIndex, mu: typing.SupportsFloat | typing.SupportsIndex) -> None:
        """
        Construct the matrix-free transition operator.
        
        Parameters
        ----------
        population_size : int
            Number of individuals Z (must be >= 2).
        game : egttools.games.AbstractGame
            Game object defining strategy fitnesses.
        beta : float
            Intensity of selection (Fermi parameter, >= 0).
        mu : float
            Mutation probability per step (in [0, 1]).
        """
    def apply(self, x: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"], y: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]", "flags.writeable"]) -> None:
        """
        Compute y = P x in-place.
        
        Parameters
        ----------
        x : numpy.ndarray
            Input vector of length ``size``.
        y : numpy.ndarray
            Output vector of length ``size``; zeroed and overwritten.
        """
    def apply_residual(self, x: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"], y: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]", "flags.writeable"]) -> None:
        """
        Compute y = (I - P^T) x in-place.
        
        Useful for iterative linear solvers: find π such that (I - P^T) π = 0.
        
        Parameters
        ----------
        x : numpy.ndarray
            Input vector of length ``size``.
        y : numpy.ndarray
            Output vector of length ``size``; overwritten.
        """
    def apply_transpose(self, x: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"], y: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]", "flags.writeable"]) -> None:
        """
        Compute y = P^T x in-place.
        
        The stationary distribution π satisfies P^T π = π, so this is the
        primary operation for iterative eigensolver use.
        
        Parameters
        ----------
        x : numpy.ndarray
            Input vector of length ``size``.
        y : numpy.ndarray
            Output vector of length ``size``; zeroed and overwritten.
        """
    def compute_stationary_distribution(self, tol: typing.SupportsFloat | typing.SupportsIndex = 1e-10, max_iter: typing.SupportsInt | typing.SupportsIndex = 10000) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"]:
        """
        Compute the stationary distribution via power iteration (pure C++).
        
        Iterates π ← P^T π / ‖P^T π‖₁ until L1 convergence or *max_iter* is
        reached.  Runs entirely in C++ with no Python callbacks — much faster
        than wrapping the operator in a ``scipy.sparse.linalg.LinearOperator``
        for large state spaces.
        
        Parameters
        ----------
        tol : float
            L1 convergence threshold (default 1e-10).
        max_iter : int
            Maximum number of power-iteration steps (default 10000).
        
        Returns
        -------
        numpy.ndarray
            Normalised stationary distribution of length ``size``.
        
        Raises
        ------
        RuntimeError
            If convergence is not reached within *max_iter* iterations.
        """
    @property
    def beta(self) -> float:
        """
        Intensity of selection β.
        """
    @property
    def mu(self) -> float:
        """
        Mutation probability μ.
        """
    @property
    def nb_strategies(self) -> int:
        """
        Number of strategies k.
        """
    @property
    def population_size(self) -> int:
        """
        Population size Z.
        """
    @property
    def size(self) -> int:
        """
        Total number of simplex states C(Z+k-1, k-1).
        """
def calculate_expected_group_success(pop_size: typing.SupportsInt | typing.SupportsIndex, group_size: typing.SupportsInt | typing.SupportsIndex, nb_strategies: typing.SupportsInt | typing.SupportsIndex, stationary_distribution: scipy.sparse.csr_matrix, threshold: typing.SupportsInt | typing.SupportsIndex, contributing_strategies: collections.abc.Sequence[typing.SupportsInt | typing.SupportsIndex]) -> float:
    """
    Calculate the expected group success eta_G under the stationary distribution.
    
    eta_G = sum_s sd(s) * sum_g P(g|s) * I(sum_{k in contributing_strategies} g[k] >= threshold)
    
    Any strategy whose index appears in ``contributing_strategies`` contributes its
    group count towards the threshold check.  This supports games where multiple
    strategy types each contribute to collective success (e.g. cooperators + altruists).
    
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
    >>> # CRD: only cooperators (strategy 0) count, threshold = 3
    >>> eta_G = calculate_expected_group_success(
    ...     pop_size, group_size, nb_strategies, sd,
    ...     threshold=3, contributing_strategies=[0]
    ... )
    >>> # Extended CRD: cooperators (0) and altruists (2) both count
    >>> eta_G = calculate_expected_group_success(
    ...     pop_size, group_size, nb_strategies, sd,
    ...     threshold=3, contributing_strategies=[0, 2]
    ... )
    """
def calculate_expected_indicator(pop_size: typing.SupportsInt | typing.SupportsIndex, group_size: typing.SupportsInt | typing.SupportsIndex, nb_strategies: typing.SupportsInt | typing.SupportsIndex, stationary_distribution: scipy.sparse.csr_matrix, indicator: collections.abc.Callable[[list[int]], typing.SupportsFloat | typing.SupportsIndex]) -> float:
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
    >>> # Expected group success for CRD: cooperators are strategy 0, threshold = 3
    >>> eta_G = calculate_expected_indicator(
    ...     pop_size, group_size, nb_strategies, sd,
    ...     lambda g: float(g[0] >= 3)
    ... )
    """
def calculate_expected_indicators(pop_size: typing.SupportsInt | typing.SupportsIndex, group_size: typing.SupportsInt | typing.SupportsIndex, nb_strategies: typing.SupportsInt | typing.SupportsIndex, stationary_distribution: scipy.sparse.csr_matrix, indicators: collections.abc.Sequence[collections.abc.Callable[[list[int]], typing.SupportsFloat | typing.SupportsIndex]]) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"]:
    """
    Calculate E[f_k] for multiple indicator functions in a single pass.
    
    Equivalent to calling calculate_expected_indicator once per indicator, but the
    multivariate hypergeometric PDF is computed only once per (state, group_config) pair
    and shared across all indicators.  Cost is O(states × groups + K) rather than
    O(K × states × groups).
    
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
    indicators : list[callable]
        List of functions, each with signature ``f(group_config: list[int]) -> float``.
    
    Returns
    -------
    numpy.ndarray
        One-dimensional array of length ``len(indicators)``; element k is the expected
        value of ``indicators[k]``.
    
    Examples
    --------
    >>> # Compute cooperation level and group success simultaneously
    >>> results = calculate_expected_indicators(
    ...     pop_size, group_size, nb_strategies, sd,
    ...     [
    ...         lambda g: g[0] / group_size,       # cooperation level
    ...         lambda g: float(g[0] >= threshold), # group success
    ...     ]
    ... )
    >>> cooperation_level, eta_G = results
    """
def calculate_expected_indicators_precomputed(pop_size: typing.SupportsInt | typing.SupportsIndex, group_size: typing.SupportsInt | typing.SupportsIndex, nb_strategies: typing.SupportsInt | typing.SupportsIndex, stationary_distribution: scipy.sparse.csr_matrix, indicator_matrix: typing.Annotated[numpy.typing.ArrayLike, numpy.float64, "[m, n]"]) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"]:
    """
    Compute expected indicators from a precomputed indicator matrix (pure C++, GIL released).
    
    This is the fast path when the indicator values per group configuration are already known.
    Precomputing the indicator matrix in Python (e.g. via numpy) and then calling this function
    avoids all Python callbacks inside the hot loop and enables full GIL release.
    
    The matrix ``indicator_matrix[g, k]`` must contain the value of indicator ``k`` for group
    configuration ``g``.  For boolean indicators use 0.0 / 1.0.  The row order must match
    the group configuration enumeration of ``sample_simplex``, i.e. row ``g`` corresponds to
    ``sample_simplex(g, group_size, nb_strategies)``.
    
    result[k] = sum_s sd(s) * indicator_matrix[:, k] @ prob_vector_for_state_s
    
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
    indicator_matrix : numpy.ndarray
        Dense matrix of shape (nb_group_configs, nb_indicators).
    
    Returns
    -------
    numpy.ndarray
        Array of length ``nb_indicators``.
    
    Examples
    --------
    >>> nb_group_configs = calculate_nb_states(group_size, nb_strategies)
    >>> indicator_matrix = np.array([
    ...     [float(sample_simplex(g, group_size, nb_strategies)[0] >= threshold)]
    ...     for g in range(nb_group_configs)
    ... ])
    >>> eta_G = calculate_expected_indicators_precomputed(
    ...     pop_size, group_size, nb_strategies, sd, indicator_matrix
    ... )[0]
    """
def calculate_expected_payoff(pop_size: typing.SupportsInt | typing.SupportsIndex, group_size: typing.SupportsInt | typing.SupportsIndex, nb_strategies: typing.SupportsInt | typing.SupportsIndex, stationary_distribution: scipy.sparse.csr_matrix, payoff_matrix: typing.Annotated[numpy.typing.ArrayLike, numpy.float64, "[m, n]"]) -> float:
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
def calculate_expected_state_indicator(pop_size: typing.SupportsInt | typing.SupportsIndex, nb_strategies: typing.SupportsInt | typing.SupportsIndex, stationary_distribution: scipy.sparse.csr_matrix, indicator: collections.abc.Callable[[list[int]], typing.SupportsFloat | typing.SupportsIndex]) -> float:
    """
    Calculate E[f] = sum_s sd(s) * f(s) for a single state-level indicator.
    
    Unlike ``calculate_expected_indicator``, no group sampling is performed.
    The callable ``indicator`` receives the full population state vector
    (integer counts, sums to pop_size) and returns a scalar.
    
    Parameters
    ----------
    pop_size : int
        Total number of individuals in the population.
    nb_strategies : int
        Number of strategies available in the population.
    stationary_distribution : scipy.sparse.csr_matrix
        Sparse stationary distribution over population states.
    indicator : callable
        Function mapping a population state (list[int]) to a float.
    
    Returns
    -------
    float
        Expected value of the indicator.
    
    See Also
    --------
    egttools.calculate_expected_state_indicators
    egttools.calculate_expected_state_indicators_precomputed
    egttools.calculate_expected_indicator
    """
def calculate_expected_state_indicators(pop_size: typing.SupportsInt | typing.SupportsIndex, nb_strategies: typing.SupportsInt | typing.SupportsIndex, stationary_distribution: scipy.sparse.csr_matrix, indicators: collections.abc.Sequence[collections.abc.Callable[[list[int]], typing.SupportsFloat | typing.SupportsIndex]]) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"]:
    """
    Calculate E[f_k] = sum_s sd(s) * f_k(s) for multiple state-level indicators in one pass.
    
    Iterates over non-zero states in the stationary distribution exactly once,
    evaluating all indicators at each state.  Cost: O(nb_nonzero_states * K) where K
    is the number of indicators.
    
    Parameters
    ----------
    pop_size : int
        Total number of individuals in the population.
    nb_strategies : int
        Number of strategies available in the population.
    stationary_distribution : scipy.sparse.csr_matrix
        Sparse stationary distribution over population states.
    indicators : list[callable]
        List of functions, each mapping a population state (list[int]) to a float.
    
    Returns
    -------
    numpy.ndarray
        Vector of length len(indicators) with the expected value of each indicator.
    
    See Also
    --------
    egttools.calculate_expected_state_indicator
    egttools.calculate_expected_state_indicators_precomputed
    """
def calculate_expected_state_indicators_precomputed(stationary_distribution: scipy.sparse.csr_matrix, indicator_values: typing.Annotated[numpy.typing.ArrayLike, numpy.float64, "[m, n]"]) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"]:
    """
    Fast path: E[f_k] = sum_s sd(s) * indicator_values(s, k) using a precomputed matrix.
    
    The caller evaluates all indicators on all population states upfront and
    stores the results in ``indicator_values`` (shape: nb_states × nb_indicators).
    The computation reduces to a sparse-dense dot product per column — no Python
    callbacks, GIL fully released.
    
    Parameters
    ----------
    stationary_distribution : scipy.sparse.csr_matrix
        Sparse stationary distribution over population states.
    indicator_values : numpy.ndarray
        Dense matrix of shape (nb_states, nb_indicators).  Row s must contain the
        values of all indicators evaluated on the population state for index s.
    
    Returns
    -------
    numpy.ndarray
        Vector of length nb_indicators.
    
    See Also
    --------
    egttools.calculate_expected_state_indicator
    egttools.calculate_expected_indicators_precomputed
    egttools.numerical.PairwiseComparisonNumerical.estimate_stationary_indicators
    """
def calculate_hypergeometric_expected_value(pop_size: typing.SupportsInt | typing.SupportsIndex, group_size: typing.SupportsInt | typing.SupportsIndex, nb_strategies: typing.SupportsInt | typing.SupportsIndex, state: typing.Annotated[numpy.typing.NDArray[numpy.uint64], "[m, 1]"], f_values: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"]) -> float:
    """
    Compute E[f | state] = sum_g P(g | state) * f(g) for a single population state.
    
    Calculates the expected value of a function f over all group configurations g,
    weighted by the multivariate hypergeometric probability P(g | state) that a
    randomly sampled group of ``group_size`` individuals from a population in
    ``state`` has composition g.
    
    This is the inner loop used by fitness calculations in N-player games.  Exposing
    it here lets users write custom game fitness functions in Python without
    reimplementing the hypergeometric weighting.
    
    Parameters
    ----------
    pop_size : int
        Total number of individuals in the population.
    group_size : int
        Number of individuals sampled per group interaction.
    nb_strategies : int
        Number of distinct strategies.
    state : numpy.ndarray
        Integer array of length ``nb_strategies`` with counts of each strategy in
        the population.  Must sum to ``pop_size``.
    f_values : numpy.ndarray
        Float array of length ``calculate_nb_states(group_size, nb_strategies)``
        where ``f_values[g]`` is the value of f for the group configuration
        ``sample_simplex(g, group_size, nb_strategies)``.
    
    Returns
    -------
    float
        Expected value of f given the population state.
    
    Examples
    --------
    >>> import numpy as np
    >>> import egttools as egt
    >>> pop_size, group_size, nb_strategies = 10, 3, 2
    >>> state = np.array([6, 4], dtype=np.uint64)
    >>> nb_configs = egt.calculate_nb_states(group_size, nb_strategies)
    >>> # Cooperation level: fraction of cooperators (strategy 0) in the group
    >>> f_values = np.array([
    ...     egt.sample_simplex(g, group_size, nb_strategies)[0] / group_size
    ...     for g in range(nb_configs)
    ... ])
    >>> egt.calculate_hypergeometric_expected_value(pop_size, group_size, nb_strategies, state, f_values)
    """
def calculate_hypergeometric_fitness(player_type: typing.SupportsInt | typing.SupportsIndex, pop_size: typing.SupportsInt | typing.SupportsIndex, group_size: typing.SupportsInt | typing.SupportsIndex, nb_strategies: typing.SupportsInt | typing.SupportsIndex, strategies: typing.Annotated[numpy.typing.NDArray[numpy.uint64], "[m, 1]"], payoffs_row: typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"]) -> float:
    """
    Compute the fitness of a focal player (not included in ``strategies``) via hypergeometric sampling.
    
    This is the standard EGT fitness calculation for finite populations:
    
        fitness = sum_g  P(g | strategies, pop_size-1, group_size-1)  *  payoff(player_type, g)
    
    where g ranges over all group configurations that include the focal player (i.e.
    ``g[player_type] >= 1``), and P is the multivariate hypergeometric probability
    that the *remaining* group_size-1 slots are filled from the background population
    (which has ``strategies`` individuals, not including the focal player).
    
    Parameters
    ----------
    player_type : int
        Index of the focal player's strategy (0-based).
    pop_size : int
        Total population size *including* the focal player.
    group_size : int
        Number of individuals in the interaction group *including* the focal player.
    nb_strategies : int
        Number of distinct strategies.
    strategies : numpy.ndarray
        Integer array of length ``nb_strategies`` with counts of each strategy
        in the population *excluding* the focal player.  Must sum to
        ``pop_size - 1``.
    payoffs_row : numpy.ndarray
        Float array of length ``calculate_nb_states(group_size, nb_strategies)``
        where ``payoffs_row[g]`` is the payoff of ``player_type`` in group
        configuration ``sample_simplex(g, group_size, nb_strategies)``.
    
    Returns
    -------
    float
        Expected fitness of the focal player.
    
    Examples
    --------
    >>> import numpy as np
    >>> import egttools as egt
    >>> pop_size, group_size, nb_strategies = 10, 3, 2
    >>> # Focal player is a cooperator (strategy 1), population has 5 C, 4 D (excluding focal)
    >>> strategies = np.array([4, 5], dtype=np.uint64)
    >>> # payoffs_row[g] = payoff of cooperator in group config g
    >>> nb_configs = egt.calculate_nb_states(group_size, nb_strategies)
    >>> payoffs_row = np.zeros(nb_configs)
    >>> for g in range(nb_configs):
    ...     gc = egt.sample_simplex(g, group_size, nb_strategies)
    ...     payoffs_row[g] = gc[1] - 1.0  # cooperators pay cost 1
    >>> egt.calculate_hypergeometric_fitness(1, pop_size, group_size, nb_strategies, strategies, payoffs_row)
    """
def calculate_nb_states(group_size: typing.SupportsInt | typing.SupportsIndex, nb_strategies: typing.SupportsInt | typing.SupportsIndex) -> typing.Any:
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
@typing.overload
def calculate_state(group_size: typing.SupportsInt | typing.SupportsIndex, group_composition: collections.abc.Sequence[typing.SupportsInt | typing.SupportsIndex]) -> int:
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
    """
@typing.overload
def calculate_state(group_size: typing.SupportsInt | typing.SupportsIndex, group_composition: typing.Annotated[numpy.typing.NDArray[numpy.uint64], "[m, 1]"]) -> int:
    """
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
def calculate_strategies_distribution(pop_size: typing.SupportsInt | typing.SupportsIndex, nb_strategies: typing.SupportsInt | typing.SupportsIndex, stationary_distribution: scipy.sparse.csr_matrix) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"]:
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
def is_arpack_enabled() -> bool:
    """
    Check if EGTtools was compiled with ARPACK eigensolver support.
    """
def is_blas_lapack_enabled() -> bool:
    """
    Check if EGTtools was compiled with BLAS/LAPACK acceleration.
    """
def is_boost_enabled() -> bool:
    """
    Check if EGTtools was compiled with Boost support.
    """
def is_openmp_enabled() -> bool:
    """
    Check if EGTtools was compiled with OpenMP support.
    """
def is_petsc_enabled() -> bool:
    """
    Check if EGTtools was compiled with PETSc/SLEPc MPI eigensolver support.
    """
def precompute_group_to_state_indicator_matrix(pop_size: typing.SupportsInt | typing.SupportsIndex, group_size: typing.SupportsInt | typing.SupportsIndex, nb_strategies: typing.SupportsInt | typing.SupportsIndex, indicators: collections.abc.Sequence[collections.abc.Callable[[list[int]], typing.SupportsFloat | typing.SupportsIndex]]) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]"]:
    """
    Build a state-level indicator matrix from group-level callables.
    
    Converts group-level indicators ``f_k(group_config)`` to a state-level matrix by
    marginalising over group configurations using the multivariate hypergeometric
    distribution:
    
        indicator_values(s, k) = sum_g  P(g | s) * f_k(g)
    
    where P(g | s) is the multivariate hypergeometric probability.  The returned
    matrix can be passed directly to
    ``calculate_expected_state_indicators_precomputed`` or to
    ``PairwiseComparisonNumerical.estimate_stationary_indicators``.
    
    For group_size == 2, pairwise probabilities are computed with simple
    combinatorics instead of the full hypergeometric formula.
    
    Parameters
    ----------
    pop_size : int
        Total number of individuals in the population.
    group_size : int
        Number of individuals sampled per group interaction.
    nb_strategies : int
        Number of strategies available in the population.
    indicators : list[callable]
        Group-level indicator functions, each mapping a group configuration
        (list[int], sums to group_size) to a float.
    
    Returns
    -------
    numpy.ndarray
        Dense matrix of shape (nb_states, len(indicators)).
    
    See Also
    --------
    egttools.calculate_expected_state_indicators_precomputed
    egttools.numerical.PairwiseComparisonNumerical.estimate_stationary_indicators
    """
@typing.overload
def replicator_equation(frequencies: typing.Annotated[numpy.typing.ArrayLike, numpy.float64, "[m, 1]"], payoff_matrix: typing.Annotated[numpy.typing.ArrayLike, numpy.float64, "[m, n]"]) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"]:
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
    """
@typing.overload
def replicator_equation(frequencies: typing.Annotated[numpy.typing.ArrayLike, numpy.float64, "[m, 1]"], game: games.AbstractReplicatorGame) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"]:
    """
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
@typing.overload
def replicator_equation_n_player(frequencies: typing.Annotated[numpy.typing.ArrayLike, numpy.float64, "[m, 1]"], payoff_matrix: typing.Annotated[numpy.typing.ArrayLike, numpy.float64, "[m, n]"], group_size: typing.SupportsInt | typing.SupportsIndex) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"]:
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
    """
@typing.overload
def replicator_equation_n_player(frequencies: typing.Annotated[numpy.typing.ArrayLike, numpy.float64, "[m, 1]"], game: games.AbstractReplicatorGame) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"]:
    """
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
def sample_simplex(index: typing.SupportsInt | typing.SupportsIndex, pop_size: typing.SupportsInt | typing.SupportsIndex, nb_strategies: typing.SupportsInt | typing.SupportsIndex) -> typing.Annotated[numpy.typing.NDArray[numpy.uint64], "[m, 1]"]:
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
def sample_simplex_directly(nb_strategies: typing.SupportsInt | typing.SupportsIndex, pop_size: typing.SupportsInt | typing.SupportsIndex) -> typing.Annotated[numpy.typing.NDArray[numpy.int64], "[m, 1]"]:
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
def sample_unit_simplex(nb_strategies: typing.SupportsInt | typing.SupportsIndex) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 1]"]:
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
def vectorized_replicator_equation(x1: typing.Annotated[numpy.typing.ArrayLike, numpy.float64, "[m, n]"], x2: typing.Annotated[numpy.typing.ArrayLike, numpy.float64, "[m, n]"], x3: typing.Annotated[numpy.typing.ArrayLike, numpy.float64, "[m, n]"], game: games.AbstractReplicatorGame) -> tuple[typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]"], typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]"], typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]"]]:
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
@typing.overload
def vectorized_replicator_equation_n_player(x1: typing.Annotated[numpy.typing.ArrayLike, numpy.float64, "[m, n]"], x2: typing.Annotated[numpy.typing.ArrayLike, numpy.float64, "[m, n]"], x3: typing.Annotated[numpy.typing.ArrayLike, numpy.float64, "[m, n]"], payoff_matrix: typing.Annotated[numpy.typing.ArrayLike, numpy.float64, "[m, n]"], group_size: typing.SupportsInt | typing.SupportsIndex) -> tuple[typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]"], typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]"], typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]"]]:
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
    """
@typing.overload
def vectorized_replicator_equation_n_player(x1: typing.Annotated[numpy.typing.ArrayLike, numpy.float64, "[m, n]"], x2: typing.Annotated[numpy.typing.ArrayLike, numpy.float64, "[m, n]"], x3: typing.Annotated[numpy.typing.ArrayLike, numpy.float64, "[m, n]"], game: games.AbstractReplicatorGame) -> tuple[typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]"], typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]"], typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, n]"]]:
    """
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
__init__: str = 'The `numerical` module contains optimized functions and classes to simulate evolutionary dynamics in large populations.'
__version__: str = '0.2.0'
__version_info__: tuple = (0, 2, 0)

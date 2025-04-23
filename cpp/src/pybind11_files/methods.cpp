/** Copyright (c) 2022-2025  Elias Fernandez
*
* This file is part of EGTtools.
*
* EGTtools is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* EGTtools is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with EGTtools.  If not, see <http://www.gnu.org/licenses/>
*/

#include "methods.hpp"


using namespace egttools;
using PairwiseComparison = FinitePopulations::PairwiseComparisonNumerical<>;

namespace egttools {
    VectorXli sample_simplex_directly(const int64_t nb_strategies, const int64_t pop_size) {
        std::mt19937_64 generator(egttools::Random::SeedGenerator::getInstance().getSeed());
        egttools::VectorXli state = egttools::VectorXli::Zero(nb_strategies);

        egttools::FinitePopulations::sample_simplex_direct_method<long int, long int, egttools::VectorXli,
            std::mt19937_64>(nb_strategies, pop_size, state, generator);

        return state;
    }

    Vector sample_unit_simplex(const int64_t nb_strategies) {
        std::mt19937_64 generator(egttools::Random::SeedGenerator::getInstance().getSeed());
        const auto real_rand = std::uniform_real_distribution<double>(0, 1);
        egttools::Vector state = egttools::Vector::Zero(nb_strategies);
        egttools::FinitePopulations::sample_unit_simplex<int64_t, std::mt19937_64>(
            nb_strategies, state, real_rand, generator);

        return state;
    }
} // namespace egttools

void init_methods(py::module_ &m) { {
        py::options options;
        options.disable_function_signatures();

        m.def(
            "calculate_state",
            static_cast<size_t (*)(const size_t &, const egttools::Factors &)>(&
                egttools::FinitePopulations::calculate_state),
            R"pbdoc(
        Converts a discrete population configuration into a unique index.

        This is typically used to map a population state (i.e., counts of each strategy)
        to a 1D index, which is useful for vectorized representations and algorithms
        like replicator dynamics or finite Markov chains.

        Parameters
        ----------
        group_size : int
            The total number of individuals (e.g., population size or group size).
        group_composition : List[int]
            A list representing the number of individuals using each strategy.

        Returns
        -------
        int
            A unique index corresponding to the group composition.

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
        )pbdoc",
            py::arg("group_size"),
            py::arg("group_composition")
        );

        m.def(
            "calculate_state",
            static_cast<size_t (*)(const size_t &, const Eigen::Ref<const egttools::VectorXui> &)>(&
                egttools::FinitePopulations::calculate_state),
            R"pbdoc(
        Converts a discrete population configuration (NumPy vector) into a unique index.

        This version takes an integer vector from NumPy and maps it to a 1D index, useful
        for discrete state space indexing.

        Parameters
        ----------
        group_size : int
            The total number of individuals in the population.
        group_composition : NDArray[np.int64]
            NumPy array of shape (n,) where n is the number of strategies.

        Returns
        -------
        int
            A unique index corresponding to the input configuration.

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
        )pbdoc",
            py::arg("group_size"),
            py::arg("group_composition")
        );


        m.def(
            "sample_simplex",
            static_cast<egttools::VectorXui (*)(size_t, const size_t &, const size_t &)>(&
                egttools::FinitePopulations::sample_simplex),
            R"pbdoc(
        Converts a state index into a group composition vector.

        This function performs the inverse of `calculate_state`, returning a vector
        representing the number of individuals using each strategy from a given index.

        Parameters
        ----------
        index : int
            Index of the population state (from 0 to total number of states - 1).
        pop_size : int
            Population size (total number of individuals in the group).
        nb_strategies : int
            Number of available strategies.

        Returns
        -------
        NDArray[np.int64]
            A vector of length `nb_strategies` where each entry represents the
            number of individuals using the corresponding strategy.

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
        )pbdoc",
            py::arg("index"),
            py::arg("pop_size"),
            py::arg("nb_strategies"),
            py::return_value_policy::move
        );
        m.def(
            "sample_simplex_directly",
            &sample_simplex_directly,
            R"pbdoc(
        Samples a discrete population state uniformly at random from the simplex.

        This method uses a direct sampling approach to draw a single composition
        of strategies such that the total population size is preserved.

        Parameters
        ----------
        nb_strategies : int
            Number of available strategies.
        pop_size : int
            Total number of individuals in the population.

        Returns
        -------
        NDArray[np.int64]
            A vector of length `nb_strategies`, where each entry indicates how
            many individuals adopt the corresponding strategy.

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
        )pbdoc",
            py::arg("nb_strategies"),
            py::arg("pop_size"),
            py::return_value_policy::move
        );

        m.def(
            "sample_unit_simplex",
            &sample_unit_simplex,
            R"pbdoc(
        Samples a continuous strategy composition uniformly at random from the unit simplex.

        This function generates a random vector of non-negative floats that sum to 1.
        It is typically used to initialize strategy distributions in infinite population models.

        Parameters
        ----------
        nb_strategies : int
            Number of strategies in the population.

        Returns
        -------
        NDArray[np.float64]
            A 1D array of length `nb_strategies`, representing a point in the unit simplex
            (i.e., a valid probability distribution over strategies).

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
        )pbdoc",
            py::arg("nb_strategies"),
            py::return_value_policy::move
        );


#if (HAS_BOOST)
        m.def(
            "calculate_nb_states",
            [](const size_t group_size, const size_t nb_strategies) {
                auto result = starsBars<size_t, boost::multiprecision::cpp_int>(group_size, nb_strategies);
                return py::cast(result);
            },
            R"pbdoc(
        Calculates the number of possible states in a discrete simplex.

        This corresponds to the number of integer compositions of a population of size `group_size`
        into `nb_strategies` categories. Internally, it uses the "stars and bars" combinatorial formula.

        Parameters
        ----------
        group_size : int
            Size of the population or group (number of "stars").
        nb_strategies : int
            Number of available strategies (number of "bins").

        Returns
        -------
        int
            The number of possible integer states of the simplex, i.e.,
            the number of ways to assign `group_size` individuals to `nb_strategies` strategies.

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
        )pbdoc",
            py::arg("group_size"),
            py::arg("nb_strategies")
        );
#else
    m.def("calculate_nb_states",
          &egttools::starsBars<size_t>,
          R"pbdoc(Calculates the average frequency of each strategy available in
                the population given the stationary distribution.

                It expects that the stationary_distribution is in sparse form.

                Parameters
                ----------
                pop_size : int
                    Size of the population.
                nb_strategies : int
                    Number of strategies that can be assigned to players.
                stationary_distribution : scipy.sparse.csr_matrix
                    A sparse matrix which contains the stationary distribution (the frequency with which the evolutionary system visits each
                    stationary state).

                Returns
                -------
                numpy.ndarray[numpy.float64[m, 1]]
                    Average frequency of each strategy in the stationary evolutionary system.

                See Also
                --------
                egttools.numerical.calculate_state, egttools.numerical.sample_simplex,
                egttools.numerical.calculate_nb_states, egttools.numerical.PairwiseComparisonNumerical.estimate_stationary_distribution,
                egttools.numerical.calculate_nb_states, egttools.numerical.PairwiseComparisonNumerical.estimate_stationary_distribution_sparse
                )pbdoc",
          py::arg("group_size"), py::arg("nb_strategies"));
#endif


        m.def(
            "calculate_strategies_distribution",
            &utils::calculate_strategies_distribution,
            R"pbdoc(
        Calculates the average frequency of each strategy given a stationary distribution.

        This method computes the average strategy frequencies in the population
        based on the stationary distribution over all population states.
        It is assumed that the stationary distribution is sparse.

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
        NDArray[np.float64]
            A 1D NumPy array of shape (nb_strategies,) containing the average frequency
            of each strategy across all states in the stationary distribution.

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
        )pbdoc",
            py::arg("pop_size"),
            py::arg("nb_strategies"),
            py::arg("stationary_distribution"),
            py::return_value_policy::move
        );

        m.def(
            "replicator_equation",
            &infinite_populations::replicator_equation,
            R"pbdoc(
        Computes the replicator dynamics gradient for a 2-player game.

        This function implements the standard replicator equation for infinite populations
        interacting in pairwise games. The result is a vector of growth rates (gradients)
        for each strategy.

        Parameters
        ----------
        frequencies : NDArray[np.float64]
            A 1D NumPy array of shape (nb_strategies,) representing the current frequency
            of each strategy in the population. The entries should sum to 1.
        payoff_matrix : NDArray[np.float64]
            A 2D NumPy array of shape (nb_strategies, nb_strategies) containing payoffs.
            Entry [i, j] is the payoff for strategy i when interacting with j.

        Returns
        -------
        NDArray[np.float64]
            A 1D NumPy array of shape (nb_strategies,) representing the replicator gradient
            for each strategy.

        See Also
        --------
        egttools.replicator_equation_n_player
        egttools.games.AbstractGame
        egttools.numerical.PairwiseComparison
        egttools.analytical.StochDynamics

        Examples
        --------
        >>> freqs = np.array([0.4, 0.6])
        >>> A = np.array([[1, 0], [3, 2]])
        >>> grad = replicator_equation(freqs, A)
        >>> grad
        array([-0.24,  0.24])
        )pbdoc",
            py::arg("frequencies"),
            py::arg("payoff_matrix"),
            py::return_value_policy::move
        );

        m.def(
            "replicator_equation_n_player",
            &egttools::infinite_populations::replicator_equation_n_player,
            R"pbdoc(
        Computes the replicator dynamics gradient for N-player games.

        This function extends the replicator equation to games involving more than two players.
        The payoff for a strategy depends on the configuration of all other strategies in the group,
        encoded in the `payoff_matrix`.

        Parameters
        ----------
        frequencies : NDArray[np.float64]
            A 1D NumPy array of shape (nb_strategies,) representing the current frequency
            of each strategy in the population. The entries must sum to 1.
        payoff_matrix : NDArray[np.float64]
            A 2D NumPy array of shape (nb_strategies, nb_group_configurations). Each row
            corresponds to a strategy, and each column represents a group composition,
            indexed using the lexicographic order defined by `egttools.sample_simplex`.
        group_size : int
            The number of players interacting simultaneously.

        Returns
        -------
        NDArray[np.float64]
            A 1D NumPy array of shape (nb_strategies,) containing the replicator gradient
            for each strategy.

        See Also
        --------
        egttools.replicator_equation
        egttools.games.AbstractGame
        egttools.analytical.StochDynamics
        egttools.numerical.PairwiseComparison

        Examples
        --------
        >>> freqs = np.array([0.3, 0.5, 0.2])
        >>> group_size = 3
        >>> A = np.random.rand(3, 10)  # payoff matrix with nb_group_configurations columns
        >>> grad = replicator_equation_n_player(freqs, A, group_size)
        >>> grad.shape
        (3,)
        )pbdoc",
            py::arg("frequencies"),
            py::arg("payoff_matrix"),
            py::arg("group_size"),
            py::return_value_policy::move
        );

        m.def(
            "vectorized_replicator_equation_n_player",
            &egttools::infinite_populations::vectorized_replicator_equation_n_player,
            R"pbdoc(
        Vectorized computation of replicator dynamics for 3-strategy N-player games.

        This function computes replicator gradients over a meshgrid of frequency values
        for 3-strategy populations. It is optimized for performance using vectorization.

        The three input matrices `x1`, `x2`, `x3` must represent the frequencies of each
        strategy over a 2D grid. The sum of x1 + x2 + x3 must equal 1 elementwise.

        Parameters
        ----------
        x1 : NDArray[np.float64]
            2D array of the first strategy's frequencies.
        x2 : NDArray[np.float64]
            2D array of the second strategy's frequencies.
        x3 : NDArray[np.float64]
            2D array of the third strategy's frequencies.
        payoff_matrix : NDArray[np.float64]
            Array of shape (3, nb_group_configurations) with payoffs for each strategy.
        group_size : int
            Number of players in a group interaction.

        Returns
        -------
        Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]
            A tuple with three 2D arrays (same shape as input) representing the replicator
            gradient for each strategy at each grid point.

        See Also
        --------
        egttools.replicator_equation_n_player
        egttools.vectorized_replicator_equation

        Examples
        --------
        >>> X, Y = np.meshgrid(np.linspace(0, 1, 50), np.linspace(0, 1, 50))
        >>> x1 = X
        >>> x2 = Y
        >>> x3 = 1 - x1 - x2
        >>> A = np.random.rand(3, 10)
        >>> u1, u2, u3 = vectorized_replicator_equation_n_player(x1, x2, x3, A, 3)
        >>> u1.shape
        (50, 50)
        )pbdoc",
            py::arg("x1"),
            py::arg("x2"),
            py::arg("x3"),
            py::arg("payoff_matrix"),
            py::arg("group_size"),
            py::return_value_policy::move,
            py::call_guard<py::gil_scoped_release>()
        );

        py::class_<FinitePopulations::analytical::PairwiseComparison>(m, "PairwiseComparison")
                .def(
                    py::init<int, FinitePopulations::AbstractGame &>(),
                    R"pbdoc(
            Analytical pairwise comparison model.

            This class computes fixation probabilities, gradients of selection, and transition matrices
            for finite populations using the pairwise comparison rule. Results are exact, and rely
            on symbolic or deterministic calculation of fitness and payoffs.

            Parameters
            ----------
            population_size : int
                Size of the population.
            game : egttools.games.AbstractGame
                Game object that implements `AbstractGame`. Provides payoffs and fitness calculations.

            See Also
            --------
            egttools.numerical.PairwiseComparisonNumerical
            egttools.analytical.StochDynamics

            Examples
            --------
            >>> from egttools.games import Matrix2PlayerGameHolder
            >>> from egttools import PairwiseComparison
            >>> game = Matrix2PlayerGameHolder(3, np.random.rand(3, 3))
            >>> model = PairwiseComparison(100, game)
        )pbdoc",
                    py::arg("population_size"),
                    py::arg("game"),
                    py::keep_alive<0, 2>()
                )
                .def(
                    py::init<int, FinitePopulations::AbstractGame &, size_t>(),
                    R"pbdoc(
            Analytical pairwise comparison model with configurable cache.

            Extends the base constructor with a specified cache size, which accelerates repeated
            payoff/fitness evaluations in large simulations.

            Parameters
            ----------
            population_size : int
                Size of the population.
            game : egttools.games.AbstractGame
                Game object that implements `AbstractGame`. Provides payoffs and fitness calculations.
            cache_size : int
                Maximum number of evaluations to cache.

            See Also
            --------
            egttools.numerical.PairwiseComparisonNumerical
            egttools.analytical.StochDynamics

            Note
            ----
            Avoid using this model for large state spaces due to memory and performance limitations.
        )pbdoc",
                    py::arg("population_size"),
                    py::arg("game"),
                    py::arg("cache_size"),
                    py::keep_alive<0, 2>()
                )
                .def("pre_calculate_edge_fitnesses",
                     &egttools::FinitePopulations::analytical::PairwiseComparison::pre_calculate_edge_fitnesses,
                     R"pbdoc(
            Precompute fitnesses at the edges of the simplex.

            This optimization step helps accelerate calculations when simulating dynamics
            repeatedly over boundary conditions.
        )pbdoc")
                .def("calculate_transition_matrix",
                     &FinitePopulations::analytical::PairwiseComparison::calculate_transition_matrix,
                     py::arg("beta"),
                     py::arg("mu"),
                     py::return_value_policy::move,
                     R"pbdoc(
            Computes the full transition matrix under mutation.

            Parameters
            ----------
            beta : float
                Intensity of selection.
            mu : float
                Mutation probability.

            Returns
            -------
            scipy.sparse.csr_matrix
                Sparse matrix of shape (nb_states, nb_states) representing transition probabilities.
        )pbdoc")
                .def("calculate_gradient_of_selection",
                     &FinitePopulations::analytical::PairwiseComparison::calculate_gradient_of_selection,
                     py::arg("beta"),
                     py::arg("state"),
                     R"pbdoc(
            Computes the deterministic gradient of selection at a specific state.

            Parameters
            ----------
            beta : float
                Intensity of selection.
            state : NDArray[np.int64]
                Population state as a count vector of shape (nb_strategies,).

            Returns
            -------
            NDArray[np.float64]
                Gradient vector of shape (nb_strategies,).

            Example
            -------
            >>> model.calculate_gradient_of_selection(beta=0.5, state=np.array([50, 25, 25]))
        )pbdoc")
                .def("calculate_fixation_probability",
                     &FinitePopulations::analytical::PairwiseComparison::calculate_fixation_probability,
                     py::arg("invading_strategy_index"),
                     py::arg("resident_strategy_index"),
                     py::arg("beta"),
                     R"pbdoc(
            Computes the fixation probability of one mutant in a monomorphic population.

            Parameters
            ----------
            invading_strategy_index : int
                Index of the mutant strategy.
            resident_strategy_index : int
                Index of the resident strategy.
            beta : float
                Intensity of selection.

            Returns
            -------
            float
                Fixation probability of the mutant strategy.

            Example
            -------
            >>> model.calculate_fixation_probability(1, 0, 0.1)
        )pbdoc")
                .def("calculate_transition_and_fixation_matrix_sml",
                     &FinitePopulations::analytical::PairwiseComparison::calculate_transition_and_fixation_matrix_sml,
                     py::arg("beta"),
                     py::return_value_policy::move,
                     py::call_guard<py::gil_scoped_release>(),
                     R"pbdoc(
            Returns transition and fixation matrices assuming small mutation limit (SML).

            By assuming the limit of small mutations (SML), we can reduce the number of states of the dynamical system
            to those which are monomorphic, i.e., the whole population adopts the same strategy.

            Thus, the dimensions of the transition matrix in the SML is (nb_strategies, nb_strategies), and
            the transitions are given by the normalized fixation probabilities. This means that a transition
            where i \neq j, T[i, j] = fixation(i, j) / (nb_strategies - 1) and T[i, i] = 1 - \sum{T[i, j]}.

            This method will also return the matrix of fixation probabilities,
            where fixation_probabilities[i, j] gives the probability that one mutant j fixates in a population
            of i.

            Parameters
            ----------
            beta : float
                Selection strength.

            Returns
            -------
            Tuple[NDArray[np.float64], NDArray[np.float64]]
                Tuple with transition matrix and fixation probabilities matrix.
        )pbdoc")
                .def("update_population_size",
                     &egttools::FinitePopulations::analytical::PairwiseComparison::update_population_size)
                .def("nb_strategies", &egttools::FinitePopulations::analytical::PairwiseComparison::nb_strategies)
                .def("nb_states", &egttools::FinitePopulations::analytical::PairwiseComparison::nb_states)
                .def("population_size", &egttools::FinitePopulations::analytical::PairwiseComparison::population_size)
                .def("game", &egttools::FinitePopulations::analytical::PairwiseComparison::game);

        options.enable_function_signatures();
    } {
        py::options options;
        options.disable_function_signatures();

        auto pair_comp = py::class_<PairwiseComparison>(m, "PairwiseComparisonNumerical",
                                                        R"pbdoc(
        Numerical solver for evolutionary dynamics under the Pairwise Comparison rule.

        This class provides efficient simulation-based methods to estimate the fixation probabilities,
        stationary distributions, and evolutionary trajectories in finite populations.

        See Also
        --------
        egttools.analytical.PairwiseComparison
        egttools.analytical.StochDynamics
        egttools.games.AbstractGame
        )pbdoc")
                .def(py::init<size_t, FinitePopulations::AbstractGame &, size_t>(),
                     py::arg("pop_size"), py::arg("game"), py::arg("cache_size"), py::keep_alive<0, 2>(),
                     R"pbdoc(
                    Construct a numerical solver for a finite population game.

                    This class defines methods to estimate numerically fixation probabilities, stationary distributions with or without
                    mutation, and strategy distributions.

                    Parameters
                    ----------
                    pop_size : int
                        The number of individuals in the population.
                    game : AbstractGame
                        A game object implementing the payoff and fitness structure.
                    cache_size : int
                        The maximum size of the cache to store fitness computations.

                    Example
                    -------
                    >>> game = egttools.games.Matrix2PlayerGameHolder(3, payoff_matrix)
                    >>> pc = egttools.PairwiseComparisonNumerical(100, game, 10000)

                    Notes
                    -----
                    Numerical computations are not exact. Moreover, for now we still did not implement a method to automatically
                    detect if the precision of the estimation of the stationary and strategy distributions are good enough and,
                    thus, stop the simulation. You are advised to test different nb_generations and transitory periods for your
                    specific problem (game).

                    If you want to have exact calculations, you can use egttools.analytical.PairwiseComparison. However, this
                    is only advisable for systems with a smaller number of states (i.e., not too big population size or number of strategies).
                    Otherwise, the calculations might require too much memory.
         )pbdoc")
                .def("evolve",
                     static_cast<VectorXui (PairwiseComparison::*)(
                         size_t, double, double, const Eigen::Ref<const VectorXui> &
                     )>(&PairwiseComparison::evolve),
                     py::arg("nb_generations"), py::arg("beta"), py::arg("mu"), py::arg("init_state"),
                     py::return_value_policy::move,
                     R"pbdoc(
         Simulate the pairwise comparison process with mutation.

         Parameters
         ----------
         nb_generations : int
             Number of generations to simulate.
         beta : float
             Intensity of selection.
         mu : float
             Mutation rate.
         init_state : NDArray[np.int64]
             Initial state vector of shape (n_strategies,) with counts per strategy.

         Returns
         -------
         NDArray[np.int64]
             Final population state as counts of each strategy.

         Example
         -------
         >>> pc.evolve(5000, 1.0, 0.01, np.array([99, 1, 0]))
         )pbdoc")
                .def("estimate_fixation_probability",
                     &PairwiseComparison::estimate_fixation_probability,
                     py::arg("index_invading_strategy"), py::arg("index_resident_strategy"),
                     py::arg("nb_runs"), py::arg("nb_generations"), py::arg("beta"),
                     py::call_guard<py::gil_scoped_release>(),
                     R"pbdoc(
         Estimate fixation probability of an invading strategy in a resident population.

        This method estimates the fixation probability of one mutant of the invading strategy
        in a population where all other individuals adopt the resident strategy.
        The parameter `nb_runs` is very important, since simulations
        are stopped once a monomorphic state is reached (all individuals adopt the same
        strategy). The more runs you specify, the better the estimation. You should consider
        specifying at least a 1000 runs.

         Parameters
         ----------
         index_invading_strategy : int
         index_resident_strategy : int
         nb_runs : int
             Number of independent simulations.
         nb_generations : int
         beta : float

         Returns
         -------
         float

         Example
         -------
         >>> pc.estimate_fixation_probability(0, 1, 1000, 5000, 1.0)
         )pbdoc")
                .def("estimate_stationary_distribution",
                     &PairwiseComparison::estimate_stationary_distribution,
                     py::arg("nb_runs"), py::arg("nb_generations"), py::arg("transitory"),
                     py::arg("beta"), py::arg("mu"),
                     py::call_guard<py::gil_scoped_release>(),
                     R"pbdoc(
         Estimate the full stationary distribution of states in sparse format.

        This method directly estimates how frequent each strategy is in the population, without calculating
        the stationary distribution as an intermediary step. You should use this method when the number
        of states of the system is bigger than `MAX_LONG_INT`, since it would not be possible to index the states
        in this case, and estimate_stationary_distribution and estimate_stationary_distribution_sparse would run into an
        overflow error.

         Parameters
         ----------
         nb_runs : int
            Number of independent simulations to perform. The final result will be an average over all the runs.
         nb_generations : int
         transitory : int
             Burn-in generations to discard.
         beta : float
         mu : float

         Returns
         -------
         NDArray[np.float64]
            The average frequency of each strategy in the population stored in a sparse array.

         Example
         -------
         >>> pc.estimate_stationary_distribution(100, 10000, 1000, 1.0, 0.01)
         )pbdoc")
                .def("estimate_stationary_distribution_sparse",
                     &PairwiseComparison::estimate_stationary_distribution_sparse,
                     py::arg("nb_runs"), py::arg("nb_generations"), py::arg("transitory"),
                     py::arg("beta"), py::arg("mu"),
                     py::call_guard<py::gil_scoped_release>(),
                     R"pbdoc(
        Sparse estimation of the stationary distribution. Optimized for large sparse state spaces.

        Same as `estimate_stationary_distribution`, but faster and more memory efficient.

        Parameters
        ----------
        nb_runs : int
            Number of independent simulations to perform. The final result will be an average over all the runs.
        nb_generations : int
        transitory : int
            Burn-in generations to discard.
        beta : float
        mu : float

        Returns
        -------
        scipy.sparse.csr_matrix
         )pbdoc")
                .def("estimate_strategy_distribution",
                     &PairwiseComparison::estimate_strategy_distribution,
                     py::arg("nb_runs"), py::arg("nb_generations"), py::arg("transitory"),
                     py::arg("beta"), py::arg("mu"),
                     py::call_guard<py::gil_scoped_release>(),
                     R"pbdoc(
        Estimate average frequency of each strategy over time.

        This method directly estimates how frequent each strategy is in the population, without calculating
        the stationary distribution as an intermediary step. You should use this method when the number
        of states of the system is bigger than MAX_LONG_INT, since it would not be possible to index the states
        in this case, and estimate_stationary_distribution and estimate_stationary_distribution_sparse would run into an
        overflow error.

        Parameters
        ----------
        nb_runs : int
            Number of independent simulations to perform. The final result will be an average over all the runs.
        nb_generations : int
        transitory : int
            Burn-in generations to discard.
        beta : float
        mu : float

        Returns
        -------
        NDArray[np.float64]

        Example
        -------
        >>> pc.estimate_strategy_distribution(100, 10000, 1000, 1.0, 0.01)
         )pbdoc")
                .def_property_readonly("nb_strategies", &PairwiseComparison::nb_strategies,
                                       "Number of strategies in the population.")

                .def_property_readonly("payoffs", &PairwiseComparison::payoffs,
                                       "Payoff matrix used for selection dynamics.")

                .def_property_readonly("nb_states", &PairwiseComparison::nb_states,
                                       "Number of discrete states in the population.")

                .def_property("pop_size",
                              &PairwiseComparison::population_size,
                              &PairwiseComparison::set_population_size,
                              "Current population size.")

                .def_property("cache_size",
                              &PairwiseComparison::cache_size,
                              &PairwiseComparison::set_cache_size,
                              "Maximum number of cached fitness values.");


        pair_comp.def("run_without_mutation",
                      static_cast<egttools::MatrixXui2D (PairwiseComparison::*)(
                          int64_t, double, const Eigen::Ref<const egttools::VectorXui> &
                      )>(&PairwiseComparison::run),
                      py::arg("nb_generations"),
                      py::arg("beta"),
                      py::arg("init_state"),
                      py::return_value_policy::move,
                      R"pbdoc(
    Simulates the stochastic dynamics without mutation.

    This function returns all the intermediate states of the population for each generation,
    starting from `init_state`. No mutation occurs; the process stops when fixation is reached
    or all generations are simulated.

    Parameters
    ----------
    nb_generations : int
        Number of generations to simulate.
    beta : float
        Intensity of selection.
    init_state : NDArray[np.int64]
        Initial population state (counts of each strategy).

    Returns
    -------
    NDArray[np.int64]
        Matrix of shape (nb_generations + 1, nb_strategies) containing all population states.

    Example
    -------
    >>> pc.run_without_mutation(1000, 1.0, np.array([99, 1, 0]))
    )pbdoc");

        pair_comp.def("run_without_mutation",
                      static_cast<MatrixXui2D (PairwiseComparison::*)(
                          int64_t, int64_t, double, const Eigen::Ref<const VectorXui> &
                      )>(&PairwiseComparison::run),
                      py::arg("nb_generations"),
                      py::arg("transient"),
                      py::arg("beta"),
                      py::arg("init_state"),
                      py::return_value_policy::move,
                      R"pbdoc(
    Simulates the stochastic dynamics without mutation, skipping transient states.

    This overload skips the first `transient` generations in the output.

    Parameters
    ----------
    nb_generations : int
        Total number of generations to simulate.
    transient : int
        Burn-in period; these generations are excluded from the return.
    beta : float
        Intensity of selection.
    init_state : NDArray[np.int64]
        Initial population state (counts of each strategy).

    Returns
    -------
    NDArray[np.int64]
        Matrix of shape (nb_generations - transient, nb_strategies).

    Example
    -------
    >>> pc.run_without_mutation(1000, 200, 1.0, np.array([50, 50, 0]))
    )pbdoc");

        pair_comp.def("run_with_mutation",
                      static_cast<MatrixXui2D (PairwiseComparison::*)(
                          int64_t, double, double, const Eigen::Ref<const VectorXui> &
                      )>(&PairwiseComparison::run),
                      py::arg("nb_generations"),
                      py::arg("beta"),
                      py::arg("mu"),
                      py::arg("init_state"),
                      py::return_value_policy::move,
                      R"pbdoc(
    Simulates stochastic dynamics with mutation for the specified number of generations.

    All intermediate states are returned, starting from the initial condition.

    Parameters
    ----------
    nb_generations : int
        Number of generations to simulate.
    beta : float
        Intensity of selection.
    mu : float
        Mutation rate.
    init_state : NDArray[np.int64]
        Initial state of the population.

    Returns
    -------
    NDArray[np.int64]
        Matrix of shape (nb_generations + 1, nb_strategies) with population states.

    Example
    -------
    >>> pc.run_with_mutation(5000, 1.0, 0.01, np.array([33, 33, 34]))
    )pbdoc");

        pair_comp.def("run_with_mutation",
                      static_cast<MatrixXui2D (PairwiseComparison::*)(
                          int64_t, int64_t, double, double, const Eigen::Ref<const VectorXui> &
                      )>(&PairwiseComparison::run),
                      py::arg("nb_generations"),
                      py::arg("transient"),
                      py::arg("beta"),
                      py::arg("mu"),
                      py::arg("init_state"),
                      py::return_value_policy::move,
                      R"pbdoc(
    Simulates stochastic dynamics with mutation, skipping transient states.

    Parameters
    ----------
    nb_generations : int
        Total number of generations.
    transient : int
        Number of initial generations to discard from the result.
    beta : float
        Intensity of selection.
    mu : float
        Mutation rate.
    init_state : NDArray[np.int64]
        Initial state of the population.

    Returns
    -------
    NDArray[np.int64]
        Matrix of shape (nb_generations - transient, nb_strategies) with population states.

    Example
    -------
    >>> pc.run_with_mutation(5000, 1000, 1.0, 0.01, np.array([33, 33, 34]))
    )pbdoc");


        pair_comp.def("run", [](pybind11::object &self, py::args args) -> void {
            PyErr_WarnEx(PyExc_DeprecationWarning, "DEPRECATED. Use run_without_mutation or run_with_mutation instead.",
                         1);
        });

        options.enable_function_signatures();
    } {
        py::options options;
        options.disable_function_signatures();

        py::class_<FinitePopulations::evolvers::GeneralPopulationEvolver>(m, "GeneralPopulationEvolver")
                .def(py::init<FinitePopulations::structure::AbstractStructure &>(),
                     py::arg("structure"), py::keep_alive<1, 2>(),
                     R"pbdoc(
            Evolves a general population structure.

            This class simulates evolutionary dynamics based on a user-defined structure
            (e.g., spatial, group, or network-based interaction).

            Parameters
            ----------
            structure : egttools.numerical.structure.AbstractStructure
                The structure that defines how individuals interact and update their strategies.

            See Also
            --------
            egttools.numerical.structure.AbstractStructure
            egttools.numerical.PairwiseComparisonNumerical

            Example
            -------
            >>> from egttools.numerical.structure import SomeConcreteStructure
            >>> struct = SomeConcreteStructure(...)
            >>> evolver = GeneralPopulationEvolver(struct)
         )pbdoc")

                .def("evolve",
                     &FinitePopulations::evolvers::GeneralPopulationEvolver::evolve,
                     py::call_guard<py::gil_scoped_release>(),
                     py::arg("nb_generations"),
                     py::return_value_policy::move,
                     R"pbdoc(
            Evolves the population and returns the final state.

            Runs the simulation for a fixed number of generations and returns
            the final counts of each strategy in the population.

            Parameters
            ----------
            nb_generations : int
                Number of generations to simulate.

            Returns
            -------
            NDArray[np.int64]
                Final counts of each strategy in the population.

            Example
            -------
            >>> final = evolver.evolve(1000)
         )pbdoc")

                .def("run",
                     &egttools::FinitePopulations::evolvers::GeneralPopulationEvolver::run,
                     py::call_guard<py::gil_scoped_release>(),
                     py::arg("nb_generations"), py::arg("transitory"),
                     py::return_value_policy::move,
                     R"pbdoc(
            Simulates the population and returns the final state after discarding transitory steps.

            This method evolves the population for `nb_generations` generations but returns the
            final state after discarding the first `transitory` generations.

            Parameters
            ----------
            nb_generations : int
                Total number of generations to simulate.
            transitory : int
                Number of initial generations to discard (burn-in period).

            Returns
            -------
            NDArray[np.int64]
                Final counts of each strategy after the transitory phase.

            Example
            -------
            >>> final = evolver.run(2000, 500)
         )pbdoc")

                .def("structure",
                     &FinitePopulations::evolvers::GeneralPopulationEvolver::structure,
                     R"pbdoc(
            Returns the structure used by the evolver.

            Returns
            -------
            egttools.numerical.structure.AbstractStructure
                The structure defining interaction and update rules.

            Example
            -------
            >>> structure = evolver.structure()
         )pbdoc");


        py::class_<FinitePopulations::evolvers::NetworkEvolver>(m, "NetworkEvolver")

                .def_static("evolve",
                            static_cast<VectorXui (*)(
                                int64_t,
                                FinitePopulations::structure::AbstractNetworkStructure &
                            )>(&FinitePopulations::evolvers::NetworkEvolver::evolve),
                            py::arg("nb_generations"),
                            py::arg("network"),
                            py::return_value_policy::move,
                            R"pbdoc(
            Evolves the network population and returns the final state.

            This simulates the dynamics over the given network structure and returns
            the final strategy counts after a number of generations.

            Parameters
            ----------
            nb_generations : int
                Number of generations to simulate.
            network : egttools.numerical.structure.AbstractNetworkStructure
                The network structure describing the population and its interactions.

            Returns
            -------
            NDArray[np.int64]
                Final strategy counts after evolution.

            Example
            -------
            >>> final = NetworkEvolver.evolve(1000, my_network)
        )pbdoc")

                .def_static("evolve",
                            static_cast<VectorXui (*)(
                                int64_t,
                                VectorXui &,
                                FinitePopulations::structure::AbstractNetworkStructure &
                            )>(&FinitePopulations::evolvers::NetworkEvolver::evolve),
                            py::arg("nb_generations"),
                            py::arg("initial_state"),
                            py::arg("network"),
                            py::return_value_policy::move,
                            R"pbdoc(
            Evolves the network population from a given initial state.

            Parameters
            ----------
            nb_generations : int
                Number of generations to simulate.
            initial_state : NDArray[np.int64]
                Initial counts of each strategy in the population.
            network : egttools.numerical.structure.AbstractNetworkStructure
                The network structure containing the population.

            Returns
            -------
            NDArray[np.int64]
                Final strategy counts after evolution.

            Example
            -------
            >>> final = NetworkEvolver.evolve(1000, initial_state, my_network)
        )pbdoc")

                .def_static("run",
                            static_cast<MatrixXui2D (*)(
                                int64_t,
                                int64_t,
                                FinitePopulations::structure::AbstractNetworkStructure &
                            )>(&FinitePopulations::evolvers::NetworkEvolver::run),
                            py::arg("nb_generations"),
                            py::arg("transitory"),
                            py::arg("network"),
                            py::return_value_policy::move,
                            R"pbdoc(
            Simulates the full trajectory of the population states.

            This method runs the simulation and returns all population states
            after the transitory period.

            Parameters
            ----------
            nb_generations : int
                Total number of generations to simulate.
            transitory : int
                Number of generations to discard before returning results.
            network : egttools.numerical.structure.AbstractNetworkStructure
                The network structure containing the population.

            Returns
            -------
            NDArray[np.int64]
                A matrix of shape (nb_generations - transitory, nb_strategies) representing
                the strategy counts over time.

            Example
            -------
            >>> trace = NetworkEvolver.run(1000, 100, my_network)
        )pbdoc")

                .def_static("run",
                            static_cast<MatrixXui2D (*)(
                                int64_t,
                                int64_t,
                                VectorXui &,
                                FinitePopulations::structure::AbstractNetworkStructure &
                            )>(&FinitePopulations::evolvers::NetworkEvolver::run),
                            py::arg("nb_generations"),
                            py::arg("transitory"),
                            py::arg("initial_state"),
                            py::arg("network"),
                            py::return_value_policy::move,
                            R"pbdoc(
            Runs the simulation from a custom initial state.

            Parameters
            ----------
            nb_generations : int
                Total number of generations.
            transitory : int
                Number of initial generations to discard.
            initial_state : NDArray[np.int64]
                Initial counts of strategies in the population.
            network : egttools.numerical.structure.AbstractNetworkStructure
                A network structure containing the population.

            Returns
            -------
            NDArray[np.int64]
                Trajectory of population states after the transitory phase.

            Example
            -------
            >>> trace = NetworkEvolver.run(1000, 100, init_state, my_network)
        )pbdoc")

                .def_static("estimate_time_dependent_average_gradients_of_selection",
                            static_cast<Matrix2D (*)(
                                std::vector<VectorXui> &,
                                int64_t,
                                int64_t,
                                int64_t,
                                FinitePopulations::structure::AbstractNetworkStructure &
                            )>(&
                                FinitePopulations::evolvers::NetworkEvolver::estimate_time_dependent_average_gradients_of_selection),
                            py::arg("states"),
                            py::arg("nb_simulations"),
                            py::arg("generation_start"),
                            py::arg("generation_stop"),
                            py::arg("network"),
                            py::return_value_policy::move,
                            R"pbdoc(
            Estimates the time-dependent gradient of selection for the specified states.

            This method simulates evolution starting from each state and calculates
            the average gradient of selection between `generation_start` and `generation_stop`.

            Parameters
            ----------
            states : List[NDArray[np.int64]]
                List of population states (strategy counts) to evaluate.
            nb_simulations : int
                Number of simulations per state.
            generation_start : int
                First generation to include in averaging.
            generation_stop : int
                Last generation to include in averaging.
            network : egttools.numerical.structure.AbstractNetworkStructure
                A network structure for population evolution.

            Returns
            -------
            NDArray[np.float64]
                Averaged gradient matrix for all given states.

            Example
            -------
            >>> avg_grad = NetworkEvolver.estimate_time_dependent_average_gradients_of_selection(
            ...     states, 50, 100, 200, my_network)
        )pbdoc")

                .def_static("estimate_time_dependent_average_gradients_of_selection",
                            static_cast<Matrix2D (*)(
                                std::vector<VectorXui> &,
                                int64_t,
                                int64_t,
                                int64_t,
                                std::vector<FinitePopulations::structure::AbstractNetworkStructure *>)>(&
                                FinitePopulations::evolvers::NetworkEvolver::estimate_time_dependent_average_gradients_of_selection),
                            py::arg("states"),
                            py::arg("nb_simulations"),
                            py::arg("generation_start"),
                            py::arg("generation_stop"),
                            py::arg("networks"),
                            py::call_guard<py::gil_scoped_release>(),
                            py::return_value_policy::move,
                            R"pbdoc(
            Same as the single-network version but averages over multiple networks.

            Parameters
            ----------
            states : List[NDArray[np.int64]]
                Initial population states.
            nb_simulations : int
                Number of simulations per state.
            generation_start : int
                First generation to consider.
            generation_stop : int
                Last generation to consider.
            networks : List[egttools.numerical.structure.AbstractNetworkStructure]
                Multiple network structures for averaging.

            Returns
            -------
            NDArray[np.float64]
                Averaged gradients for each initial state.

            Example
            -------
            >>> avg_grad = NetworkEvolver.estimate_time_dependent_average_gradients_of_selection(
            ...     states, 100, 50, 100, [net1, net2])
        )pbdoc")

                .def_static("estimate_time_independent_average_gradients_of_selection",
                            static_cast<Matrix2D (*)(
                                std::vector<VectorXui> &,
                                int64_t,
                                int64_t,
                                FinitePopulations::structure::AbstractNetworkStructure &
                            )>(&
                                FinitePopulations::evolvers::NetworkEvolver::estimate_time_independent_average_gradients_of_selection),
                            py::arg("states"),
                            py::arg("nb_simulations"),
                            py::arg("nb_generations"),
                            py::arg("network"),
                            py::return_value_policy::move,
                            R"pbdoc(
            Estimates time-independent gradients of selection for given states.

            Parameters
            ----------
            states : List[NDArray[np.int64]]
                Initial states to evaluate.
            nb_simulations : int
                Number of simulations per state.
            nb_generations : int
                Total generations to evolve per simulation.
            network : AbstractNetworkStructure
                Network describing the evolutionary interactions.

            Returns
            -------
            NDArray[np.float64]
                A matrix with one row per initial state and one column per strategy.

            Example
            -------
            >>> gradients = NetworkEvolver.estimate_time_independent_average_gradients_of_selection(
            ...     states, 100, 200, my_network)
        )pbdoc")

                .def_static("estimate_time_independent_average_gradients_of_selection",
                            static_cast<Matrix2D (*)(
                                std::vector<VectorXui> &,
                                int64_t,
                                int64_t,
                                std::vector<FinitePopulations::structure::AbstractNetworkStructure *>)>(&
                                FinitePopulations::evolvers::NetworkEvolver::estimate_time_independent_average_gradients_of_selection),
                            py::arg("states"),
                            py::arg("nb_simulations"),
                            py::arg("nb_generations"),
                            py::arg("networks"),
                            py::call_guard<py::gil_scoped_release>(),
                            py::return_value_policy::move,
                            R"pbdoc(
            Estimates time-independent gradients of selection across multiple networks.

            Parameters
            ----------
            states : List[NDArray[np.int64]]
                Initial population states.
            nb_simulations : int
                Number of simulations per state.
            nb_generations : int
                Number of generations per simulation.
            networks : List[AbstractNetworkStructure]
                List of network structures for averaging.

            Returns
            -------
            NDArray[np.float64]
                Matrix of averaged gradients, one row per state.

            Example
            -------
            >>> gradients = NetworkEvolver.estimate_time_independent_average_gradients_of_selection(
            ...     states, 50, 100, [net1, net2, net3])
        )pbdoc");


        options.enable_function_signatures();
    }
}

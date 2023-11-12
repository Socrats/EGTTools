/** Copyright (c) 2022-2023  Elias Fernandez
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
using PairwiseComparison = egttools::FinitePopulations::PairwiseMoran<egttools::Utils::LRUCache<std::string, double>>;

namespace egttools {
    egttools::VectorXli sample_simplex_directly(int64_t nb_strategies, int64_t pop_size) {
        std::mt19937_64 generator(egttools::Random::SeedGenerator::getInstance().getSeed());
        egttools::VectorXli state = egttools::VectorXli::Zero(nb_strategies);

        egttools::FinitePopulations::sample_simplex_direct_method<long int, long int, egttools::VectorXli, std::mt19937_64>(nb_strategies, pop_size, state, generator);

        return state;
    }
    egttools::Vector sample_unit_simplex(int64_t nb_strategies) {
        std::mt19937_64 generator(egttools::Random::SeedGenerator::getInstance().getSeed());
        auto real_rand = std::uniform_real_distribution<double>(0, 1);
        egttools::Vector state = egttools::Vector::Zero(nb_strategies);
        egttools::FinitePopulations::sample_unit_simplex<int64_t, std::mt19937_64>(nb_strategies, state, real_rand, generator);

        return state;
    }
}// namespace egttools

void init_methods(py::module_ &m) {

    // Use this function to get access to the singleton
    py::class_<Random::SeedGenerator, std::unique_ptr<Random::SeedGenerator, py::nodelete>>(m, "Random", "Random seed generator.")
            .def_static(
                    "init", []() {
                        return std::unique_ptr<Random::SeedGenerator, py::nodelete>(&Random::SeedGenerator::getInstance());
                    },
                    R"pbdoc(
                            This static method initializes the random seed.

                            This static method initializes the random seed generator from random_device
                            and returns an instance of egttools.Random which is used
                            to seed the random generators used across egttools.

                            Returns
                            -------
                            egttools.Random
                                An instance of the random seed generator.
           )pbdoc")
            .def_static(
                    "init", [](unsigned long int seed) {
                        auto instance = std::unique_ptr<Random::SeedGenerator, py::nodelete>(&Random::SeedGenerator::getInstance());
                        instance->setMainSeed(seed);
                        return instance;
                    },
                    R"pbdoc(
                            This static method initializes the random seed generator from seed.

                            This static method initializes the random seed generator from seed
                            and returns an instance of `egttools.Random` which is used
                            to seed the random generators used across `egttools`.

                            Parameters
                            ----------
                            seed : int
                                Integer value used to seed the random generator.

                            Returns
                            -------
                            egttools.Random
                                An instance of the random seed generator.
                    )pbdoc",
                    py::arg("seed"))
            .def_property_readonly_static(
                    "_seed", [](const py::object &) {
                        return egttools::Random::SeedGenerator::getInstance().getMainSeed();
                    },
                    "The initial seed of `egttools.Random`.")
            .def_static(
                    "generate", []() {
                        return egttools::Random::SeedGenerator::getInstance().getSeed();
                    },
                    R"pbdoc(
                    Generates a random seed.

                    The generated seed can be used to seed other pseudo-random generators,
                    so that the initial state of the simulation can always be tracked and
                    the simulation can be reproduced. This is very important both for debugging
                    purposes as well as for scientific research. However, this approach should
                    NOT be used in any cryptographic applications, it is NOT safe.

                    Returns
                    -------
                    int
                        A random seed which can be used to seed new random generators.
                    )pbdoc")
            .def_static(
                    "seed", [](unsigned long int seed) {
                        egttools::Random::SeedGenerator::getInstance().setMainSeed(seed);
                    },
                    R"pbdoc(
                    This static methods changes the seed of `egttools.Random`.

                    Parameters
                    ----------
                    int
                        The new seed for the `egttools.Random` module which is used to seed
                        every other pseudo-random generation in the `egttools` package.
                    )pbdoc",
                    py::arg("seed"));

    {
        py::options options;
        options.disable_function_signatures();

        m.def("calculate_state",
              static_cast<size_t (*)(const size_t &, const egttools::Factors &)>(&egttools::FinitePopulations::calculate_state),
              R"pbdoc(
            This function converts a vector containing counts into an index.

            This method was copied from @Svalorzen.

            Parameters
            ----------
            group_size : int
                Maximum bin size (it can also be the population size).
            group_composition : List[int]
                The vector to convert from simplex coordinates to index.

            Returns
            -------
            int
                The unique index in [0, egttools.calculate_nb_states(group_size, len(group_composition))
                representing the n-dimensional simplex.

            See Also
            --------
            egttools.sample_simplex, egttools.calculate_nb_states
          )pbdoc",
              py::arg("group_size"), py::arg("group_composition"));
        m.def("calculate_state",
              static_cast<size_t (*)(const size_t &,
                                     const Eigen::Ref<const egttools::VectorXui> &)>(&egttools::FinitePopulations::calculate_state),
              R"pbdoc(
            This function converts a vector containing counts into an index.

            This method was copied from @Svalorzen.

            Parameters
            ----------
            group_size : int
                Maximum bin size (it can also be the population size).
            group_composition : numpy.ndarray[numpy.int64[m, 1]]
                The vector to convert from simplex coordinates to index.

            Returns
            -------
            int
                The unique index in [0, egttools.calculate_nb_states(group_size, len(group_composition))
                representing the n-dimensional simplex.

            See Also
            --------
            egttools.sample_simplex, egttools.calculate_nb_states
                    )pbdoc",
              py::arg("group_size"), py::arg("group_composition"));

        options.enable_function_signatures();
    }

    m.def("sample_simplex",
          static_cast<egttools::VectorXui (*)(size_t, const size_t &, const size_t &)>(&egttools::FinitePopulations::sample_simplex),
          R"pbdoc(
            Transforms a state index into a vector.

            Parameters
            ----------
            index : int
                State index.
            pop_size : int
                Size of the population.
            nb_strategies : int
                Number of strategies.

            Returns
            -------
            numpy.ndarray[numpy.int64[m, 1]]
                Vector with the sampled state.

            See Also
            --------
            egttools.numerical.calculate_state, egttools.numerical.calculate_nb_states
                    )pbdoc",
          py::arg("index"), py::arg("pop_size"),
          py::arg("nb_strategies"));
    m.def("sample_simplex_directly",
          &sample_simplex_directly,
          R"pbdoc(
                    Samples an N-dimensional point directly from the simplex.

                    N is the number of strategies.

                    Parameters
                    ----------
                    nb_strategies : int
                        Number of strategies.
                    pop_size : int
                        Size of the population.

                    Returns
                    -------
                    numpy.ndarray[numpy.int64[m, 1]]
                        Vector with the sampled state.

                    See Also
                    --------
                    egttools.numerical.calculate_state, egttools.numerical.calculate_nb_states, egttools.numerical.sample_simplex
                    )pbdoc",
          py::arg("nb_strategies"),
          py::arg("pop_size"));
    m.def("sample_unit_simplex",
          &sample_unit_simplex,
          R"pbdoc(
                    Samples uniformly at random the unit simplex with nb_strategies dimensionse.

                    Parameters
                    ----------
                    nb_strategies : int
                        Number of strategies.

                    Returns
                    -------
                    numpy.ndarray[numpy.int64[m, 1]]
                        Vector with the sampled state.

                    See Also
                    --------
                    egttools.numerical.calculate_state, egttools.numerical.calculate_nb_states, egttools.numerical.sample_simplex
                    )pbdoc",
          py::arg("nb_strategies"));

#if (HAS_BOOST)
    m.def(
            "calculate_nb_states", [](size_t group_size, size_t nb_strategies) {
                auto result = starsBars<boost::multiprecision::cpp_int, boost::multiprecision::cpp_int>(group_size, nb_strategies);
                return py::cast(result);
            },
            R"pbdoc(
                    Calculates the number of states (combinations) of the members of a group in a subgroup.

                    It can be used to calculate the maximum number of states in a discrete simplex.

                    The implementation of this method follows the stars and bars algorithm (see Wikipedia).

                    Parameters
                    ----------
                    group_size : int
                        Size of the group (maximum number of players/elements that can adopt each possible strategy).
                    nb_strategies : int
                        number of strategies that can be assigned to players.

                    Returns
                    -------
                    int
                        Number of states (possible combinations of strategies and players).

                    See Also
                    --------
                    egttools.numerical.calculate_state, egttools.numerical.sample_simplex
                    )pbdoc",
            py::arg("group_size"), py::arg("nb_strategies"));
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

    m.def("calculate_strategies_distribution",
          static_cast<egttools::Vector (*)(size_t, size_t, egttools::SparseMatrix2D &)>(&egttools::utils::calculate_strategies_distribution),
          R"pbdoc(
                Calculates the average frequency of each strategy available in the population given the stationary distribution.

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
          py::arg("pop_size"), py::arg("nb_strategies"), py::arg("stationary_distribution"));

    m.def("replicator_equation", &egttools::infinite_populations::replicator_equation,
          py::arg("frequencies"), py::arg("payoff_matrix"),
          py::return_value_policy::move,
          R"pbdoc(
                    Calculates the gradient of the replicator dynamics given the current population state.

                    Parameters
                    ----------
                    frequencies : numpy.ndarray
                        Vector of frequencies of each strategy in the population (it must have
                        shape=(nb_strategies,)
                    payoff_matrix : numpy.ndarray
                        Square matrix containing the payoff of each row strategy against each column strategy

                    Returns
                    -------
                    numpy.ndarray
                        A vector with the gradient for each strategy. The vector has shape (nb_strategies,)

                    See Also
                    --------
                    egttools.analytical.replicator_equation_n_player
                    egttools.numerical.PairwiseComparison
                    egttools.numerical.PairwiseComparisonNumerical
                    egttools.analytical.StochDynamics
                    egttools.games.AbstractGame
                )pbdoc");

    m.def("replicator_equation_n_player", &egttools::infinite_populations::replicator_equation_n_player,
          py::arg("frequencies"), py::arg("payoff_matrix"), py::arg("group_size"),
          py::return_value_policy::move,
          R"pbdoc(
                    Calculates the gradient of the replicator dynamics given the current population state.

                    Parameters
                    ----------
                    frequencies : numpy.ndarray
                        Vector of frequencies of each strategy in the population (it must have
                        shape=(nb_strategies,)
                    payoff_matrix : numpy.ndarray
                        A payoff matrix containing the payoff of each row strategy for each
                        possible group configuration, indicated by the column index.
                        The matrix must have shape (nb_strategies, nb_group_configurations).
                    group_size : int
                        size of the group

                    Returns
                    -------
                    numpy.ndarray
                        A vector with the gradient for each strategy. The vector has shape (nb_strategies,)

                    See Also
                    --------
                    egttools.analytical.replicator_equation
                    egttools.numerical.PairwiseComparison
                    egttools.numerical.PairwiseComparisonNumerical
                    egttools.analytical.StochDynamics
                    egttools.games.AbstractGame
                )pbdoc");

    m.def("vectorized_replicator_equation_n_player", &egttools::infinite_populations::vectorized_replicator_equation_n_player,
          py::arg("x1"), py::arg("x2"), py::arg("x3"), py::arg("payoff_matrix"), py::arg("group_size"),
          py::return_value_policy::move, py::call_guard<py::gil_scoped_release>(),
          R"pbdoc(
                    Calculates the gradient of the replicator dynamics given the current population state.

                    This function must only be used for 3 strategy populations! It provides a fast way
                    to compute the gradient of selection for a large number of population states.

                    You need to pass 3 matrices each containing the frequency of one strategy.

                    The combination of [x1[i,j], x2[i,j], x3[i,j]], gives the population state.

                    Parameters
                    ----------
                    x1 : numpy.ndarray
                        Matrix containing the first component of the frequencies
                    x2 : numpy.ndarray
                        Matrix containing the second component of the frequencies
                    x3 : numpy.ndarray
                        Matrix containing the third component of the frequencies
                    payoff_matrix : numpy.ndarray
                        A payoff matrix containing the payoff of each row strategy for each
                        possible group configuration, indicated by the column index.
                        The matrix must have shape (nb_strategies, nb_group_configurations).
                    group_size : int
                        size of the group

                    Returns
                    -------
                    Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]
                        Returns 3 matrices containing the gradient of each strategy. Each Matrix
                        has the same shape as x1, x2 and x3.

                    See Also
                    --------
                    egttools.analytical.replicator_equation
                    egttools.numerical.PairwiseComparison
                    egttools.numerical.PairwiseComparisonNumerical
                    egttools.analytical.StochDynamics
                    egttools.games.AbstractGame
                )pbdoc");

    py::class_<egttools::FinitePopulations::analytical::PairwiseComparison>(m, "PairwiseComparison")
            .def(py::init<int, egttools::FinitePopulations::AbstractGame &>(),
                 R"pbdoc(
                    A class containing methods to study analytically the evolutionary dynamics using the Pairwise comparison rule.

                    This class defines methods to compute fixation probabilities, transition matrices in the Small Mutation
                    Limit (SML), gradients of selection, and the full transition matrices of the system when considering
                    mutation > 0.

                    Parameters
                    ----------
                    population_size : int
                        Size of the population.
                    game : egttools.games.AbstractGame
                        A game object which must implement the abstract class `egttools.games.AbstractGame`.
                        This game will contain the expected payoffs for each strategy in the game, or at least
                        a method to compute it, and a method to calculate the fitness of each strategy for a given
                        population state.

                    See Also
                    --------
                    egttools.numerical.PairwiseComparisonNumerical,
                    egttools.analytical.StochDynamics,
                    egttools.games.AbstractGame

                    Note
                    -----
                    Analytical computations should be avoided for problems with very large state spaces.
                    This means very big populations with many strategies. The bigger the state space, the
                    more memory and time these methods will require!

                    Also, for now it is not possible to update the game without having to instantiate PairwiseComparison
                    again. Hopefully, this will be fixed in the future.
                )pbdoc",
                 py::arg("population_size"), py::arg("game"), py::keep_alive<1, 2>())
            .def(py::init<int, egttools::FinitePopulations::AbstractGame &, size_t>(),
                 R"pbdoc(
                    A class containing methods to study analytically the evolutionary dynamics using the Pairwise comparison rule.

                    This class defines methods to compute fixation probabilities, transition matrices in the Small Mutation
                    Limit (SML), gradients of selection, and the full transition matrices of the system when considering
                    mutation > 0.

                    Parameters
                    ----------
                    population_size : int
                        Size of the population.
                    game : egttools.games.AbstractGame
                        A game object which must implement the abstract class `egttools.games.AbstractGame`.
                        This game will contain the expected payoffs for each strategy in the game, or at least
                        a method to compute it, and a method to calculate the fitness of each strategy for a given
                        population state.
                    cache_size : in
                        The size of the Cache.

                    See Also
                    --------
                    egttools.numerical.PairwiseComparisonNumerical,
                    egttools.analytical.StochDynamics,
                    egttools.games.AbstractGame

                    Note
                    -----
                    Analytical computations should be avoided for problems with very large state spaces.
                    This means very big populations with many strategies. The bigger the state space, the
                    more memory and time these methods will require!

                    Also, for now it is not possible to update the game without having to instantiate PairwiseComparison
                    again. Hopefully, this will be fixed in the future.
                )pbdoc",
                 py::arg("population_size"), py::arg("game"), py::arg("cache_size"), py::keep_alive<1, 2>())
            .def("pre_calculate_edge_fitnesses", &egttools::FinitePopulations::analytical::PairwiseComparison::pre_calculate_edge_fitnesses,
                 "pre calculates the payoffs of the edges of the simplex.")
            .def("calculate_transition_matrix",
                 &egttools::FinitePopulations::analytical::PairwiseComparison::calculate_transition_matrix,
                 R"pbdoc(
                    Computes the transition matrix of the Markov Chain which defines the population dynamics.

                    It is not advisable to use this method for very large state spaces since the memory required
                    to store the matrix might explode. In these cases you should resort to dimensional reduction
                    techniques, such as the Small Mutation Limit (SML).

                    Parameters
                    ----------
                    beta : float
                        Intensity of selection
                    mu : float
                        Mutation rate

                    Returns
                    -------
                    scipy.sparse.csr_matrix
                        Sparse vector containing the transition probabilities from any population state to another.
                        This matrix will be of shape nb_states x nb_states.

                    See Also
                    --------
                    egttools.analytical.StochDynamics,
                    egttools.analytical.StochDynamics.calculate_full_transition_matrix,
                    egttools.analytical.PairwiseComparison.calculate_transition_and_fixation_matrix_sml,
                    egttools.numerical.PairwiseComparisonNumerical,
                    egttools.numerical.PairwiseComparisonNumerical.estimate_stationary_distribution,
                    egttools.numerical.PairwiseComparisonNumerical.estimate_stationary_distribution_sparse
                )pbdoc",
                 py::arg("beta"), py::arg("mu"), py::return_value_policy::move)
            .def("calculate_gradient_of_selection", &egttools::FinitePopulations::analytical::PairwiseComparison::calculate_gradient_of_selection,
                 R"pbdoc(
                    Calculates the gradient of selection without mutation for the given state.

                    This method calculates the gradient of selection (without mutation), which is, the
                    most likely direction of evolution of the system.

                    Parameters
                    ----------
                    beta : float
                        Intensity of selection
                    state : numpy.ndarray
                        Vector containing the counts of each strategy in the population.

                    Returns
                    -------
                    numpy.ndarray
                        Vector of shape (nb_strategies,) containing the gradient of selection, i.e.,
                        The most likely path of evolution of the stochastic system.

                    See Also
                    --------
                    egttools.analytical.StochDynamics,
                    egttools.analytical.StochDynamics.full_gradient_selection,
                    egttools.analytical.PairwiseComparison.calculate_transition_matrix,
                    egttools.analytical.PairwiseComparison.calculate_transition_and_fixation_matrix_sml,
                    egttools.numerical.PairwiseComparisonNumerical,
                    egttools.numerical.PairwiseComparisonNumerical.estimate_stationary_distribution,
                    egttools.numerical.PairwiseComparisonNumerical.estimate_stationary_distribution_sparse
                )pbdoc",
                 py::arg("beta"), py::arg("state"))
            .def("calculate_fixation_probability", &egttools::FinitePopulations::analytical::PairwiseComparison::calculate_fixation_probability,
                 R"pbdoc(
                    Calculates the fixation probability of an invading strategy in a population o resident strategy.

                    This method calculates the fixation probability of one mutant of the invading strategy
                    in a population where all other individuals adopt the resident strategy.

                    Parameters
                    ----------
                    index_invading_strategy: int
                        Index of the invading strategy
                    index_resident_strategy: int
                        Index of the resident strategy
                    beta : float
                        Intensity of selection

                    Returns
                    -------
                    float
                        The fixation probability of one mutant of the invading strategy in a population
                        where all other members adopt the resident strategy.

                    See Also
                    --------
                    egttools.analytical.StochDynamics,
                    egttools.analytical.StochDynamics.fixation_probability,
                    egttools.analytical.PairwiseComparison.calculate_transition_matrix,
                    egttools.analytical.PairwiseComparison.calculate_transition_and_fixation_matrix_sml,
                    egttools.analytical.PairwiseComparison.calculate_gradient_of_selection,
                    egttools.numerical.PairwiseComparisonNumerical,
                    egttools.numerical.PairwiseComparisonNumerical.estimate_stationary_distribution,
                    egttools.numerical.PairwiseComparisonNumerical.estimate_stationary_distribution_sparse,
                    egttools.numerical.PairwiseComparisonNumerical.estimate_fixation_probability
                )pbdoc",
                 py::arg("invading_strategy_index"), py::arg("resident_strategy_index"), py::arg("beta"))
            .def("calculate_transition_and_fixation_matrix_sml", &egttools::FinitePopulations::analytical::PairwiseComparison::calculate_transition_and_fixation_matrix_sml,
                 py::call_guard<py::gil_scoped_release>(),
                 R"pbdoc(
                    Calculates the transition matrix of the reduced Markov Chain that emerges when assuming SML.

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
                        Intensity of selection

                    Returns
                    -------
                    Tuple[numpy.ndarray, numpy.ndarray]
                        A tuple including the transition matrix and a matrix with the fixation probabilities.
                        Both matrices have shape (nb_strategies, nb_strategies).

                    See Also
                    --------
                    egttools.analytical.StochDynamics,
                    egttools.analytical.StochDynamics.fixation_probability,
                    egttools.analytical.StochDynamics.transition_and_fixation_matrix,
                    egttools.analytical.PairwiseComparison.calculate_fixation_probability
                    egttools.analytical.PairwiseComparison.calculate_transition_matrix,
                    egttools.analytical.PairwiseComparison.calculate_transition_and_fixation_matrix_sml,
                    egttools.analytical.PairwiseComparison.calculate_gradient_of_selection,
                    egttools.numerical.PairwiseComparisonNumerical,
                    egttools.numerical.PairwiseComparisonNumerical.estimate_stationary_distribution,
                    egttools.numerical.PairwiseComparisonNumerical.estimate_stationary_distribution_sparse,
                    egttools.numerical.PairwiseComparisonNumerical.estimate_fixation_probability
                )pbdoc",
                 py::arg("beta"), py::return_value_policy::move)
            .def("update_population_size", &egttools::FinitePopulations::analytical::PairwiseComparison::update_population_size)
            .def("nb_strategies", &egttools::FinitePopulations::analytical::PairwiseComparison::nb_strategies)
            .def("nb_states", &egttools::FinitePopulations::analytical::PairwiseComparison::nb_states)
            .def("population_size", &egttools::FinitePopulations::analytical::PairwiseComparison::population_size)
            .def("game", &egttools::FinitePopulations::analytical::PairwiseComparison::game);

    {
        auto pair_comp = py::class_<PairwiseComparison>(m, "PairwiseComparisonNumerical")
                                 .def(py::init<size_t, egttools::FinitePopulations::AbstractGame &, size_t>(),
                                      R"pbdoc(
                    A class containing methods to study numerically the evolutionary dynamics using the Pairwise comparison rule.

                    This class defines methods to estimate numerically fixation probabilities, stationary distributions with or without
                    mutation, and strategy distributions.

                    Parameters
                    ----------
                    population_size : int
                        Size of the population.
                    game : egttools.games.AbstractGame
                        A game object which must implement the abstract class `egttools.games.AbstractGame`.
                        This game will contain the expected payoffs for each strategy in the game, or at least
                        a method to compute it, and a method to calculate the fitness of each strategy for a given
                        population state.
                    cache_size : int
                        The maximum size of the cache.

                    See Also
                    --------
                    egttools.analytical.PairwiseComparison,
                    egttools.analytical.StochDynamics,
                    egttools.games.AbstractGame

                    Note
                    -----
                    Numerical computations are not exact. Moreover, for now we still did not implement a method to automatically
                    detect if the precision of the estimation of the stationary and strategy distributions are good enough and,
                    thus, stop the simulation. You are advised to test different nb_generations and transitory periods for your
                    specific problem (game).

                    If you want to have exact calculations, you can use egttools.analytical.PairwiseComparison. However, this
                    is only advisable for systems with a smaller number of states (i.e., not too big population size or number of strategies).
                    Otherwise, the calculations might require too much memory.
                )pbdoc",
                                      py::arg("pop_size"), py::arg("game"), py::arg("cache_size"), py::keep_alive<1, 3>())
                                 .def("evolve",
                                      static_cast<egttools::VectorXui (PairwiseComparison::*)(size_t, double, double,
                                                                                              const Eigen::Ref<const egttools::VectorXui> &)>(&PairwiseComparison::evolve),
                                      R"pbdoc(
                    Runs the moran process for a given number of generations.

                    Parameters
                    ----------
                    nb_generations : int
                        Maximum number of generations.
                    beta : float
                        Intensity of selection.
                    mu: float
                        Mutation rate.
                    init_state: numpy.ndarray
                        Initial state of the population. This must be a vector of integers of shape (nb_strategies,),
                        containing the counts of each strategy in the population. It serves as the initial state
                        from which the evolutionary process will start.

                    Returns
                    -------
                    numpy.ndarray
                        A vector of integers containing the final state reached during the evolutionary process.

                    See Also
                    --------
                    egttools.numerical.PairwiseComparisonNumerical.run,
                    egttools.numerical.PairwiseComparisonNumerical.estimate_stationary_distribution_sparse,
                    egttools.numerical.PairwiseComparisonNumerical.estimate_strategy_distribution
                )pbdoc",
                                      py::arg("nb_generations"), py::arg("beta"),
                                      py::arg("mu"), py::arg("init_state"), py::return_value_policy::move)
                                 .def("estimate_fixation_probability",
                                      &PairwiseComparison::estimate_fixation_probability,
                                      py::call_guard<py::gil_scoped_release>(),
                                      R"pbdoc(
                        Estimates the fixation probability of an invading strategy in a population o resident strategy.

                        This method estimates the fixation probability of one mutant of the invading strategy
                        in a population where all other individuals adopt the resident strategy.

                        The :param nb_runs is very important, since simulations
                        are stopped once a monomorphic state is reached (all individuals adopt the same
                        strategy). The more runs you specify, the better the estimation. You should consider
                        specifying at least a 1000 runs.

                        Parameters
                        ----------
                        index_invading_strategy : int
                            Index of the invading strategy.
                        index_resident_strategy : int
                            Index of the resident strategy.
                        nb_runs : int
                            Number of independent runs. This parameter is very important, since simulations
                            are stopped once a monomorphic state is reached (all individuals adopt the same
                            strategy). The more runs you specify, the better the estimation. You should consider
                            specifying at least a 1000 runs.
                        nb_generations : int
                            Maximum number of generations for a single run.
                        beta: float
                            Intensity of selection.

                        Returns
                        -------
                        numpy.ndarray
                            A matrix containing all the states the system when through, including also the initial state.
                            The shape of the matrix is (nb_generations - transient, nb_strategies).

                        See Also
                        --------
                        egttools.numerical.PairwiseComparisonNumerical.evolve,
                        egttools.numerical.PairwiseComparisonNumerical.run,
                        egttools.numerical.PairwiseComparisonNumerical.estimate_stationary_distribution_sparse,
                        egttools.numerical.PairwiseComparisonNumerical.estimate_strategy_distribution,
                        egttools.analytical.StochDynamics,
                        egttools.analytical.StochDynamics.fixation_probability,
                        egttools.analytical.StochDynamics.transition_and_fixation_matrix,
                        egttools.analytical.PairwiseComparison.calculate_fixation_probability
                        egttools.analytical.PairwiseComparison.calculate_transition_matrix,
                        egttools.analytical.PairwiseComparison.calculate_transition_and_fixation_matrix_sml,
                )pbdoc",
                                      py::arg("index_invading_strategy"), py::arg("index_resident_strategy"), py::arg("nb_runs"), py::arg("nb_generations"), py::arg("beta"))
                                 .def("estimate_stationary_distribution", &PairwiseComparison::estimate_stationary_distribution,
                                      py::call_guard<py::gil_scoped_release>(),
                                      R"pbdoc(
                        Estimates the stationary distribution of the population of strategies given the game.

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
                            Total number of generations.
                        transitory: int
                            Transitory period. These generations will be excluded from the final average. Thus, only the last
                            nb_generations - transitory generations will be taken into account. This is important, since in
                            order to obtain a correct average at the steady state, we need to skip the transitory period.
                        beta: float
                            Intensity of selection. This parameter determines how important the difference in payoff between players
                            is for the probability of imitation. If beta is small, the system will mostly undergo random drift
                            between strategies. If beta is high, a slight difference in payoff will make a strategy disapear.
                        mu: float
                            Probability of mutation. This parameter defines how likely it is for a mutation event to occur at a given generation

                        Returns
                        -------
                        scipy.sparse.csr_matrix
                            The average frequency of each strategy in the population stored in a sparse array.

                        See Also
                        --------
                        egttools.numerical.PairwiseComparisonNumerical.estimate_stationary_distribution_sparse,
                        egttools.numerical.PairwiseComparisonNumerical.estimate_strategy_distribution,
                        egttools.analytical.StochDynamics,
                        egttools.analytical.StochDynamics.fixation_probability,
                        egttools.analytical.StochDynamics.transition_and_fixation_matrix,
                        egttools.analytical.PairwiseComparison.calculate_transition_matrix,
                        egttools.analytical.PairwiseComparison.calculate_transition_and_fixation_matrix_sml,
                        egttools.analytical.PairwiseComparison.calculate_gradient_of_selection
                )pbdoc",
                                      py::arg("nb_runs"), py::arg("nb_generations"), py::arg("transitory"), py::arg("beta"), py::arg("mu"))
                                 .def("estimate_stationary_distribution_sparse", &PairwiseComparison::estimate_stationary_distribution_sparse,
                                      py::call_guard<py::gil_scoped_release>(),
                                      R"pbdoc(
                        Estimates the stationary distribution of the population of strategies given the game.

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
                            Total number of generations.
                        transitory: int
                            Transitory period. These generations will be excluded from the final average. Thus, only the last
                            nb_generations - transitory generations will be taken into account. This is important, since in
                            order to obtain a correct average at the steady state, we need to skip the transitory period.
                        beta: float
                            Intensity of selection. This parameter determines how important the difference in payoff between players
                            is for the probability of imitation. If beta is small, the system will mostly undergo random drift
                            between strategies. If beta is high, a slight difference in payoff will make a strategy disapear.
                        mu: float
                            Probability of mutation. This parameter defines how likely it is for a mutation event to
                            occur at a given generation

                        Returns
                        -------
                        scipy.sparse.csr_matrix
                        The average frequency of each strategy in the population stored in a sparse array.

                        See Also
                        --------
                        egttools.numerical.PairwiseComparisonNumerical.estimate_stationary_distribution,
                        egttools.numerical.PairwiseComparisonNumerical.estimate_strategy_distribution,
                        egttools.analytical.StochDynamics,
                        egttools.analytical.StochDynamics.fixation_probability,
                        egttools.analytical.StochDynamics.transition_and_fixation_matrix,
                        egttools.analytical.PairwiseComparison.calculate_transition_matrix,
                        egttools.analytical.PairwiseComparison.calculate_transition_and_fixation_matrix_sml,
                        egttools.analytical.PairwiseComparison.calculate_gradient_of_selection
                )pbdoc",
                                      py::arg("nb_runs"), py::arg("nb_generations"), py::arg("transitory"), py::arg("beta"), py::arg("mu"))
                                 .def("estimate_strategy_distribution", &PairwiseComparison::estimate_strategy_distribution,
                                      py::call_guard<py::gil_scoped_release>(),
                                      R"pbdoc(
                        Estimates the distribution of strategies in the population given the current game.

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
                            Total number of generations.
                        transitory: int
                            Transitory period. These generations will be excluded from the final average. Thus, only the last
                            nb_generations - transitory generations will be taken into account. This is important, since in
                            order to obtain a correct average at the steady state, we need to skip the transitory period.
                        beta: float
                            Intensity of selection. This parameter determines how important the difference in payoff between players
                            is for the probability of imitation. If beta is small, the system will mostly undergo random drift
                            between strategies. If beta is high, a slight difference in payoff will make a strategy disapear.
                        mu: float
                            Probability of mutation. This parameter defines how likely it is for a mutation event to occur at a given generation

                        Returns
                        -------
                        numpy.ndarray[numpy.float64[m, 1]]
                            The average frequency of each strategy in the population.

                        See Also
                        --------
                        egttools.numerical.PairwiseComparisonNumerical.estimate_stationary_distribution,
                        egttools.numerical.PairwiseComparisonNumerical.estimate_stationary_distribution_sparse,
                        egttools.analytical.StochDynamics,
                        egttools.analytical.StochDynamics.fixation_probability,
                        egttools.analytical.StochDynamics.transition_and_fixation_matrix,
                        egttools.analytical.PairwiseComparison.calculate_transition_matrix,
                        egttools.analytical.PairwiseComparison.calculate_transition_and_fixation_matrix_sml,
                        egttools.analytical.PairwiseComparison.calculate_gradient_of_selection
                )pbdoc",
                                      py::arg("nb_runs"), py::arg("nb_generations"), py::arg("transitory"), py::arg("beta"), py::arg("mu"))
                                 .def_property_readonly("nb_strategies", &PairwiseComparison::nb_strategies, "Number of strategies in the population.")
                                 .def_property_readonly("payoffs", &PairwiseComparison::payoffs,
                                                        "Payoff matrix containing the payoff of each strategy (row) for each game state (column)")
                                 .def_property_readonly("nb_states", &PairwiseComparison::nb_states, "number of possible population states")
                                 .def_property("pop_size", &PairwiseComparison::population_size, &PairwiseComparison::set_population_size,
                                               "Size of the population.")
                                 .def_property("cache_size", &PairwiseComparison::cache_size, &PairwiseComparison::set_cache_size,
                                               "Maximum memory which can be used to cache the fitness calculations.");

        py::options options;
        options.disable_function_signatures();

        pair_comp.def("run",
                      static_cast<egttools::MatrixXui2D (PairwiseComparison::*)(int_fast64_t, double,
                                                                                const Eigen::Ref<const egttools::VectorXui> &)>(&PairwiseComparison::run),
                      R"pbdoc(
                    Runs the evolutionary process and returns a matrix with all the states the system went through.

                    Parameters
                    ----------
                    nb_generations : int
                        Maximum number of generations.
                    beta : float
                        Intensity of selection.
                    init_state: numpy.ndarray
                        Initial state of the population. This must be a vector of integers of shape (nb_strategies,),
                        containing the counts of each strategy in the population. It serves as the initial state
                        from which the evolutionary process will start.

                    Returns
                    -------
                    numpy.ndarray
                        A matrix containing all the states the system when through, including also the initial state.
                        The shape of the matrix is (nb_generations + 1, nb_strategies).

                    See Also
                    --------
                    egttools.numerical.PairwiseComparisonNumerical.evolve,
                    egttools.numerical.PairwiseComparisonNumerical.estimate_stationary_distribution_sparse,
                    egttools.numerical.PairwiseComparisonNumerical.estimate_strategy_distribution
                )pbdoc",
                      py::arg("nb_generations"),
                      py::arg("beta"),
                      py::arg("init_state"), py::return_value_policy::move);
        pair_comp.def("run",
                      static_cast<egttools::MatrixXui2D (PairwiseComparison::*)(int_fast64_t, int_fast64_t, double, double,
                                                                                const Eigen::Ref<const egttools::VectorXui> &)>(&PairwiseComparison::run),
                      R"pbdoc(
                    Runs the evolutionary process and returns a matrix with all the states the system went through.

                    Mutation events will happen with rate :param mu, and the transient states will not be returned.

                    Parameters
                    ----------
                    nb_generations : int
                        Maximum number of generations.
                    transient : int
                        Transient period. Amount of generations that should not be skipped in the return vector.
                    beta : float
                        Intensity of selection.
                    mu : float
                        Mutation rate.
                    init_state: numpy.ndarray
                        Initial state of the population. This must be a vector of integers of shape (nb_strategies,),
                        containing the counts of each strategy in the population. It serves as the initial state
                        from which the evolutionary process will start.

                    Returns
                    -------
                    numpy.ndarray
                        A matrix containing all the states the system when through, including also the initial state.
                        The shape of the matrix is (nb_generations - transient, nb_strategies).

                    See Also
                    --------
                    egttools.numerical.PairwiseComparisonNumerical.evolve,
                    egttools.numerical.PairwiseComparisonNumerical.estimate_stationary_distribution_sparse,
                    egttools.numerical.PairwiseComparisonNumerical.estimate_strategy_distribution
                )pbdoc",
                      py::arg("nb_generations"),
                      py::arg("transient"),
                      py::arg("beta"),
                      py::arg("mu"),
                      py::arg("init_state"), py::return_value_policy::move);
        pair_comp.def("run",
                      static_cast<egttools::MatrixXui2D (PairwiseComparison::*)(int_fast64_t, int_fast64_t, double,
                                                                                const Eigen::Ref<const egttools::VectorXui> &)>(&PairwiseComparison::run),
                      R"pbdoc(
                    Runs the evolutionary process and returns a matrix with all the states the system went through.

                    Mutation events will happen with rate :param mu, and the transient states will not be returned.

                    Parameters
                    ----------
                    nb_generations : int
                        Maximum number of generations.
                    transient : int
                        Transient period. Amount of generations that should not be skipped in the return vector.
                    beta : float
                        Intensity of selection.
                    mu : float
                        Mutation rate.
                    init_state: numpy.ndarray
                        Initial state of the population. This must be a vector of integers of shape (nb_strategies,),
                        containing the counts of each strategy in the population. It serves as the initial state
                        from which the evolutionary process will start.

                    Returns
                    -------
                    numpy.ndarray
                        A matrix containing all the states the system when through, including also the initial state.
                        The shape of the matrix is (nb_generations - transient, nb_strategies).

                    See Also
                    --------
                    egttools.numerical.PairwiseComparisonNumerical.evolve,
                    egttools.numerical.PairwiseComparisonNumerical.estimate_stationary_distribution_sparse,
                    egttools.numerical.PairwiseComparisonNumerical.estimate_strategy_distribution
                )pbdoc",
                      py::arg("nb_generations"),
                      py::arg("transient"),
                      py::arg("beta"),
                      py::arg("init_state"), py::return_value_policy::move);
        pair_comp.def("run",
                      static_cast<egttools::MatrixXui2D (PairwiseComparison::*)(int_fast64_t, double, double,
                                                                                const Eigen::Ref<const egttools::VectorXui> &)>(&PairwiseComparison::run),
                      R"pbdoc(
                    Runs the evolutionary process and returns a matrix with all the states the system went through.

                    Mutation events will happen with rate :param mu.

                    Parameters
                    ----------
                    nb_generations : int
                        Maximum number of generations.
                    beta : float
                        Intensity of selection.
                    mu : float
                        Mutation rate.
                    init_state: numpy.ndarray
                        Initial state of the population. This must be a vector of integers of shape (nb_strategies,),
                        containing the counts of each strategy in the population. It serves as the initial state
                        from which the evolutionary process will start.

                    Returns
                    -------
                    numpy.ndarray
                        A matrix containing all the states the system when through, including also the initial state.
                        The shape of the matrix is (nb_generations - transient, nb_strategies).

                    See Also
                    --------
                    egttools.numerical.PairwiseComparisonNumerical.evolve,
                    egttools.numerical.PairwiseComparisonNumerical.estimate_stationary_distribution_sparse,
                    egttools.numerical.PairwiseComparisonNumerical.estimate_strategy_distribution
                )pbdoc",
                      py::arg("nb_generations"),
                      py::arg("beta"),
                      py::arg("mu"),
                      py::arg("init_state"), py::return_value_policy::move);

        options.enable_function_signatures();
    }

    py::class_<egttools::FinitePopulations::evolvers::GeneralPopulationEvolver>(m, "GeneralPopulationEvolver")
            .def(py::init<egttools::FinitePopulations::structure::AbstractStructure &>(),
                 py::arg("structure"), py::keep_alive<1, 2>(),
                 R"pbdoc(
                    General population evolver.

                    This class is designed to simulation the evolution of a population defined inside
                    the `structure` object.

                    Parameters
                    ----------
                    structure : egttools.numerical.structure.AbstractStructure
                        A Structure object which defines the relations between individuals in the population
                        as well as how individuals update their behavior.

                    See Also
                    --------
                    egttools.numerical.PairwiseComparisonNumerical
                )pbdoc")
            .def("evolve", &egttools::FinitePopulations::evolvers::GeneralPopulationEvolver::evolve,
                 py::call_guard<py::gil_scoped_release>(),
                 py::arg("nb_generations"), py::return_value_policy::move,
                 R"pbdoc(
                    Evolves the population in structure for `nb_generations`.

                    This method only returns the last total counts of strategies in the population.

                    Parameters
                    ----------
                    nb_generations : int
                        Maximum number of generations.

                    Returns
                    -------
                    numpy.ndarray
                        An array with the final count of strategies in the population.

                    See Also
                    --------
                    egttools.numerical.PairwiseComparisonNumerical.evolve
                )pbdoc")
            .def("run", &egttools::FinitePopulations::evolvers::GeneralPopulationEvolver::run,
                 py::call_guard<py::gil_scoped_release>(),
                 py::arg("nb_generations"), py::arg("transitory"), py::return_value_policy::move,
                 R"pbdoc(
                    Evolves the population in structure for `nb_generations`.

                    This method only returns the last total counts of strategies in the population.

                    Parameters
                    ----------
                    nb_generations : int
                        Maximum number of generations.
                    transitory : int
                        The transitory period. The generations until transitory are not taken into account.

                    Returns
                    -------
                    numpy.ndarray
                        An array with the final count of strategies in the population.

                    See Also
                    --------
                    egttools.numerical.PairwiseComparisonNumerical.evolve
                )pbdoc")
            .def("structure", &egttools::FinitePopulations::evolvers::GeneralPopulationEvolver::structure);

    py::class_<egttools::FinitePopulations::evolvers::NetworkEvolver>(m, "NetworkEvolver")
            .def_static("evolve", static_cast<egttools::VectorXui (*)(int_fast64_t, egttools::FinitePopulations::structure::AbstractNetworkStructure &)>(&egttools::FinitePopulations::evolvers::NetworkEvolver::evolve),
                        py::arg("nb_generations"),
                        py::arg("network"),
                        py::return_value_policy::move,
                        R"pbdoc(
                    Evolves an `AbstractNetworkStructure` for `nb_generations`.

                    This method only returns the last total counts of strategies in the population.

                    Parameters
                    ----------
                    nb_generations : int
                        Maximum number of generations.
                    network : egttools.numerical.structure.AbstractNetworkStructure
                        A network structure containing a population to evolve

                    Returns
                    -------
                    numpy.ndarray
                        An array with the final count of strategies in the population.

                    See Also
                    --------
                    egttools.numerical.PairwiseComparisonNumerical.evolve
                )pbdoc")
            .def_static("evolve", static_cast<egttools::VectorXui (*)(int_fast64_t, egttools::VectorXui &, egttools::FinitePopulations::structure::AbstractNetworkStructure &)>(&egttools::FinitePopulations::evolvers::NetworkEvolver::evolve),
                        py::arg("nb_generations"),
                        py::arg("initial_state"),
                        py::arg("network"),
                        py::return_value_policy::move,
                        R"pbdoc(
                    Evolves an `AbstractNetworkStructure` for `nb_generations`.

                    This method only returns the last total counts of strategies in the population.

                    Parameters
                    ----------
                    nb_generations : int
                        Maximum number of generations.
                    initial_state : numpy.ndarray
                        The initial counts of each strategy in the populations
                    network : egttools.numerical.structure.AbstractNetworkStructure
                        A network structure containing a population to evolve

                    Returns
                    -------
                    numpy.ndarray
                        An array with the final count of strategies in the population.

                    See Also
                    --------
                    egttools.numerical.PairwiseComparisonNumerical.evolve
                )pbdoc")
            .def_static("run", static_cast<egttools::MatrixXui2D (*)(int_fast64_t, int_fast64_t, egttools::FinitePopulations::structure::AbstractNetworkStructure &)>(&egttools::FinitePopulations::evolvers::NetworkEvolver::run),
                        py::arg("nb_generations"),
                        py::arg("transitory"),
                        py::arg("network"),
                        py::return_value_policy::move,
                        R"pbdoc(
                    Simulates the evolution of an `AbstractNetworkStructure` for `nb_generations`.

                    This method returns all the states the population goes through between `transient` and
                    `nb_generations`.

                    Parameters
                    ----------
                    nb_generations : int
                        Maximum number of generations.
                    transitory : int
                        The transitory period. The generations until transitory are not taken into account.
                    network : egttools.numerical.structure.AbstractNetworkStructure
                        A network structure containing a population to evolve

                    Returns
                    -------
                    numpy.ndarray
                        An array with the final count of strategies in the population.

                    See Also
                    --------
                    egttools.numerical.PairwiseComparisonNumerical.evolve
                )pbdoc")
            .def_static("run", static_cast<egttools::MatrixXui2D (*)(int_fast64_t, int_fast64_t, egttools::VectorXui &, egttools::FinitePopulations::structure::AbstractNetworkStructure &)>(&egttools::FinitePopulations::evolvers::NetworkEvolver::run),
                        py::arg("nb_generations"),
                        py::arg("transitory"),
                        py::arg("initial_state"),
                        py::arg("network"),
                        py::return_value_policy::move,
                        R"pbdoc(
                    Simulates the evolution of an `AbstractNetworkStructure` for `nb_generations`.

                    This method returns all the states the population goes through between `transient` and
                    `nb_generations`.

                    Parameters
                    ----------
                    nb_generations : int
                        Maximum number of generations.
                    transitory : int
                        The transitory period. The generations until transitory are not taken into account.
                    initial_state : numpy.ndarray
                        The initial counts of each strategy in the population
                    network : egttools.numerical.structure.AbstractNetworkStructure
                        A network structure containing a population to evolve

                    Returns
                    -------
                    numpy.ndarray
                        An array with the final count of strategies in the population.

                    See Also
                    --------
                    egttools.numerical.PairwiseComparisonNumerical.evolve
                )pbdoc")
            .def_static("estimate_time_dependent_average_gradients_of_selection", static_cast<egttools::Matrix2D (*)(std::vector<VectorXui> &, int_fast64_t, int_fast64_t, int_fast64_t, egttools::FinitePopulations::structure::AbstractNetworkStructure &)>(&egttools::FinitePopulations::evolvers::NetworkEvolver::estimate_time_dependent_average_gradients_of_selection),
                        py::arg("states"), py::arg("nb_simulations"),
                        py::arg("generation_start"),
                        py::arg("generation_stop"),
                        py::arg("network"),
                        py::return_value_policy::move,
                        R"pbdoc(
                    Estimates the time-dependant gradient of selection

                    This method will first evolve the population for (generation - 1) generations. Afterwards,
                    it will average the gradient of selection observed for each state the population goes though in
                    the next generation. This means, that if the update is asynchronous, the population will be
                    evolved for population_size time-steps and the gradient will be computed for each time-step
                    and averaged over other simulations in which the population has gone through the same aggregated
                    state.

                    Note
                    ----
                    We recommend only using this method with asynchronous updates.

                    Parameters
                    ----------
                    initial_states : List[numpy.ndarray]
                        A list of population states for which to calculate the gradients.
                    nb_simulations : int
                        The number of simulations to perform for the given state
                    generation_start : int
                        the generation at which we start to calculate the average gradient of selection
                    generation_stop : int
                        the final generation of the simulation
                    network : egttools.numerical.structure.AbstractNetworkStructure
                        A network structure containing a population to evolve

                    Returns
                    -------
                    numpy.ndarray
                        A matrix with the final count of strategies in the population for each possible initial state.

                    See Also
                    --------
                    egttools.numerical.PairwiseComparisonNumerical.evolve
                )pbdoc")
            .def_static("estimate_time_dependent_average_gradients_of_selection", static_cast<egttools::Matrix2D (*)(std::vector<VectorXui> &, int_fast64_t, int_fast64_t, int_fast64_t, std::vector<egttools::FinitePopulations::structure::AbstractNetworkStructure *>)>(&egttools::FinitePopulations::evolvers::NetworkEvolver::estimate_time_dependent_average_gradients_of_selection),
                        py::arg("states"), py::arg("nb_simulations"),
                        py::arg("generation_start"),
                        py::arg("generation_stop"),
                        py::arg("networks"),
                        py::call_guard<py::gil_scoped_release>(),
                        py::return_value_policy::move,
                        R"pbdoc(
                    Estimates the time-dependant gradient of selection

                    This method will first evolve the population for (generation - 1) generations. Afterwards,
                    it will average the gradient of selection observed for each state the population goes though in
                    the next generation. This means, that if the update is asynchronous, the population will be
                    evolved for population_size time-steps and the gradient will be computed for each time-step
                    and averaged over other simulations in which the population has gone through the same aggregated
                    state.

                    Note
                    ----
                    We recommend only using this method with asynchronous updates.

                    Parameters
                    ----------
                    initial_states : List[numpy.ndarray]
                        A list of population states for which to calculate the gradients.
                    nb_simulations : int
                        The number of simulations to perform for the given state
                    generation_start : int
                        the generation at which we start to calculate the average gradient of selection
                    generation_stop : int
                        the final generation of the simulation
                    networks : List[egttools.numerical.structure.AbstractNetworkStructure]
                        A list of network structures containing a population to evolve

                    Returns
                    -------
                    numpy.ndarray
                        A matrix with the final count of strategies in the population for each possible initial state.

                    See Also
                    --------
                    egttools.numerical.PairwiseComparisonNumerical.evolve
                )pbdoc")
            .def_static("estimate_time_independent_average_gradients_of_selection", static_cast<egttools::Matrix2D (*)(std::vector<egttools::VectorXui> &states, int_fast64_t, int_fast64_t, egttools::FinitePopulations::structure::AbstractNetworkStructure &)>(&egttools::FinitePopulations::evolvers::NetworkEvolver::estimate_time_independent_average_gradients_of_selection),
                        py::arg("states"), py::arg("nb_simulations"),
                        py::arg("nb_generations"), py::arg("network"),
                        py::return_value_policy::move,
                        R"pbdoc(
                    Estimates the time independent average gradient of selection.

                    It is important here that the user takes into account that generations have a slightly different meaning if
                    the network updates are synchronous or asynchronous. In a synchronous case, in each generation, there is
                    a simultaneous update of every member of the population, thus, there a Z (population_size) steps.

                    In the asynchronous case, we will adopt the definition used in Pinheiro, Pacheco and Santos 2012,
                    and assume that 1 generation = Z time-steps (Z asynchronous updates of the population). Thus, a simulation
                    with 25 generations and with 1000 individuals, will run for 25000 time-steps.

                    This method will run a total of simulations * networks.size() simulations. The final gradients are averaged over
                    simulations * networks.size() * nb_generations * nb_initial_states.

                    Warning
                    -------
                    Don't use this method if the population has too many possible states, since it will likely take both a long time,
                    produce a bad estimation, and possible your computer will run out of memory.

                    Parameters
                    ----------
                    initial_states : List[numpy.ndarray]
                        A list of population states for which to calculate the gradients.
                    nb_simulations : int
                        The number of simulations to perform for the given state
                    nb_generations : int
                        Maximum number of generations.
                    network : AbstractNetworkStructure
                        A network structure.

                    Returns
                    -------
                    numpy.ndarray
                        A 2D numpy array containing the averaged gradients for each state given (each row is one gradient).

                    See Also
                    --------
                    egttools.numerical.PairwiseComparisonNumerical.evolve
                )pbdoc")
            .def_static("estimate_time_independent_average_gradients_of_selection", static_cast<egttools::Matrix2D (*)(std::vector<egttools::VectorXui> &states, int_fast64_t, int_fast64_t, std::vector<egttools::FinitePopulations::structure::AbstractNetworkStructure *>)>(&egttools::FinitePopulations::evolvers::NetworkEvolver::estimate_time_independent_average_gradients_of_selection),
                        py::arg("states"), py::arg("nb_simulations"),
                        py::arg("nb_generations"), py::arg("networks"),
                        py::return_value_policy::move,
                        py::call_guard<py::gil_scoped_release>(),
                        R"pbdoc(
                    Estimates the average gradient of selection averaging over multiple simulations, generations
                    and network for each state given state.

                    Parameters
                    ----------
                    states : List[numpy.ndarray]
                        A list of population states for which to calculate the gradients.
                    nb_simulations : int
                        The number of simulations to perform for the given state
                    nb_generations : int
                        Maximum number of generations.
                    network : List[AbstractNetworkStructure]
                        A list of network structures.

                    Returns
                    -------
                    numpy.ndarray
                        A 2D numpy array containing the averaged gradients for each state given (each row is one gradient).

                    See Also
                    --------
                    egttools.numerical.PairwiseComparisonNumerical.evolve
                )pbdoc");
}
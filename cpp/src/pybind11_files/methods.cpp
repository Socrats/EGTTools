/** Copyright (c) 2022-2026  Elias Fernandez
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
            static_cast<size_t (*)(const size_t &, const egttools::Factors &)>(
                &egttools::FinitePopulations::calculate_state
            ),
            R"pbdoc(
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
)pbdoc",
            py::arg("group_size"),
            py::arg("group_composition")
        );

        m.def(
            "calculate_state",
            static_cast<size_t (*)(const size_t &, const Eigen::Ref<const egttools::VectorXui> &)>(
                &egttools::FinitePopulations::calculate_state
            ),
            R"pbdoc(
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
)pbdoc",
            py::arg("group_size"),
            py::arg("group_composition")
        );

        m.def(
            "sample_simplex",
            static_cast<egttools::VectorXui (*)(size_t, const size_t &, const size_t &)>(
                &egttools::FinitePopulations::sample_simplex
            ),
            R"pbdoc(
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
)pbdoc",
            py::arg("nb_strategies"),
            py::arg("pop_size"),
            py::return_value_policy::move
        );

        m.def(
            "sample_unit_simplex",
            &sample_unit_simplex,
            R"pbdoc(
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
)pbdoc",
            py::arg("group_size"),
            py::arg("nb_strategies")
        );
#else
        m.def(
            "calculate_nb_states",
            &egttools::starsBars<size_t>,
            R"pbdoc(
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
)pbdoc",
            py::arg("group_size"),
            py::arg("nb_strategies")
        );
#endif

        m.def(
            "calculate_strategies_distribution",
            &utils::calculate_strategies_distribution,
            R"pbdoc(
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
)pbdoc",
            py::arg("pop_size"),
            py::arg("nb_strategies"),
            py::arg("stationary_distribution"),
            py::return_value_policy::move,
            // Pure C++: no Python callbacks, safe to release the GIL.
            py::call_guard<py::gil_scoped_release>()
        );

        m.def(
            "calculate_expected_payoff",
            &utils::calculate_expected_payoff,
            R"pbdoc(
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
)pbdoc",
            py::arg("pop_size"),
            py::arg("group_size"),
            py::arg("nb_strategies"),
            py::arg("stationary_distribution"),
            py::arg("payoff_matrix"),
            // Pure C++: no Python callbacks, GIL not needed; release it so other
            // threads can run and so OpenMP threads inside the template are safe.
            py::call_guard<py::gil_scoped_release>()
        );

        m.def(
            "calculate_expected_indicator",
            static_cast<double (*)(int64_t, int64_t, int64_t,
                                   egttools::SparseMatrix2D &,
                                   const std::function<double(const std::vector<size_t> &)> &)>(
                &utils::calculate_expected_indicator),
            R"pbdoc(
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
)pbdoc",
            py::arg("pop_size"),
            py::arg("group_size"),
            py::arg("nb_strategies"),
            py::arg("stationary_distribution"),
            py::arg("indicator")
            // No gil_scoped_release: indicator is a Python callable and must be
            // invoked with the GIL held.  The std::function overload deliberately
            // runs the serial loop to avoid calling Python from OpenMP threads.
        );

        m.def(
            "calculate_expected_indicators",
            static_cast<egttools::Vector (*)(
                int64_t, int64_t, int64_t,
                egttools::SparseMatrix2D &,
                const std::vector<std::function<double(const std::vector<size_t> &)>> &)>(
                &utils::calculate_expected_indicators),
            R"pbdoc(
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
)pbdoc",
            py::arg("pop_size"),
            py::arg("group_size"),
            py::arg("nb_strategies"),
            py::arg("stationary_distribution"),
            py::arg("indicators"),
            py::return_value_policy::move
            // No gil_scoped_release: Python callables are invoked in phase 1.
        );

        m.def(
            "calculate_expected_indicators_precomputed",
            &utils::calculate_expected_indicators_precomputed,
            R"pbdoc(
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
)pbdoc",
            py::arg("pop_size"),
            py::arg("group_size"),
            py::arg("nb_strategies"),
            py::arg("stationary_distribution"),
            py::arg("indicator_matrix"),
            py::return_value_policy::move,
            py::call_guard<py::gil_scoped_release>()
        );

        m.def(
            "calculate_expected_group_success",
            &utils::calculate_expected_group_success,
            R"pbdoc(
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
)pbdoc",
            py::arg("pop_size"),
            py::arg("group_size"),
            py::arg("nb_strategies"),
            py::arg("stationary_distribution"),
            py::arg("threshold"),
            py::arg("contributing_strategies"),
            // Pure C++: lambdas capture only C++ data, GIL not needed.
            py::call_guard<py::gil_scoped_release>()
        );

        m.def(
            "replicator_equation",
            py::overload_cast<
                const egttools::Vector &,
                const egttools::Matrix2D &>(&egttools::infinite_populations::replicator_equation),
            R"pbdoc(
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
)pbdoc",
            py::arg("frequencies"),
            py::arg("payoff_matrix"),
            py::return_value_policy::move
        );

        m.def(
            "replicator_equation",
            py::overload_cast<
                const egttools::Vector &,
                const egttools::infinite_populations::AbstractReplicatorGame &>(
                &egttools::infinite_populations::replicator_equation),
            R"pbdoc(
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
)pbdoc",
            py::arg("frequencies"),
            py::arg("game"),
            py::return_value_policy::move
        );

        m.def(
            "replicator_equation_n_player",
            py::overload_cast<
                const egttools::Vector &,
                const egttools::Matrix2D &,
                size_t
            >(&egttools::infinite_populations::replicator_equation_n_player),
            R"pbdoc(
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
)pbdoc",
            py::arg("frequencies"),
            py::arg("payoff_matrix"),
            py::arg("group_size"),
            py::return_value_policy::move
        );

        m.def(
            "replicator_equation_n_player",
            py::overload_cast<
                const egttools::Vector &,
                const egttools::infinite_populations::AbstractReplicatorGame &>(
                &egttools::infinite_populations::replicator_equation_n_player),
            R"pbdoc(
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
)pbdoc",
            py::arg("frequencies"),
            py::arg("game"),
            py::return_value_policy::move
        );

        m.def(
            "vectorized_replicator_equation",
            &egttools::infinite_populations::vectorized_replicator_equation,
            R"pbdoc(
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
)pbdoc",
            py::arg("x1"),
            py::arg("x2"),
            py::arg("x3"),
            py::arg("game"),
            py::return_value_policy::move,
            py::call_guard<py::gil_scoped_release>()
        );

        m.def(
            "vectorized_replicator_equation_n_player",
            py::overload_cast<
                const egttools::Matrix2D &,
                const egttools::Matrix2D &,
                const egttools::Matrix2D &,
                const egttools::Matrix2D &,
                size_t
            >(&egttools::infinite_populations::vectorized_replicator_equation_n_player),
            R"pbdoc(
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
)pbdoc",
            py::arg("x1"),
            py::arg("x2"),
            py::arg("x3"),
            py::arg("payoff_matrix"),
            py::arg("group_size"),
            py::return_value_policy::move,
            py::call_guard<py::gil_scoped_release>()
        );

        m.def(
            "vectorized_replicator_equation_n_player",
            py::overload_cast<
                const egttools::Matrix2D &,
                const egttools::Matrix2D &,
                const egttools::Matrix2D &,
                const egttools::infinite_populations::AbstractReplicatorGame &>(
                &egttools::infinite_populations::vectorized_replicator_equation_n_player),
            R"pbdoc(
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
)pbdoc",
            py::arg("x1"),
            py::arg("x2"),
            py::arg("x3"),
            py::arg("game"),
            py::return_value_policy::move,
            py::call_guard<py::gil_scoped_release>()
        );

        py::class_<FinitePopulations::analytical::PairwiseComparison>(
                    m,
                    "PairwiseComparison",
                    R"pbdoc(
Analytical pairwise-comparison process for finite populations.

This class studies evolutionary dynamics in a well-mixed population of fixed size
:math:`Z`, whose state is represented by a vector of strategy counts
:math:`x = (x_1, \dots, x_n)` satisfying :math:`\sum_{i=1}^n x_i = Z`.

Under the pairwise comparison rule, strategy updates are driven by pairwise imitation,
typically through the Fermi kernel

.. math::

    p_{i \to j}(x) =
    \frac{1}{1 + \exp[-\beta (f_j(x) - f_i(x))]},

where :math:`\beta \ge 0` is the intensity of selection and :math:`f_i(x)` is the
fitness of strategy :math:`i` in state :math:`x`.

The class provides tools to construct the full Markov transition matrix, compute
gradients of selection, fixation probabilities, and the reduced small-mutation-limit
(SML) dynamics.
)pbdoc"
                )
                .def(
                    py::init<int, FinitePopulations::AbstractGame &>(),
                    py::arg("population_size"),
                    py::arg("game"),
                    py::keep_alive<1, 3>(),
                    R"pbdoc(
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

    |\mathcal{S}| = \binom{Z + n - 1}{n - 1},

where :math:`n` is the number of strategies.
)pbdoc"
                )
                .def(
                    py::init<int, FinitePopulations::AbstractGame &, size_t>(),
                    py::arg("population_size"),
                    py::arg("game"),
                    py::arg("cache_size"),
                    py::keep_alive<1, 3>(),
                    R"pbdoc(
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
)pbdoc"
                )
                .def(
                    "pre_calculate_edge_fitnesses",
                    &egttools::FinitePopulations::analytical::PairwiseComparison::pre_calculate_edge_fitnesses,
                    R"pbdoc(
Precompute fitness values along all edges of the simplex.

This is particularly useful for repeated pairwise fixation calculations, since
fixation probabilities only depend on states involving two strategies at a time.
)pbdoc"
                )
                .def(
                    "calculate_transition_matrix",
                    &FinitePopulations::analytical::PairwiseComparison::calculate_transition_matrix,
                    py::arg("beta"),
                    py::arg("mu"),
                    py::return_value_policy::move,
                    R"pbdoc(
Compute the full transition matrix of the finite-population Markov chain.

The chain evolves on the set of all population states

.. math::

    \mathcal{S} = \left\{x \in \mathbb{N}^n : \sum_{i=1}^n x_i = Z \right\}.

Each off-diagonal transition changes the state by replacing one individual of one
strategy by one individual of another strategy. Mutation is incorporated directly
into the transition probabilities, and diagonal entries are set so that each row
sums to one.

Parameters
----------
beta : float
    Intensity of selection :math:`\beta`.
mu : float
    Mutation probability :math:`\mu`.

Returns
-------
scipy.sparse.csr_matrix
    Sparse transition matrix of shape `(nb_states, nb_states)`.

Notes
-----
For large state spaces, explicitly constructing this matrix may require a large
amount of memory.
)pbdoc"
                )
                .def(
                    "calculate_gradient_of_selection",
                    &FinitePopulations::analytical::PairwiseComparison::calculate_gradient_of_selection,
                    py::arg("beta"),
                    py::arg("state"),
                    R"pbdoc(
Compute the gradient of selection without mutation at a given population state.

Let :math:`x = (x_1,\dots,x_n)` be the current state. This method returns the
expected one-step drift induced only by selection. For each strategy :math:`i`,

.. math::

    g_i(x)
    =
    \frac{1}{n}
    \sum_{j \ne i}
    \left[
    T^{\mathrm{sel}}_{j \to i}(x) - T^{\mathrm{sel}}_{i \to j}(x)
    \right],

where :math:`T^{\mathrm{sel}}_{j \to i}(x)` is the probability that one
individual of strategy :math:`j` is replaced by one individual of strategy
:math:`i` under pairwise comparison alone.

Under the Fermi rule, the local net flux can be written as

.. math::

    T^{\mathrm{sel}}_{j \to i}(x) - T^{\mathrm{sel}}_{i \to j}(x)
    =
    \frac{x_i x_j}{Z(Z-1)}
    \tanh\!\left(\frac{\beta}{2}(f_i(x)-f_j(x))\right).

The resulting vector is tangent to the simplex, so

.. math::

    \sum_{i=1}^n g_i(x) = 0.

Parameters
----------
beta : float
    Intensity of selection :math:`\beta`.
state : numpy.ndarray
    One-dimensional integer array of shape `(nb_strategies,)` containing the
    current population state.

Returns
-------
numpy.ndarray
    One-dimensional array of shape `(nb_strategies,)` containing the mutation-free
    gradient of selection.
)pbdoc"
                )
                .def(
                    "calculate_gradient_of_selection_with_mutation",
                    &FinitePopulations::analytical::PairwiseComparison::calculate_gradient_of_selection_with_mutation,
                    py::arg("beta"),
                    py::arg("mu"),
                    py::arg("state"),
                    R"pbdoc(
Compute the gradient of selection with mutation at a given population state.

Let :math:`x = (x_1,\dots,x_n)` be the current state, with population size
:math:`Z` and :math:`n` strategies. This method returns the expected one-step
drift when both pairwise comparison and mutation are active:

.. math::

    g_i^{(\mu)}(x)
    =
    (1-\mu)\, g_i(x)
    +
    \frac{\mu_{\mathrm{eff}}}{nZ}\left(Z - n x_i\right),

where :math:`g_i(x)` is the mutation-free gradient returned by
:meth:`calculate_gradient_of_selection`, and :math:`\mu_{\mathrm{eff}}` is the
effective mutation probability towards one specific alternative strategy:

.. math::

    \mu_{\mathrm{eff}} =
    \begin{cases}
    \mu, & n = 2, \\
    \mu/(n-1), & n > 2.
    \end{cases}

The first term is the selection contribution scaled by :math:`(1-\mu)`, and the
second term is the mutation drift induced by uniform mutation towards the other
strategies.

As in the mutation-free case, the resulting vector is tangent to the simplex:

.. math::

    \sum_{i=1}^n g_i^{(\mu)}(x) = 0.

Parameters
----------
beta : float
    Intensity of selection :math:`\beta`.
mu : float
    Mutation probability :math:`\mu`.
state : numpy.ndarray
    One-dimensional integer array of shape `(nb_strategies,)` containing the
    current population state.

Returns
-------
numpy.ndarray
    One-dimensional array of shape `(nb_strategies,)` containing the gradient
    with mutation.
)pbdoc"
                )
                .def(
                    "calculate_fixation_probability",
                    &FinitePopulations::analytical::PairwiseComparison::calculate_fixation_probability,
                    py::arg("invading_strategy_index"),
                    py::arg("resident_strategy_index"),
                    py::arg("beta"),
                    R"pbdoc(
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
    Intensity of selection :math:`\beta`.

Returns
-------
float
    Probability that one invader fixates in a population of residents.
)pbdoc"
                )
                .def(
                    "calculate_transition_and_fixation_matrix_sml",
                    &FinitePopulations::analytical::PairwiseComparison::calculate_transition_and_fixation_matrix_sml,
                    py::arg("beta"),
                    py::return_value_policy::move,
                    py::call_guard<py::gil_scoped_release>(),
                    R"pbdoc(
Return the reduced transition matrix and fixation matrix in the small-mutation limit.

In the Small Mutation Limit (SML), mutations are assumed sufficiently rare that
the population is almost always monomorphic before the next mutation occurs.
The resulting reduced Markov chain acts only on the monomorphic states.

If the current monomorphic state is strategy :math:`i`, then for :math:`i \ne j`

.. math::

    T_{ij}^{\mathrm{SML}} = \frac{\rho_{ij}}{n-1},

where :math:`\rho_{ij}` is the fixation probability of one mutant of strategy
:math:`j` in a resident population of strategy :math:`i`. The diagonal entries
are set so that each row sums to one.

Parameters
----------
beta : float
    Intensity of selection :math:`\beta`.

Returns
-------
tuple[numpy.ndarray, numpy.ndarray]
    A tuple `(transition_matrix, fixation_probabilities)` where:

    - `transition_matrix` is the reduced SML transition matrix of shape
      `(nb_strategies, nb_strategies)`;
    - `fixation_probabilities[i, j]` is the probability that one mutant of
      strategy `j` fixates in a population of strategy `i`.
)pbdoc"
                )
                .def(
                    "update_population_size",
                    &egttools::FinitePopulations::analytical::PairwiseComparison::update_population_size,
                    py::arg("population_size"),
                    R"pbdoc(
Update the population size.

Parameters
----------
population_size : int
    New population size :math:`Z`.
)pbdoc"
                )
                .def(
                    "nb_strategies",
                    &egttools::FinitePopulations::analytical::PairwiseComparison::nb_strategies,
                    R"pbdoc(
Return the number of strategies.

Returns
-------
int
    Number of strategies.
)pbdoc"
                )
                .def(
                    "nb_states",
                    &egttools::FinitePopulations::analytical::PairwiseComparison::nb_states,
                    R"pbdoc(
Return the total number of population states.

Returns
-------
int
    Number of states in the full Markov chain.
)pbdoc"
                )
                .def(
                    "population_size",
                    &egttools::FinitePopulations::analytical::PairwiseComparison::population_size,
                    R"pbdoc(
Return the population size.

Returns
-------
int
    Population size :math:`Z`.
)pbdoc"
                )
                .def(
                    "game",
                    &egttools::FinitePopulations::analytical::PairwiseComparison::game,
                    py::return_value_policy::reference_internal,
                    R"pbdoc(
Return the underlying game.

Returns
-------
egttools.games.AbstractGame
    Reference to the game used to evaluate fitness.
)pbdoc"
                );

        options.enable_function_signatures();
    } {
        py::options options;
        options.disable_function_signatures();

        auto pair_comp = py::class_<PairwiseComparison>(
                    m,
                    "PairwiseComparisonNumerical",
                    R"pbdoc(
Numerical solver for evolutionary dynamics under the pairwise comparison rule.
)pbdoc"
                )
                .def(
                    py::init<size_t, FinitePopulations::AbstractGame &, size_t>(),
                    py::arg("pop_size"),
                    py::arg("game"),
                    py::arg("cache_size"),
                    py::keep_alive<1, 3>(),
                    R"pbdoc(
Construct a numerical solver for a finite-population game.

Parameters
----------
pop_size : int
    Number of individuals in the population.
game : egttools.games.AbstractGame
    Game object implementing the payoff and fitness structure.
cache_size : int
    Maximum cache size for fitness computations.
)pbdoc"
                )
                .def(
                    "evolve",
                    static_cast<VectorXui (PairwiseComparison::*)(
                        size_t, double, double, const Eigen::Ref<const VectorXui> &
                    )>(&PairwiseComparison::evolve),
                    py::arg("nb_generations"),
                    py::arg("beta"),
                    py::arg("mu"),
                    py::arg("init_state"),
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
init_state : numpy.ndarray
    Initial population state.

Returns
-------
numpy.ndarray
    Final population state.
)pbdoc"
                )
                .def(
                    "estimate_fixation_probability",
                    &PairwiseComparison::estimate_fixation_probability,
                    py::arg("index_invading_strategy"),
                    py::arg("index_resident_strategy"),
                    py::arg("nb_runs"),
                    py::arg("nb_generations"),
                    py::arg("beta"),
                    py::call_guard<py::gil_scoped_release>(),
                    R"pbdoc(
Estimate the fixation probability of an invading strategy in a resident population.
)pbdoc"
                )
                .def(
                    "estimate_stationary_distribution",
                    [](PairwiseComparison &self,
                       size_t nb_runs, size_t nb_generations, size_t transitory,
                       double beta, double mu) {
                        const double expected_mutations =
                            mu * static_cast<double>(nb_generations - transitory);
                        if (expected_mutations < 10.0) {
                            PyErr_WarnEx(
                                PyExc_UserWarning,
                                "mu is very small relative to (nb_generations - transitory): "
                                "the geometric-skip approximation may produce inaccurate results "
                                "(fewer than 10 expected mutations in the counting window). "
                                "Consider increasing mu, nb_generations, or decreasing transitory.",
                                1);
                        }
                        py::gil_scoped_release release;
                        return self.estimate_stationary_distribution(
                            nb_runs, nb_generations, transitory, beta, mu);
                    },
                    py::arg("nb_runs"),
                    py::arg("nb_generations"),
                    py::arg("transitory"),
                    py::arg("beta"),
                    py::arg("mu"),
                    R"pbdoc(
Estimate the stationary distribution of population states.

.. warning::
   If ``mu * (nb_generations - transitory)`` is much less than 10 (i.e. fewer
   than ~10 mutations are expected in the counting window) a ``UserWarning`` is
   raised. The geometric-skip approximation becomes inaccurate in this regime.
   Increase ``nb_generations``, decrease ``transitory``, or raise ``mu``.

Returns
-------
numpy.ndarray
    Estimated stationary distribution.
)pbdoc"
                )
                .def(
                    "estimate_stationary_distribution_sparse",
                    [](PairwiseComparison &self,
                       size_t nb_runs, size_t nb_generations, size_t transitory,
                       double beta, double mu) {
                        const double expected_mutations =
                            mu * static_cast<double>(nb_generations - transitory);
                        if (expected_mutations < 10.0) {
                            PyErr_WarnEx(
                                PyExc_UserWarning,
                                "mu is very small relative to (nb_generations - transitory): "
                                "the geometric-skip approximation may produce inaccurate results "
                                "(fewer than 10 expected mutations in the counting window). "
                                "Consider increasing mu, nb_generations, or decreasing transitory.",
                                1);
                        }
                        py::gil_scoped_release release;
                        return self.estimate_stationary_distribution_sparse(
                            nb_runs, nb_generations, transitory, beta, mu);
                    },
                    py::arg("nb_runs"),
                    py::arg("nb_generations"),
                    py::arg("transitory"),
                    py::arg("beta"),
                    py::arg("mu"),
                    R"pbdoc(
Estimate the stationary distribution in sparse format.

.. warning::
   If ``mu * (nb_generations - transitory)`` is much less than 10 a
   ``UserWarning`` is raised. See ``estimate_stationary_distribution`` for details.

Returns
-------
scipy.sparse.csr_matrix
    Estimated stationary distribution in sparse format.
)pbdoc"
                )
                .def(
                    "estimate_strategy_distribution",
                    [](PairwiseComparison &self,
                       size_t nb_runs, size_t nb_generations, size_t transitory,
                       double beta, double mu) {
                        const double expected_mutations =
                            mu * static_cast<double>(nb_generations - transitory);
                        if (expected_mutations < 10.0) {
                            PyErr_WarnEx(
                                PyExc_UserWarning,
                                "mu is very small relative to (nb_generations - transitory): "
                                "the geometric-skip approximation may produce inaccurate results "
                                "(fewer than 10 expected mutations in the counting window). "
                                "Consider increasing mu, nb_generations, or decreasing transitory.",
                                1);
                        }
                        py::gil_scoped_release release;
                        return self.estimate_strategy_distribution(
                            nb_runs, nb_generations, transitory, beta, mu);
                    },
                    py::arg("nb_runs"),
                    py::arg("nb_generations"),
                    py::arg("transitory"),
                    py::arg("beta"),
                    py::arg("mu"),
                    R"pbdoc(
Estimate the average frequency of each strategy over time.

.. warning::
   If ``mu * (nb_generations - transitory)`` is much less than 10 a
   ``UserWarning`` is raised. See ``estimate_stationary_distribution`` for details.

Returns
-------
numpy.ndarray
    Average frequency of each strategy.
)pbdoc"
                )
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
                              "Maximum number of cached fitness values.")
                .def(
                    "change_game",
                    &PairwiseComparison::change_game,
                    py::arg("game"),
                    py::keep_alive<1, 2>(),
                    R"pbdoc(
Replace the game used for fitness computation.

The solver retains a pointer to the new game object; the caller must ensure
the game stays alive for the lifetime of this solver (enforced automatically
when called from Python via the keep-alive policy).

Parameters
----------
game : egttools.games.AbstractGame
    New game object. Must have the same number of strategies as the current game.
)pbdoc"
                );

        pair_comp.def(
            "run_without_mutation",
            static_cast<egttools::MatrixXui2D (PairwiseComparison::*)(
                int64_t, double, const Eigen::Ref<const egttools::VectorXui> &
            )>(&PairwiseComparison::run),
            py::arg("nb_generations"),
            py::arg("beta"),
            py::arg("init_state"),
            py::return_value_policy::move,
            R"pbdoc(
Simulate the stochastic dynamics without mutation.

Returns
-------
numpy.ndarray
    Matrix containing all intermediate population states.
)pbdoc"
        );

        pair_comp.def(
            "run_without_mutation",
            static_cast<MatrixXui2D (PairwiseComparison::*)(
                int64_t, int64_t, double, const Eigen::Ref<const VectorXui> &
            )>(&PairwiseComparison::run),
            py::arg("nb_generations"),
            py::arg("transient"),
            py::arg("beta"),
            py::arg("init_state"),
            py::return_value_policy::move,
            R"pbdoc(
Simulate the stochastic dynamics without mutation, skipping the transient phase.

Returns
-------
numpy.ndarray
    Matrix containing the population states after the transient period.
)pbdoc"
        );

        pair_comp.def(
            "run_with_mutation",
            static_cast<MatrixXui2D (PairwiseComparison::*)(
                int64_t, double, double, const Eigen::Ref<const VectorXui> &
            )>(&PairwiseComparison::run),
            py::arg("nb_generations"),
            py::arg("beta"),
            py::arg("mu"),
            py::arg("init_state"),
            py::return_value_policy::move,
            R"pbdoc(
Simulate the stochastic dynamics with mutation.

Returns
-------
numpy.ndarray
    Matrix containing all intermediate population states.
)pbdoc"
        );

        pair_comp.def(
            "run_with_mutation",
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
Simulate the stochastic dynamics with mutation, skipping the transient phase.

Returns
-------
numpy.ndarray
    Matrix containing the population states after the transient period.
)pbdoc"
        );

        pair_comp.def("run", [](pybind11::object &self, py::args args) -> void {
            PyErr_WarnEx(
                PyExc_DeprecationWarning,
                "DEPRECATED. Use run_without_mutation or run_with_mutation instead.",
                1
            );
        });

        options.enable_function_signatures();
    } {
        py::options options;
        options.disable_function_signatures();

        py::class_<FinitePopulations::evolvers::GeneralPopulationEvolver>(
                    m,
                    "GeneralPopulationEvolver",
                    R"pbdoc(
Evolver for a general population structure.
)pbdoc"
                )
                .def(
                    py::init<FinitePopulations::structure::AbstractStructure &>(),
                    py::arg("structure"),
                    py::keep_alive<1, 2>(),
                    R"pbdoc(
Construct an evolver for a general population structure.

Parameters
----------
structure : egttools.numerical.structure.AbstractStructure
    Structure defining how individuals interact and update their strategies.
)pbdoc"
                )
                .def(
                    "evolve",
                    &FinitePopulations::evolvers::GeneralPopulationEvolver::evolve,
                    py::call_guard<py::gil_scoped_release>(),
                    py::arg("nb_generations"),
                    py::return_value_policy::move,
                    R"pbdoc(
Evolve the population and return the final state.
)pbdoc"
                )
                .def(
                    "run",
                    &egttools::FinitePopulations::evolvers::GeneralPopulationEvolver::run,
                    py::call_guard<py::gil_scoped_release>(),
                    py::arg("nb_generations"),
                    py::arg("transitory"),
                    py::return_value_policy::move,
                    R"pbdoc(
Run the population dynamics and return the result after discarding the transitory phase.
)pbdoc"
                )
                .def(
                    "structure",
                    &FinitePopulations::evolvers::GeneralPopulationEvolver::structure,
                    py::return_value_policy::reference_internal,
                    R"pbdoc(
Return the structure used by the evolver.
)pbdoc"
                );

        py::class_<FinitePopulations::evolvers::NetworkEvolver>(
                    m,
                    "NetworkEvolver",
                    R"pbdoc(
Utility class for evolving network-structured populations.
)pbdoc"
                )
                .def_static(
                    "evolve",
                    static_cast<VectorXui (*)(
                        int64_t,
                        FinitePopulations::structure::AbstractNetworkStructure &
                    )>(&FinitePopulations::evolvers::NetworkEvolver::evolve),
                    py::arg("nb_generations"),
                    py::arg("network"),
                    py::return_value_policy::move,
                    R"pbdoc(
Evolve the network population and return the final state.
)pbdoc"
                )

                .def_static(
                    "evolve",
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
Evolve the network population from a given initial state and return the final state.
)pbdoc"
                )

                .def_static(
                    "run",
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
Simulate the full trajectory of the population states.
)pbdoc"
                )

                .def_static(
                    "run",
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
Run the simulation from a custom initial state and return the trajectory.
)pbdoc"
                )

                .def_static(
                    "estimate_time_dependent_average_gradients_of_selection",
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
Estimate time-dependent average gradients of selection for a set of states.
)pbdoc"
                )

                .def_static(
                    "estimate_time_dependent_average_gradients_of_selection",
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
Estimate time-dependent average gradients of selection across multiple networks.
)pbdoc"
                )

                .def_static(
                    "estimate_time_independent_average_gradients_of_selection",
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
Estimate time-independent average gradients of selection for a set of states.
)pbdoc"
                )

                .def_static(
                    "estimate_time_independent_average_gradients_of_selection",
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
Estimate time-independent average gradients of selection across multiple networks.
)pbdoc"
                );

        options.enable_function_signatures();
    }
}

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
            [](size_t pop_size, size_t nb_strategies, egttools::SparseMatrix2D &sd) {
                if (sd.rows() > 1 && sd.cols() == 1) {
                    egttools::SparseMatrix2D sd_row = sd.transpose();
                    return utils::calculate_strategies_distribution(pop_size, nb_strategies, sd_row);
                }
                return utils::calculate_strategies_distribution(pop_size, nb_strategies, sd);
            },
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
            [](int64_t pop_size, int64_t group_size, int64_t nb_strategies,
               egttools::SparseMatrix2D &sd, egttools::Matrix2D &payoff_matrix) {
                if (sd.rows() > 1 && sd.cols() == 1) {
                    egttools::SparseMatrix2D sd_row = sd.transpose();
                    return utils::calculate_expected_payoff(pop_size, group_size, nb_strategies, sd_row, payoff_matrix);
                }
                return utils::calculate_expected_payoff(pop_size, group_size, nb_strategies, sd, payoff_matrix);
            },
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
            [](int64_t pop_size, int64_t group_size, int64_t nb_strategies,
               egttools::SparseMatrix2D &sd,
               const std::function<double(const std::vector<size_t> &)> &indicator) {
                if (sd.rows() > 1 && sd.cols() == 1) {
                    egttools::SparseMatrix2D sd_row = sd.transpose();
                    return utils::calculate_expected_indicator(pop_size, group_size, nb_strategies, sd_row, indicator);
                }
                return utils::calculate_expected_indicator(pop_size, group_size, nb_strategies, sd, indicator);
            },
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
            [](int64_t pop_size, int64_t group_size, int64_t nb_strategies,
               egttools::SparseMatrix2D &sd,
               const std::vector<std::function<double(const std::vector<size_t> &)>> &indicators) {
                if (sd.rows() > 1 && sd.cols() == 1) {
                    egttools::SparseMatrix2D sd_row = sd.transpose();
                    return utils::calculate_expected_indicators(pop_size, group_size, nb_strategies, sd_row, indicators);
                }
                return utils::calculate_expected_indicators(pop_size, group_size, nb_strategies, sd, indicators);
            },
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
            [](int64_t pop_size, int64_t group_size, int64_t nb_strategies,
               egttools::SparseMatrix2D &sd, const egttools::Matrix2D &indicator_matrix) {
                if (sd.rows() > 1 && sd.cols() == 1) {
                    egttools::SparseMatrix2D sd_row = sd.transpose();
                    return utils::calculate_expected_indicators_precomputed(pop_size, group_size, nb_strategies, sd_row, indicator_matrix);
                }
                return utils::calculate_expected_indicators_precomputed(pop_size, group_size, nb_strategies, sd, indicator_matrix);
            },
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
            [](int64_t pop_size, int64_t group_size, int64_t nb_strategies,
               egttools::SparseMatrix2D &sd, int64_t threshold,
               const std::vector<int64_t> &contributing_strategies) {
                if (sd.rows() > 1 && sd.cols() == 1) {
                    egttools::SparseMatrix2D sd_row = sd.transpose();
                    return utils::calculate_expected_group_success(pop_size, group_size, nb_strategies, sd_row, threshold, contributing_strategies);
                }
                return utils::calculate_expected_group_success(pop_size, group_size, nb_strategies, sd, threshold, contributing_strategies);
            },
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
            py::call_guard<py::gil_scoped_release>()
        );

        // -----------------------------------------------------------------------
        // State-level expected indicators  E[f] = Σ_s  sd(s) · f(s)
        // -----------------------------------------------------------------------

        m.def(
            "calculate_expected_state_indicator",
            [](size_t pop_size, size_t nb_strategies, egttools::SparseMatrix2D &sd,
               const std::function<double(const std::vector<size_t> &)> &indicator) {
                if (sd.rows() > 1 && sd.cols() == 1) {
                    egttools::SparseMatrix2D sd_row = sd.transpose();
                    return utils::calculate_expected_state_indicator(pop_size, nb_strategies, sd_row, indicator);
                }
                return utils::calculate_expected_state_indicator(pop_size, nb_strategies, sd, indicator);
            },
            R"pbdoc(
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
)pbdoc",
            py::arg("pop_size"),
            py::arg("nb_strategies"),
            py::arg("stationary_distribution"),
            py::arg("indicator")
            // GIL must be held: indicator is a Python callable.
        );

        m.def(
            "calculate_expected_state_indicators",
            [](size_t pop_size, size_t nb_strategies, egttools::SparseMatrix2D &sd,
               const std::vector<std::function<double(const std::vector<size_t> &)>> &indicators) {
                if (sd.rows() > 1 && sd.cols() == 1) {
                    egttools::SparseMatrix2D sd_row = sd.transpose();
                    return utils::calculate_expected_state_indicators(pop_size, nb_strategies, sd_row, indicators);
                }
                return utils::calculate_expected_state_indicators(pop_size, nb_strategies, sd, indicators);
            },
            R"pbdoc(
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
)pbdoc",
            py::arg("pop_size"),
            py::arg("nb_strategies"),
            py::arg("stationary_distribution"),
            py::arg("indicators")
            // GIL must be held: indicators are Python callables.
        );

        m.def(
            "calculate_expected_state_indicators_precomputed",
            [](egttools::SparseMatrix2D &sd, const egttools::Matrix2D &indicator_values) {
                if (sd.rows() > 1 && sd.cols() == 1) {
                    egttools::SparseMatrix2D sd_row = sd.transpose();
                    return utils::calculate_expected_state_indicators_precomputed(sd_row, indicator_values);
                }
                return utils::calculate_expected_state_indicators_precomputed(sd, indicator_values);
            },
            R"pbdoc(
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
)pbdoc",
            py::arg("stationary_distribution"),
            py::arg("indicator_values"),
            py::return_value_policy::move,
            py::call_guard<py::gil_scoped_release>()
        );

        m.def(
            "precompute_group_to_state_indicator_matrix",
            [](const int64_t pop_size, const int64_t group_size, const int64_t nb_strategies,
               const std::vector<std::function<double(const std::vector<size_t> &)>> &indicators) {
                // Phase 1 (GIL held): evaluate indicators on all group configs.
                // Phase 2 (pure C++): hypergeometric-weighted sum over states.
                // Release GIL for phase 2 only via manual scope.
                Matrix2D result = utils::precompute_group_to_state_indicator_matrix(
                    pop_size, group_size, nb_strategies, indicators);
                return result;
            },
            R"pbdoc(
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
)pbdoc",
            py::arg("pop_size"),
            py::arg("group_size"),
            py::arg("nb_strategies"),
            py::arg("indicators"),
            py::return_value_policy::move
            // GIL held for phase 1 (Python callbacks); phase 2 is pure C++
            // but handled inside the function — acceptable for typical nb_states.
        );

        m.def(
            "calculate_hypergeometric_expected_value",
            [](const size_t pop_size, const size_t group_size, const size_t nb_strategies,
               const Eigen::Ref<const VectorXui> &state,
               const Eigen::Ref<const egttools::Vector> &f_values) -> double {
                return egttools::utils::calculate_hypergeometric_expected_value(
                    pop_size, group_size, nb_strategies, state, f_values);
            },
            R"pbdoc(
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
)pbdoc",
            py::arg("pop_size"),
            py::arg("group_size"),
            py::arg("nb_strategies"),
            py::arg("state"),
            py::arg("f_values"),
            py::call_guard<py::gil_scoped_release>()
        );

        m.def(
            "calculate_hypergeometric_fitness",
            [](const int player_type, const size_t pop_size,
               const size_t group_size, const size_t nb_strategies,
               const Eigen::Ref<const VectorXui> &strategies,
               const Eigen::Ref<const egttools::Vector> &payoffs_row) -> double {
                return egttools::utils::calculate_hypergeometric_fitness(
                    player_type, pop_size, group_size, nb_strategies,
                    strategies, payoffs_row);
            },
            R"pbdoc(
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
)pbdoc",
            py::arg("player_type"),
            py::arg("pop_size"),
            py::arg("group_size"),
            py::arg("nb_strategies"),
            py::arg("strategies"),
            py::arg("payoffs_row"),
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
                    [](FinitePopulations::analytical::PairwiseComparison &self,
                       const double beta,
                       const double mu) -> SparseMatrix2D {
                        // Phase 1: pre-compute fitness values serially.
                        // game_.calculate_fitness() may call back into Python, so the GIL
                        // must be held here.  compute_fitness_matrix() makes no attempt to
                        // release the GIL internally.
                        egttools::Matrix2D fitness = self.compute_fitness_matrix();

                        // Phase 2: assemble the sparse matrix in parallel.
                        // No Python callbacks are made after this point, so it is safe to
                        // release the GIL and let OpenMP threads run freely.
                        py::gil_scoped_release release;
                        return self.assemble_transition_matrix_from_fitness(beta, mu, fitness);
                    },
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

Implementation note: fitness values are pre-computed in a serial pass (with the
GIL held, so Python-subclassed games work correctly), then the sparse matrix is
assembled in a parallel OpenMP pass (with the GIL released).
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
                    [](FinitePopulations::analytical::PairwiseComparison &self,
                       int inv, int res, double beta, bool high_precision) -> double {
#if (HAS_BOOST)
                        if (high_precision)
                            return self.calculate_fixation_probability_boost(inv, res, beta);
#else
                        if (high_precision) {
                            auto warnings = py::module_::import("warnings");
                            warnings.attr("warn")(
                                "high_precision=True requires Boost multiprecision, which is not "
                                "available in this build; falling back to double precision.",
                                py::module_::import("builtins").attr("UserWarning"));
                        }
#endif
                        return self.calculate_fixation_probability(inv, res, beta);
                    },
                    py::arg("invading_strategy_index"),
                    py::arg("resident_strategy_index"),
                    py::arg("beta"),
                    py::arg("high_precision") = false,
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
high_precision : bool, optional
    When *True* and the library was compiled with Boost multiprecision support,
    the internal streaming log-sum-exp is evaluated in 50-digit decimal
    arithmetic, extending the representable range of :math:`\exp(\log\phi)`.
    The return value is still a ``float``; values below ``DBL_MIN`` (~2.2e-308)
    are rounded to zero regardless.  Use
    :meth:`calculate_log_fixation_probability` for the full dynamic range.
    Defaults to ``False``.

Returns
-------
float
    Probability that one invader fixates in a population of residents.
)pbdoc"
                )
                .def(
                    "calculate_log_fixation_probability",
                    &FinitePopulations::analytical::PairwiseComparison::calculate_log_fixation_probability,
                    py::arg("invading_strategy_index"),
                    py::arg("resident_strategy_index"),
                    py::arg("beta"),
                    R"pbdoc(
Return the natural log of the fixation probability, :math:`\log\rho`.

Unlike :meth:`calculate_fixation_probability`, this method never underflows: the
result is a finite ``float`` for any combination of :math:`\beta`, population
size, and fitness values.  It is computed as
:math:`-\mathrm{softplus}(\log\phi)`, so values as small as
:math:`e^{-10^{308}}` are representable.

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
    :math:`\log\rho(\text{invader} \to \text{resident})`, always finite.
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
                    "calculate_transition_and_log_fixation_matrix_sml",
                    &FinitePopulations::analytical::PairwiseComparison::calculate_transition_and_log_fixation_matrix_sml,
                    py::arg("beta"),
                    py::return_value_policy::move,
                    py::call_guard<py::gil_scoped_release>(),
                    R"pbdoc(
Return a numerically stable SML transition matrix and the log-fixation matrix.

Like :meth:`calculate_transition_and_fixation_matrix_sml` but stores
:math:`\log\rho_{ij}` and builds the transition matrix by scaling all
off-diagonal entries by :math:`\exp(-\max_{k\ne l}\log\rho_{kl})`.  This global
rescaling preserves the stationary distribution while guaranteeing a valid
stochastic matrix even when every :math:`\rho_{ij}` underflows to zero in
double precision.

Parameters
----------
beta : float
    Intensity of selection :math:`\beta`.

Returns
-------
tuple[numpy.ndarray, numpy.ndarray]
    A tuple `(transition_matrix, log_fixation_probabilities)` where:

    - `transition_matrix` has the same stationary distribution as the standard
      SML matrix but is numerically well-conditioned;
    - `log_fixation_probabilities[i, j]` = :math:`\log\rho_{ij}`.
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
                    "estimate_mean_absorption_time",
                    [](PairwiseComparison &self,
                       double beta,
                       const Eigen::Ref<const VectorXui> &init_state,
                       size_t nb_runs) -> py::dict {
                        py::gil_scoped_release release;
                        auto [mean, se] = self.estimate_mean_absorption_time(beta, init_state, nb_runs);
                        py::gil_scoped_acquire acquire;
                        py::dict result;
                        result["mean"] = mean;
                        result["stderr"] = se;
                        result["nb_runs"] = nb_runs;
                        return result;
                    },
                    py::arg("beta"),
                    py::arg("init_state"),
                    py::arg("nb_runs"),
                    R"pbdoc(
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
)pbdoc"
                )
                .def(
                    "estimate_absorption_probabilities",
                    [](PairwiseComparison &self,
                       double beta,
                       const Eigen::Ref<const VectorXui> &init_state,
                       size_t nb_runs) {
                        py::gil_scoped_release release;
                        return self.estimate_absorption_probabilities(beta, init_state, nb_runs);
                    },
                    py::arg("beta"),
                    py::arg("init_state"),
                    py::arg("nb_runs"),
                    py::return_value_policy::move,
                    R"pbdoc(
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
)pbdoc"
                )
                .def(
                    "estimate_stationary_distribution",
                    [](PairwiseComparison &self,
                       size_t nb_runs, size_t nb_generations, size_t transitory,
                       double beta, double mu,
                       double tolerance, size_t check_every) {
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
                            nb_runs, nb_generations, transitory, beta, mu, tolerance, check_every);
                    },
                    py::arg("nb_runs"),
                    py::arg("nb_generations"),
                    py::arg("transitory"),
                    py::arg("beta"),
                    py::arg("mu"),
                    py::arg("tolerance") = 0.0,
                    py::arg("check_every") = 0,
                    R"pbdoc(
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
)pbdoc"
                )
                .def(
                    "estimate_stationary_distribution_sparse",
                    [](PairwiseComparison &self,
                       size_t nb_runs, size_t nb_generations, size_t transitory,
                       double beta, double mu,
                       double tolerance, size_t check_every) {
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
                            nb_runs, nb_generations, transitory, beta, mu, tolerance, check_every);
                    },
                    py::arg("nb_runs"),
                    py::arg("nb_generations"),
                    py::arg("transitory"),
                    py::arg("beta"),
                    py::arg("mu"),
                    py::arg("tolerance") = 0.0,
                    py::arg("check_every") = 0,
                    R"pbdoc(
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
)pbdoc"
                )
                .def(
                    "estimate_strategy_distribution",
                    [](PairwiseComparison &self,
                       size_t nb_runs, size_t nb_generations, size_t transitory,
                       double beta, double mu,
                       double tolerance, size_t check_every) {
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
                            nb_runs, nb_generations, transitory, beta, mu, tolerance, check_every);
                    },
                    py::arg("nb_runs"),
                    py::arg("nb_generations"),
                    py::arg("transitory"),
                    py::arg("beta"),
                    py::arg("mu"),
                    py::arg("tolerance") = 0.0,
                    py::arg("check_every") = 0,
                    R"pbdoc(
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
)pbdoc"
                )
                .def_property_readonly("nb_strategies", &PairwiseComparison::nb_strategies,
                                       "Number of strategies in the population.")
                .def_property_readonly("payoffs", &PairwiseComparison::payoffs,
                                       "Payoff matrix used for selection dynamics.")
                .def_property_readonly("nb_states", &PairwiseComparison::nb_states,
                                       "Number of discrete states in the population.")
                .def_property_readonly("mutation_matrix", &PairwiseComparison::mutation_matrix,
                                       "Current (nb_strategies, nb_strategies) mutation matrix. Row i is "
                                       "the (unnormalized) distribution over target strategies when "
                                       "mutating away from strategy i; the diagonal is always 0. Defaults "
                                       "to uniform (all-ones off-diagonal) until set_mutation_weights is "
                                       "called.")
                .def(
                    "set_mutation_matrix",
                    &PairwiseComparison::set_mutation_matrix,
                    py::arg("mutation_matrix"),
                    R"pbdoc(
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
)pbdoc"
                )
                .def(
                    "set_mutation_weights",
                    [](PairwiseComparison &self, py::object weights) -> void {
                        py::array arr = py::array::ensure(weights);
                        if (!arr)
                            throw py::type_error("mutation_weights must be array-like.");
                        if (arr.ndim() == 1) {
                            self.set_mutation_weights(arr.cast<Vector>());
                        } else if (arr.ndim() == 2) {
                            self.set_mutation_matrix(arr.cast<Matrix2D>());
                        } else {
                            throw py::value_error(
                                "mutation_weights must be a 1D vector (length nb_strategies) or a "
                                "2D matrix (nb_strategies, nb_strategies), got an array with " +
                                std::to_string(arr.ndim()) + " dimensions.");
                        }
                    },
                    py::arg("weights"),
                    R"pbdoc(
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
)pbdoc"
                )
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
                )
                .def(
                    "estimate_stationary_indicators_precomputed",
                    [](PairwiseComparison &self,
                       size_t nb_runs, size_t nb_generations, size_t transitory,
                       double beta, double mu,
                       const Eigen::Ref<const Matrix2D> &indicator_values,
                       double tolerance, size_t check_every) {
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
                        return self.estimate_stationary_indicators(
                            nb_runs, nb_generations, transitory, beta, mu,
                            indicator_values, tolerance, check_every);
                    },
                    py::arg("nb_runs"),
                    py::arg("nb_generations"),
                    py::arg("transitory"),
                    py::arg("beta"),
                    py::arg("mu"),
                    py::arg("indicator_values"),
                    py::arg("tolerance") = 0.0,
                    py::arg("check_every") = 0,
                    py::return_value_policy::move,
                    R"pbdoc(
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
)pbdoc"
                )
                .def(
                    "estimate_stationary_indicators",
                    [](PairwiseComparison &self,
                       py::object indicators,
                       size_t nb_runs, size_t nb_generations, size_t transitory,
                       double beta, double mu,
                       const std::string &indicator_type,
                       py::object group_size_obj,
                       double tolerance, size_t check_every,
                       double confidence, bool verbose, int n_bootstrap,
                       int64_t precompute_limit) -> py::object {

                        // --- normalise to list ----------------------------------------
                        py::list indicator_list;
                        if (py::isinstance<py::list>(indicators) || py::isinstance<py::tuple>(indicators)) {
                            for (auto item : indicators)
                                indicator_list.append(item);
                        } else if (py::hasattr(indicators, "__call__")) {
                            indicator_list.append(indicators);
                        } else {
                            throw py::type_error("indicators must be a callable or a list/tuple of callables");
                        }
                        const auto nb_indicator_count = static_cast<int64_t>(py::len(indicator_list));

                        // --- build precomputed indicator matrix (GIL held) ------------
                        auto egt = py::module_::import("egttools");
                        const int64_t nb_states     = static_cast<int64_t>(self.nb_states());
                        const int64_t nb_strategies = static_cast<int64_t>(self.nb_strategies());
                        const int64_t pop_size      = static_cast<int64_t>(self.population_size());

                        // Would the dense (nb_states x nb_indicators) matrix exceed the
                        // memory/time budget?  Computed in double to avoid overflow when
                        // nb_states itself is astronomically large.
                        const bool matrix_too_large =
                            static_cast<double>(nb_states) * static_cast<double>(nb_indicator_count) >
                            static_cast<double>(precompute_limit);

                        Matrix2D indicator_matrix;
                        bool use_direct = false;

                        if (indicator_type == "state") {
                            if (matrix_too_large) {
                                // nb_states is too large to enumerate/store a dense
                                // indicator matrix.  Fall back to evaluating the
                                // indicators directly on the live simulation state at
                                // each recorded step (see estimate_stationary_indicators_direct):
                                // O(nb_indicators) memory instead of O(nb_states).
                                use_direct = true;
                            } else {
                                indicator_matrix.resize(nb_states, nb_indicator_count);
                                for (int64_t s = 0; s < nb_states; ++s) {
                                    py::object state = egt.attr("sample_simplex")(s, pop_size, nb_strategies);
                                    for (int64_t k = 0; k < nb_indicator_count; ++k) {
                                        indicator_matrix(s, k) =
                                            indicator_list[k](state).template cast<double>();
                                    }
                                }
                            }
                        } else if (indicator_type == "group") {
                            if (matrix_too_large) {
                                throw std::invalid_argument(
                                    "Population state space is too large (nb_states=" +
                                    std::to_string(nb_states) + ", nb_indicators=" +
                                    std::to_string(nb_indicator_count) +
                                    ") to build the group-level indicator matrix (nb_states * "
                                    "nb_indicators must be <= precompute_limit=" +
                                    std::to_string(precompute_limit) + "). Group-level indicators "
                                    "do not yet support state spaces this large; use "
                                    "indicator_type='state' (which falls back automatically to a "
                                    "memory-bounded direct estimator) or reduce the population "
                                    "size / number of strategies.");
                            }
                            if (group_size_obj.is_none())
                                throw std::invalid_argument(
                                    "group_size must be specified when indicator_type='group'.");
                            const int64_t group_size = group_size_obj.cast<int64_t>();

                            // Validate that the game's payoff matrix is consistent with the
                            // requested group_size.  An N-player game must have exactly
                            // stars_bars(group_size, nb_strategies) group configurations as
                            // columns.  A 2-player (matrix) game has nb_strategies columns,
                            // which will not match for group_size > 2.
                            const int64_t expected_cols =
                                egttools::starsBars<int64_t>(group_size, nb_strategies);
                            const int64_t actual_cols =
                                static_cast<int64_t>(self.payoffs().cols());
                            if (actual_cols != expected_cols) {
                                throw std::invalid_argument(
                                    "Payoff matrix has " + std::to_string(actual_cols) +
                                    " column(s) but group_size=" + std::to_string(group_size) +
                                    " with nb_strategies=" + std::to_string(nb_strategies) +
                                    " requires " + std::to_string(expected_cols) +
                                    " group configurations (stars-and-bars). "
                                    "For group-level indicators the game must be an N-player "
                                    "game whose payoff matrix follows the sample_simplex "
                                    "enumeration — use MatrixNPlayerGameHolder (or a subclass) "
                                    "rather than a 2-player game.");
                            }

                            // Wrap callables: C++ passes std::vector<size_t> which pybind11
                            // converts to a Python list; the wrapper turns it into np.ndarray
                            // so users can write ``lambda g: g[0] >= threshold``.
                            auto np = py::module_::import("numpy");
                            py::list wrapped_list;
                            for (int64_t k = 0; k < nb_indicator_count; ++k) {
                                py::object f = indicator_list[k];
                                wrapped_list.append(
                                    py::cpp_function([f, np](py::object group) -> double {
                                        py::object arr = np.attr("asarray")(
                                            group, py::arg("dtype") = np.attr("intp"));
                                        return f(arr).template cast<double>();
                                    }));
                            }
                            indicator_matrix =
                                egt.attr("precompute_group_to_state_indicator_matrix")(
                                    pop_size, group_size, nb_strategies, wrapped_list)
                                    .template cast<Matrix2D>();
                        } else {
                            throw std::invalid_argument(
                                "indicator_type must be 'state' or 'group', got '" +
                                indicator_type + "'.");
                        }

                        // --- mu sanity warning ----------------------------------------
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

                        // --- run C++ simulation -----------------------------------------
                        Matrix2D per_run;
                        if (use_direct) {
                            // Wrap callables: C++ passes std::vector<size_t> which pybind11
                            // converts to a Python list; the wrapper turns it into np.ndarray
                            // so users can write ``lambda s: s[0] / Z`` regardless of which
                            // path is used. The GIL is intentionally NOT released here:
                            // estimate_stationary_indicators_direct is single-threaded and
                            // calls back into these Python callables on the same thread.
                            auto np = py::module_::import("numpy");
                            std::vector<std::function<double(const std::vector<size_t> &)>> direct_indicators;
                            direct_indicators.reserve(static_cast<size_t>(nb_indicator_count));
                            for (int64_t k = 0; k < nb_indicator_count; ++k) {
                                py::object f = indicator_list[k];
                                direct_indicators.emplace_back(
                                    [f, np](const std::vector<size_t> &state) -> double {
                                        py::object arr = np.attr("asarray")(
                                            state, py::arg("dtype") = np.attr("intp"));
                                        return f(arr).template cast<double>();
                                    });
                            }
                            per_run = self.estimate_stationary_indicators_direct(
                                nb_runs, nb_generations, transitory, beta, mu,
                                direct_indicators, tolerance, check_every);
                        } else {
                            py::gil_scoped_release release;
                            per_run = self.estimate_stationary_indicators(
                                nb_runs, nb_generations, transitory, beta, mu,
                                indicator_matrix, tolerance, check_every);
                        }

                        const int64_t nb_runs_used = per_run.rows();
                        const bool converged =
                            (tolerance > 0.0) && (static_cast<size_t>(nb_runs_used) < nb_runs);

                        // --- statistics ----------------------------------------------
                        py::object per_run_np = py::cast(per_run);
                        py::object grand_mean_np =
                            py::module_::import("numpy").attr("mean")(per_run_np, py::arg("axis") = 0);

                        auto indicators_mod =
                            py::module_::import("egttools.numerical.indicators");
                        auto ci_tuple = indicators_mod
                            .attr("_bootstrap_ci")(per_run_np, confidence, n_bootstrap)
                            .cast<py::tuple>();

                        // --- build result --------------------------------------------
                        auto StationaryIndicatorResult =
                            indicators_mod.attr("StationaryIndicatorResult");
                        return StationaryIndicatorResult(
                            grand_mean_np,
                            py::make_tuple(ci_tuple[0], ci_tuple[1]),
                            nb_runs_used,
                            converged,
                            verbose ? per_run_np : py::none());
                    },
                    py::arg("indicators"),
                    py::arg("nb_runs"),
                    py::arg("nb_generations"),
                    py::arg("transitory"),
                    py::arg("beta"),
                    py::arg("mu"),
                    py::arg("indicator_type") = "state",
                    py::arg("group_size")     = py::none(),
                    py::arg("tolerance")      = 0.0,
                    py::arg("check_every")    = 0,
                    py::arg("confidence")     = 0.95,
                    py::arg("verbose")        = false,
                    py::arg("n_bootstrap")    = 9999,
                    py::arg("precompute_limit") = 20'000'000,
                    R"pbdoc(
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


        options.enable_function_signatures();
    }

    // -------------------------------------------------------------------------
    // PairwiseComparisonTransitionOperator
    // -------------------------------------------------------------------------
    {
        using TransitionOperator = FinitePopulations::PairwiseComparisonTransitionOperator;

        py::options options;
        options.disable_function_signatures();

        py::class_<TransitionOperator>(
            m,
            "PairwiseComparisonTransitionOperator",
            R"pbdoc(
Matrix-free transition operator for the pairwise comparison process.

Computes matrix-vector products ``y = P x``, ``y = P^T x``, and
``y = (I - P^T) x`` without ever assembling the transition matrix P.
Designed for iterative eigensolvers (``scipy.sparse.linalg``, petsc4py)
and as the basis for future MPI-distributed computation.

The stationary distribution π satisfies ``P^T π = π``. Use
``apply_transpose`` or wrap this object with
``egttools.numerical.linear_operator.make_transition_operator`` to
obtain a ``scipy.sparse.linalg.LinearOperator``.
)pbdoc"
        )
        .def(
            py::init<size_t, FinitePopulations::AbstractGame &, double, double>(),
            py::arg("population_size"),
            py::arg("game"),
            py::arg("beta"),
            py::arg("mu"),
            py::keep_alive<1, 3>(),
            R"pbdoc(
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
)pbdoc"
        )
        .def(
            "apply_transpose",
            [](TransitionOperator &self,
               const Eigen::Ref<const egttools::Vector> &x,
               Eigen::Ref<egttools::Vector> y) {
                self.apply_transpose(x, y);
            },
            py::arg("x"),
            py::arg("y"),
            R"pbdoc(
Compute y = P^T x in-place.

The stationary distribution π satisfies P^T π = π, so this is the
primary operation for iterative eigensolver use.

Parameters
----------
x : numpy.ndarray
    Input vector of length ``size``.
y : numpy.ndarray
    Output vector of length ``size``; zeroed and overwritten.
)pbdoc"
        )
        .def(
            "apply",
            [](TransitionOperator &self,
               const Eigen::Ref<const egttools::Vector> &x,
               Eigen::Ref<egttools::Vector> y) {
                self.apply(x, y);
            },
            py::arg("x"),
            py::arg("y"),
            R"pbdoc(
Compute y = P x in-place.

Parameters
----------
x : numpy.ndarray
    Input vector of length ``size``.
y : numpy.ndarray
    Output vector of length ``size``; zeroed and overwritten.
)pbdoc"
        )
        .def(
            "apply_residual",
            [](TransitionOperator &self,
               const Eigen::Ref<const egttools::Vector> &x,
               Eigen::Ref<egttools::Vector> y) {
                self.apply_residual(x, y);
            },
            py::arg("x"),
            py::arg("y"),
            R"pbdoc(
Compute y = (I - P^T) x in-place.

Useful for iterative linear solvers: find π such that (I - P^T) π = 0.

Parameters
----------
x : numpy.ndarray
    Input vector of length ``size``.
y : numpy.ndarray
    Output vector of length ``size``; overwritten.
)pbdoc"
        )
        .def_property_readonly(
            "size",
            &TransitionOperator::size,
            "Total number of simplex states C(Z+k-1, k-1)."
        )
        .def_property_readonly(
            "population_size",
            &TransitionOperator::population_size,
            "Population size Z."
        )
        .def_property_readonly(
            "nb_strategies",
            &TransitionOperator::nb_strategies,
            "Number of strategies k."
        )
        .def_property_readonly(
            "beta",
            &TransitionOperator::beta,
            "Intensity of selection β."
        )
        .def_property_readonly(
            "mu",
            &TransitionOperator::mu,
            "Mutation probability μ."
        )
        .def(
            "compute_stationary_distribution",
            [](TransitionOperator &self, double tol, size_t max_iter) {
                return self.compute_stationary_distribution(tol, max_iter);
            },
            py::arg("tol")      = 1e-10,
            py::arg("max_iter") = size_t(10000),
            R"pbdoc(
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
)pbdoc"
        )
#if HAS_ARPACK
        .def(
            "compute_stationary_arpack",
            [](TransitionOperator &self, double tol, int ncv, int max_iter) {
                return self.compute_stationary_arpack(tol, ncv, max_iter);
            },
            py::arg("tol")      = 0.0,
            py::arg("ncv")      = 0,
            py::arg("max_iter") = 300,
            R"pbdoc(
Compute the stationary distribution via ARPACK IRAM (pure C++).

Uses ARPACK's implicitly restarted Arnoldi method to find the leading
eigenvector of P^T.  Converges much faster than power iteration when
the spectral gap is small (small μ or large Z), and eliminates Python
callbacks entirely.

Only available when EGTtools is compiled with ``EGTTOOLS_ENABLE_ARPACK=ON``.

Parameters
----------
tol : float
    ARPACK convergence tolerance (default 0.0 → machine precision).
ncv : int
    Krylov subspace size (default 0 → auto: max(2*nev+1, 20)).
max_iter : int
    Maximum Arnoldi iterations (default 300).

Returns
-------
numpy.ndarray
    Normalised stationary distribution of length ``size``.

Raises
------
RuntimeError
    On ARPACK error or non-convergence.
)pbdoc"
        )
#endif // HAS_ARPACK
        ;
    }
}

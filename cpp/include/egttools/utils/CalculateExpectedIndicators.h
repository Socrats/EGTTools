/** Copyright (c) 2019-2021  Elias Fernandez
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

#pragma once
#ifndef EGTTOOLS_UTILS_CALCULATEEXPECTEDINDICATORS_H
#define EGTTOOLS_UTILS_CALCULATEEXPECTEDINDICATORS_H

#include <egttools/Distributions.h>
#include <egttools/Types.h>
#include <egttools/finite_populations/Utils.hpp>

#include <functional>
#include <vector>

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace egttools::utils {

    using SparseMatIt = SparseMatrix2D::InnerIterator;

    /**
     * @brief Calculates the average frequency of each strategy given the stationary distribution.
     *
     * E[freq_i] = sum_s  sd(s) * (count_i(s) / pop_size)
     *
     * @param pop_size               size of the population.
     * @param nb_strategies          number of strategies.
     * @param stationary_distribution sparse row-matrix; non-zero entries are (state_index, probability).
     * @return Vector of length nb_strategies with the average frequency of each strategy.
     */
    Vector calculate_strategies_distribution(size_t pop_size, size_t nb_strategies,
                                             SparseMatrix2D &stationary_distribution);

    /**
     * @brief Calculates the expected payoff averaged over the stationary distribution.
     *
     * E[payoff] = sum_s sd(s) * sum_g P(g|s) * avg_payoff(g)
     *
     * where avg_payoff(g) = sum_j (g[j] / group_size) * payoff_matrix(j, g_index),
     * i.e. each strategy's payoff is weighted by its frequency inside the sampled group.
     *
     * @param pop_size               size of the population.
     * @param group_size             number of individuals sampled per group interaction.
     * @param nb_strategies          number of strategies.
     * @param stationary_distribution sparse row-matrix of stationary probabilities.
     * @param payoff_matrix          (nb_strategies x nb_group_compositions) matrix; entry (j, g)
     *                               is the payoff of strategy j when the group composition is g.
     * @return expected payoff scalar.
     */
    double calculate_expected_payoff(int64_t pop_size, int64_t group_size, int64_t nb_strategies,
                                     SparseMatrix2D &stationary_distribution,
                                     Matrix2D &payoff_matrix);

    /**
     * @brief Calculates E[f] = sum_s sd(s) * sum_g P(g|s) * f(g) for an arbitrary indicator f.
     *
     * This is the core building block for all expected indicators.  The group configuration
     * vector passed to @p indicator contains the counts of each strategy in the sampled group
     * (length nb_strategies, sums to group_size).
     *
     * @tparam IndicatorFn  callable with signature  double(const std::vector<size_t>&)
     * @param pop_size               size of the population.
     * @param group_size             number of individuals sampled per group interaction.
     * @param nb_strategies          number of strategies.
     * @param stationary_distribution sparse row-matrix of stationary probabilities.
     * @param indicator              function mapping a group configuration to a scalar value.
     * @return expected value of the indicator.
     */
    template<typename IndicatorFn>
    double calculate_expected_indicator(int64_t pop_size, int64_t group_size, int64_t nb_strategies,
                                        SparseMatrix2D &stationary_distribution,
                                        IndicatorFn &&indicator) {
        double result = 0.0;
        const auto nb_group_configs = egttools::starsBars<int64_t>(group_size, nb_strategies);
        const auto nb_strategies_sz = static_cast<size_t>(nb_strategies);
        const auto pop_size_sz = static_cast<size_t>(pop_size);
        const auto group_size_sz = static_cast<size_t>(group_size);

        // Outer loop: iterate over non-zero states in the stationary distribution.
        // We do NOT pre-collect the entries to avoid doubling memory for large distributions.
        // Instead we parallelize the inner loop (group configurations) which is the
        // compute-heavy part: each iteration involves a multivariate hypergeometric PDF.
        // The state vector is read-only inside the parallel region so no race condition.
        for (SparseMatIt it(stationary_distribution, 0); it; ++it) {
            egttools::VectorXui state = egttools::VectorXui::Zero(static_cast<signed long>(nb_strategies));
            egttools::FinitePopulations::sample_simplex(
                static_cast<size_t>(it.index()), pop_size_sz, nb_strategies_sz, state);

            double state_contrib = 0.0;

#if defined(_OPENMP)
#pragma omp parallel for reduction(+:state_contrib) schedule(static) \
    firstprivate(nb_strategies_sz, pop_size_sz, group_size_sz)
#endif
            for (int64_t i = 0; i < nb_group_configs; ++i) {
                // Each thread needs its own group_config buffer.
                std::vector<size_t> group_config(nb_strategies_sz, 0);
                egttools::FinitePopulations::sample_simplex(
                    static_cast<size_t>(i), group_size_sz, nb_strategies_sz, group_config);

                const double prob = egttools::multivariateHypergeometricPDF(
                    pop_size_sz, nb_strategies_sz, group_size_sz, group_config, state);

                state_contrib += prob * indicator(group_config);
            }
            result += state_contrib * it.value();
        }
        return result;
    }

    /**
     * @brief Type-erased (std::function) overload of calculate_expected_indicator for pybind11.
     *
     * Prefer the template overload in C++ code.  This overload exists so that Python callables
     * can be passed from pybind11 without needing the template to be instantiated at bind time.
     *
     * @param pop_size               size of the population.
     * @param group_size             number of individuals sampled per group interaction.
     * @param nb_strategies          number of strategies.
     * @param stationary_distribution sparse row-matrix of stationary probabilities.
     * @param indicator              callable mapping a group configuration to a scalar.
     * @return expected value of the indicator.
     */
    double calculate_expected_indicator(int64_t pop_size, int64_t group_size, int64_t nb_strategies,
                                        SparseMatrix2D &stationary_distribution,
                                        const std::function<double(const std::vector<size_t> &)> &indicator);

    /**
     * @brief Calculates the expected group success eta_G under the stationary distribution.
     *
     * eta_G = sum_s sd(s) * sum_g P(g|s) * I(sum_{k in contributing_strategies} g[k] >= threshold)
     *
     * Any strategy whose index appears in @p contributing_strategies is treated as a cooperator
     * for the purpose of the threshold check.  This supports games where multiple strategy types
     * each contribute to collective success (e.g. cooperators + altruists in an extended CRD).
     *
     * @param pop_size                    size of the population.
     * @param group_size                  number of individuals sampled per group interaction.
     * @param nb_strategies               number of strategies.
     * @param stationary_distribution     sparse row-matrix of stationary probabilities.
     * @param threshold                   minimum total count of contributing strategies required
     *                                    for the group to succeed.
     * @param contributing_strategies     indices of the strategies that count towards the threshold.
     * @return expected group success in [0, 1].
     */
    double calculate_expected_group_success(int64_t pop_size, int64_t group_size, int64_t nb_strategies,
                                            SparseMatrix2D &stationary_distribution,
                                            int64_t threshold,
                                            const std::vector<int64_t> &contributing_strategies);

}// namespace egttools::utils

#endif//EGTTOOLS_UTILS_CALCULATEEXPECTEDINDICATORS_H

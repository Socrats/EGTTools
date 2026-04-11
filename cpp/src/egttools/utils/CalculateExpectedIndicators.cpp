/** Copyright (c) 2019-2026  Elias Fernandez
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

#include <egttools/utils/CalculateExpectedIndicators.h>

egttools::Vector egttools::utils::calculate_strategies_distribution(size_t pop_size,
                                                                    size_t nb_strategies,
                                                                    SparseMatrix2D &stationary_distribution) {
    egttools::Vector strategy_distribution = egttools::Vector::Zero(static_cast<signed long>(nb_strategies));
    egttools::VectorXui state = egttools::VectorXui::Zero(static_cast<signed long>(nb_strategies));

    for (SparseMatIt it(stationary_distribution, 0); it; ++it) {
        egttools::FinitePopulations::sample_simplex(it.index(), pop_size, nb_strategies, state);
        strategy_distribution += (state.cast<double>() / static_cast<double>(pop_size)) * it.value();
    }

    return strategy_distribution;
}

double egttools::utils::calculate_expected_payoff(int64_t pop_size, int64_t group_size, int64_t nb_strategies,
                                                  SparseMatrix2D &stationary_distribution,
                                                  Matrix2D &payoff_matrix) {
    // E[payoff] = sum_s sd(s) * sum_g P(g|s) * avg_payoff(g)
    // avg_payoff(g) = sum_j (g[j] / group_size) * payoff_matrix(j, g_index)
    return calculate_expected_indicator(
        pop_size, group_size, nb_strategies, stationary_distribution,
        [&](const std::vector<size_t> &group_config) -> double {
            // Recover the column index in the payoff matrix for this group configuration.
            const auto col = static_cast<int64_t>(
                egttools::FinitePopulations::calculate_state(
                    static_cast<size_t>(group_size), group_config));
            // Weight each strategy's payoff by its frequency inside the sampled group.
            double weighted = 0.0;
            for (int64_t j = 0; j < nb_strategies; ++j) {
                weighted += (static_cast<double>(group_config[static_cast<size_t>(j)]) / static_cast<double>(group_size))
                            * payoff_matrix(j, col);
            }
            return weighted;
        });
}

egttools::Vector egttools::utils::calculate_expected_indicators_precomputed(
    int64_t pop_size, int64_t group_size, int64_t nb_strategies,
    SparseMatrix2D &stationary_distribution,
    const Matrix2D &indicator_matrix) {
    // indicator_matrix: shape (nb_group_configs, nb_indicators), values 0.0/1.0 for boolean indicators.
    //
    // For each nonzero population state s:
    //   1. Compute probs[g] = P(g | s) for all group configurations g.
    //   2. result += sd(s) * indicator_matrix.T @ probs   (BLAS dgemv)
    //
    // Python callables are not involved here at all, so the GIL is not needed.
    // The hot loop is pure Eigen/BLAS.

    const int64_t nb_group_configs = indicator_matrix.rows();
    const int64_t nb_indicators = indicator_matrix.cols();
    const auto nb_strategies_sz = static_cast<size_t>(nb_strategies);
    const auto pop_size_sz = static_cast<size_t>(pop_size);
    const auto group_size_sz = static_cast<size_t>(group_size);

    egttools::Vector result = egttools::Vector::Zero(nb_indicators);
    egttools::VectorXui state = egttools::VectorXui::Zero(static_cast<signed long>(nb_strategies));
    egttools::Vector probs = egttools::Vector::Zero(nb_group_configs);
    std::vector<size_t> group_config(nb_strategies_sz, 0);

    for (SparseMatIt it(stationary_distribution, 0); it; ++it) {
        egttools::FinitePopulations::sample_simplex(
            static_cast<size_t>(it.index()), pop_size_sz, nb_strategies_sz, state);

        // Fill the probability vector for this state.
        for (int64_t i = 0; i < nb_group_configs; ++i) {
            egttools::FinitePopulations::sample_simplex(
                static_cast<size_t>(i), group_size_sz, nb_strategies_sz, group_config);
            probs(i) = egttools::multivariateHypergeometricPDF(
                pop_size_sz, nb_strategies_sz, group_size_sz, group_config, state);
        }

        // BLAS dgemv: result += sd(s) * indicator_matrix.T @ probs
        result.noalias() += it.value() * (indicator_matrix.transpose() * probs);
    }
    return result;
}

egttools::Vector egttools::utils::calculate_expected_indicators(
    int64_t pop_size, int64_t group_size, int64_t nb_strategies,
    SparseMatrix2D &stationary_distribution,
    const std::vector<std::function<double(const std::vector<size_t> &)>> &indicators) {
    // Phase 1 (GIL held): evaluate every indicator on every group configuration once.
    // This replaces O(nb_states x nb_group_configs x K) Python calls with O(nb_group_configs x K).
    const int64_t nb_indicators = static_cast<int64_t>(indicators.size());
    const int64_t nb_group_configs = egttools::starsBars<int64_t>(group_size, nb_strategies);
    const auto nb_strategies_sz = static_cast<size_t>(nb_strategies);
    const auto group_size_sz = static_cast<size_t>(group_size);

    Matrix2D indicator_matrix = Matrix2D::Zero(nb_group_configs, nb_indicators);
    std::vector<size_t> group_config(nb_strategies_sz, 0);

    for (int64_t i = 0; i < nb_group_configs; ++i) {
        egttools::FinitePopulations::sample_simplex(
            static_cast<size_t>(i), group_size_sz, nb_strategies_sz, group_config);
        for (int64_t k = 0; k < nb_indicators; ++k) {
            indicator_matrix(i, k) = indicators[static_cast<size_t>(k)](group_config);
        }
    }

    // Phase 2 (pure C++): the GIL could be released here, but since we are already
    // inside a pybind11 call we leave that to the binding layer.
    return calculate_expected_indicators_precomputed(
        pop_size, group_size, nb_strategies, stationary_distribution, indicator_matrix);
}

double egttools::utils::calculate_expected_indicator(
    int64_t pop_size, int64_t group_size, int64_t nb_strategies,
    SparseMatrix2D &stationary_distribution,
    const std::function<double(const std::vector<size_t> &)> &indicator) {
    // NOTE: this overload is called with a Python callable from pybind11.
    // It deliberately does NOT delegate to the OpenMP template: calling a Python
    // object from an OpenMP worker thread without holding the GIL is undefined
    // behaviour.  The GIL must remain held for the entire duration of this call,
    // and the inner loop must stay serial.
    double result = 0.0;
    const auto nb_group_configs = egttools::starsBars<int64_t>(group_size, nb_strategies);
    const auto nb_strategies_sz = static_cast<size_t>(nb_strategies);
    const auto pop_size_sz = static_cast<size_t>(pop_size);
    const auto group_size_sz = static_cast<size_t>(group_size);

    egttools::VectorXui state = egttools::VectorXui::Zero(static_cast<signed long>(nb_strategies));
    std::vector<size_t> group_config(nb_strategies_sz, 0);

    for (SparseMatIt it(stationary_distribution, 0); it; ++it) {
        egttools::FinitePopulations::sample_simplex(
            static_cast<size_t>(it.index()), pop_size_sz, nb_strategies_sz, state);

        double state_contrib = 0.0;
        for (int64_t i = 0; i < nb_group_configs; ++i) {
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

double egttools::utils::calculate_expected_group_success(int64_t pop_size, int64_t group_size, int64_t nb_strategies,
                                                         SparseMatrix2D &stationary_distribution,
                                                         int64_t threshold,
                                                         const std::vector<int64_t> &contributing_strategies) {
    // eta_G = sum_s sd(s) * sum_g P(g|s) * I(sum_{k in contributing_strategies} g[k] >= threshold)
    return calculate_expected_indicator(
        pop_size, group_size, nb_strategies, stationary_distribution,
        [&](const std::vector<size_t> &group_config) -> double {
            size_t count = 0;
            for (const int64_t k : contributing_strategies) {
                count += group_config[static_cast<size_t>(k)];
            }
            return count >= static_cast<size_t>(threshold) ? 1.0 : 0.0;
        });
}

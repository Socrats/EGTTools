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

    // Hoist loop-invariant denominator: log C(pop_size, group_size) is constant for
    // every (state, group_config) pair in this function.
    const double log_denom = egttools::log_binomial_coefficient<double>(pop_size_sz, group_size_sz);

    // Precompute all group configurations once.  The inner sample_simplex call below
    // was previously executed nb_nonzero_states × nb_group_configs times; now it runs
    // only nb_group_configs times.
    std::vector<std::vector<size_t>> all_group_configs(
        nb_group_configs, std::vector<size_t>(nb_strategies_sz));
    for (int64_t i = 0; i < nb_group_configs; ++i) {
        egttools::FinitePopulations::sample_simplex(
            static_cast<size_t>(i), group_size_sz, nb_strategies_sz, all_group_configs[i]);
    }

    egttools::Vector result = egttools::Vector::Zero(nb_indicators);
    egttools::VectorXui state = egttools::VectorXui::Zero(static_cast<signed long>(nb_strategies));
    egttools::Vector probs = egttools::Vector::Zero(nb_group_configs);

    for (SparseMatIt it(stationary_distribution, 0); it; ++it) {
        egttools::FinitePopulations::sample_simplex(
            static_cast<size_t>(it.index()), pop_size_sz, nb_strategies_sz, state);

        for (int64_t i = 0; i < nb_group_configs; ++i) {
            probs(i) = egttools::multivariateHypergeometricPDF(
                log_denom, nb_strategies_sz, all_group_configs[i], state);
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

    // Hoist loop-invariant denominator.
    const double log_denom = egttools::log_binomial_coefficient<double>(pop_size_sz, group_size_sz);

    // Precompute all group configurations and their indicator values once,
    // so the inner sample_simplex and Python callable are each called only
    // nb_group_configs times instead of nb_nonzero_states × nb_group_configs times.
    std::vector<std::vector<size_t>> all_group_configs(
        nb_group_configs, std::vector<size_t>(nb_strategies_sz));
    std::vector<double> indicator_values(nb_group_configs);
    for (int64_t i = 0; i < nb_group_configs; ++i) {
        egttools::FinitePopulations::sample_simplex(
            static_cast<size_t>(i), group_size_sz, nb_strategies_sz, all_group_configs[i]);
        indicator_values[i] = indicator(all_group_configs[i]);
    }

    egttools::VectorXui state = egttools::VectorXui::Zero(static_cast<signed long>(nb_strategies));

    for (SparseMatIt it(stationary_distribution, 0); it; ++it) {
        egttools::FinitePopulations::sample_simplex(
            static_cast<size_t>(it.index()), pop_size_sz, nb_strategies_sz, state);

        double state_contrib = 0.0;
        for (int64_t i = 0; i < nb_group_configs; ++i) {
            const double prob = egttools::multivariateHypergeometricPDF(
                log_denom, nb_strategies_sz, all_group_configs[i], state);
            state_contrib += prob * indicator_values[i];
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

// ---------------------------------------------------------------------------
// State-level expected indicators  E[f] = Σ_s  sd(s) · f(s)
// ---------------------------------------------------------------------------

double egttools::utils::calculate_expected_state_indicator(
    const size_t pop_size, const size_t nb_strategies,
    SparseMatrix2D &stationary_distribution,
    const std::function<double(const std::vector<size_t> &)> &indicator) {
    double result = 0.0;
    std::vector<size_t> state_vec(nb_strategies, 0);
    egttools::VectorXui state = egttools::VectorXui::Zero(static_cast<signed long>(nb_strategies));

    for (SparseMatIt it(stationary_distribution, 0); it; ++it) {
        egttools::FinitePopulations::sample_simplex(
            static_cast<size_t>(it.index()), pop_size, nb_strategies, state);
        // Convert VectorXui to std::vector<size_t> for the Python-callable interface.
        for (size_t i = 0; i < nb_strategies; ++i)
            state_vec[i] = static_cast<size_t>(state(static_cast<signed long>(i)));
        result += it.value() * indicator(state_vec);
    }
    return result;
}

egttools::Vector egttools::utils::calculate_expected_state_indicators(
    const size_t pop_size, const size_t nb_strategies,
    SparseMatrix2D &stationary_distribution,
    const std::vector<std::function<double(const std::vector<size_t> &)>> &indicators) {
    const auto nb_indicators = static_cast<int64_t>(indicators.size());
    egttools::Vector result = egttools::Vector::Zero(nb_indicators);
    std::vector<size_t> state_vec(nb_strategies, 0);
    egttools::VectorXui state = egttools::VectorXui::Zero(static_cast<signed long>(nb_strategies));

    for (SparseMatIt it(stationary_distribution, 0); it; ++it) {
        egttools::FinitePopulations::sample_simplex(
            static_cast<size_t>(it.index()), pop_size, nb_strategies, state);
        for (size_t i = 0; i < nb_strategies; ++i)
            state_vec[i] = static_cast<size_t>(state(static_cast<signed long>(i)));
        for (int64_t k = 0; k < nb_indicators; ++k)
            result(k) += it.value() * indicators[static_cast<size_t>(k)](state_vec);
    }
    return result;
}

egttools::Vector egttools::utils::calculate_expected_state_indicators_precomputed(
    SparseMatrix2D &stationary_distribution,
    const Matrix2D &indicator_values) {
    // result[k] = Σ_s  sd(s) · indicator_values(s, k)
    // = sparse dot product of sd with each column of indicator_values.
    // We iterate over non-zero states; for each, add sd(s) * indicator_values.row(s).
    const int64_t nb_indicators = indicator_values.cols();
    egttools::Vector result = egttools::Vector::Zero(nb_indicators);

    for (SparseMatIt it(stationary_distribution, 0); it; ++it) {
        result.noalias() += it.value() * indicator_values.row(it.index()).transpose();
    }
    return result;
}

double egttools::utils::calculate_hypergeometric_expected_value(
    const size_t pop_size,
    const size_t group_size,
    const size_t nb_strategies,
    const Eigen::Ref<const VectorXui> &state,
    const Eigen::Ref<const Vector> &f_values) {
    // sum_{g} P(g | state) * f_values[g]
    // log C(pop_size, group_size) is hoisted once outside the group-config loop.
    const double log_denom = egttools::log_binomial_coefficient<double>(pop_size, group_size);
    const auto nb_group_configs = egttools::starsBars<int64_t>(
        static_cast<int64_t>(group_size), static_cast<int64_t>(nb_strategies));

    double result = 0.0;
    std::vector<size_t> group_config(nb_strategies, 0);
    for (int64_t g = 0; g < nb_group_configs; ++g) {
        egttools::FinitePopulations::sample_simplex(
            static_cast<size_t>(g), group_size, nb_strategies, group_config);
        result += static_cast<double>(f_values(g)) *
                  egttools::multivariateHypergeometricPDF(log_denom, nb_strategies, group_config, state);
    }
    return result;
}

double egttools::utils::calculate_hypergeometric_fitness(
    const int player_type,
    const size_t pop_size,
    const size_t group_size,
    const size_t nb_strategies,
    const Eigen::Ref<const VectorXui> &strategies,
    const Eigen::Ref<const Vector> &payoffs_row) {
    // fitness = sum_{g: g[player_type] > 0} payoffs_row[g] * P(g_reduced | strategies)
    // where g_reduced has one fewer focal player and P uses (pop_size-1, group_size-1).
    const double log_denom = egttools::log_binomial_coefficient<double>(pop_size - 1, group_size - 1);
    const auto nb_group_configs = egttools::starsBars<int64_t>(
        static_cast<int64_t>(group_size), static_cast<int64_t>(nb_strategies));

    double fitness = 0.0;
    const auto pt = static_cast<size_t>(player_type);
    std::vector<size_t> sample_counts(nb_strategies, 0);
    for (int64_t i = 0; i < nb_group_configs; ++i) {
        egttools::FinitePopulations::sample_simplex(
            static_cast<size_t>(i), group_size, nb_strategies, sample_counts);
        if (sample_counts[pt] > 0) {
            sample_counts[pt] -= 1;
            fitness += static_cast<double>(payoffs_row(i)) *
                       egttools::multivariateHypergeometricPDF(log_denom, nb_strategies, sample_counts, strategies);
            sample_counts[pt] += 1;
        }
    }
    return fitness;
}

egttools::Matrix2D egttools::utils::precompute_group_to_state_indicator_matrix(
    const int64_t pop_size, const int64_t group_size, const int64_t nb_strategies,
    const std::vector<std::function<double(const std::vector<size_t> &)>> &indicators) {
    // Returns indicator_values of shape (nb_states, nb_indicators), where:
    //   indicator_values(s, k) = Σ_g  P(g | s) · f_k(g)
    //
    // Phase 1: evaluate every indicator on every group configuration once.
    //          O(nb_group_configs × nb_indicators) Python calls.
    // Phase 2: for each population state, compute the hypergeometric-weighted sum.
    //          Pure C++, no Python calls.

    const auto nb_indicators = static_cast<int64_t>(indicators.size());
    const int64_t nb_group_configs = egttools::starsBars<int64_t>(group_size, nb_strategies);
    const int64_t nb_states = egttools::starsBars<int64_t>(pop_size, nb_strategies);
    const auto nb_strategies_sz = static_cast<size_t>(nb_strategies);
    const auto pop_size_sz = static_cast<size_t>(pop_size);
    const auto group_size_sz = static_cast<size_t>(group_size);

    // Phase 1: decode every group configuration once and evaluate all indicators.
    // We store the decoded configs so Phase 2 can reuse them without re-decoding.
    std::vector<std::vector<size_t>> all_group_configs(
        nb_group_configs, std::vector<size_t>(nb_strategies_sz));
    Matrix2D group_indicator_values = Matrix2D::Zero(nb_group_configs, nb_indicators);
    for (int64_t g = 0; g < nb_group_configs; ++g) {
        egttools::FinitePopulations::sample_simplex(
            static_cast<size_t>(g), group_size_sz, nb_strategies_sz, all_group_configs[g]);
        for (int64_t k = 0; k < nb_indicators; ++k)
            group_indicator_values(g, k) = indicators[static_cast<size_t>(k)](all_group_configs[g]);
    }

    // Phase 2: for each state s, compute indicator_values.row(s) = probs(s)^T * group_indicator_values.
    // The log-denominator log C(pop_size, group_size) is loop-invariant for the general path.
    Matrix2D indicator_values = Matrix2D::Zero(nb_states, nb_indicators);
    egttools::VectorXui state = egttools::VectorXui::Zero(static_cast<signed long>(nb_strategies));
    egttools::Vector probs = egttools::Vector::Zero(nb_group_configs);

    if (group_size == 2) {
        // Fast path for pairwise groups: simple combinatorial probabilities,
        // no log/exp required.
        //   P(same-strategy pair i,i | s) = s_i*(s_i-1) / (Z*(Z-1))
        //   P(mixed pair i,j | s)         = 2*s_i*s_j  / (Z*(Z-1))
        const double denom = static_cast<double>(pop_size_sz * (pop_size_sz - 1));
        for (int64_t s = 0; s < nb_states; ++s) {
            egttools::FinitePopulations::sample_simplex(
                static_cast<size_t>(s), pop_size_sz, nb_strategies_sz, state);
            probs.setZero();
            for (int64_t g = 0; g < nb_group_configs; ++g) {
                const auto &gc = all_group_configs[g];
                // Determine if this is a same-strategy or mixed pair.
                int64_t nonzero_count = 0;
                for (size_t i = 0; i < nb_strategies_sz; ++i)
                    if (gc[i] > 0) ++nonzero_count;
                if (nonzero_count == 1) {
                    // Same-strategy pair
                    for (size_t i = 0; i < nb_strategies_sz; ++i) {
                        if (gc[i] == 2) {
                            const double si = static_cast<double>(state(static_cast<signed long>(i)));
                            probs(g) = si * (si - 1.0) / denom;
                            break;
                        }
                    }
                } else {
                    // Mixed pair: locate the two strategy indices
                    size_t idx_a = 0, idx_b = 0;
                    bool found_a = false;
                    for (size_t i = 0; i < nb_strategies_sz; ++i) {
                        if (gc[i] == 1) {
                            if (!found_a) { idx_a = i; found_a = true; }
                            else { idx_b = i; break; }
                        }
                    }
                    const double sa = static_cast<double>(state(static_cast<signed long>(idx_a)));
                    const double sb = static_cast<double>(state(static_cast<signed long>(idx_b)));
                    probs(g) = 2.0 * sa * sb / denom;
                }
            }
            indicator_values.row(s).noalias() = probs.transpose() * group_indicator_values;
        }
    } else {
        // General path: multivariateHypergeometricPDF with precomputed denominator.
        // group configs were decoded in Phase 1 and are reused here.
        const double log_denom = egttools::log_binomial_coefficient<double>(pop_size_sz, group_size_sz);
        for (int64_t s = 0; s < nb_states; ++s) {
            egttools::FinitePopulations::sample_simplex(
                static_cast<size_t>(s), pop_size_sz, nb_strategies_sz, state);
            for (int64_t g = 0; g < nb_group_configs; ++g) {
                probs(g) = egttools::multivariateHypergeometricPDF(
                    log_denom, nb_strategies_sz, all_group_configs[g], state);
            }
            indicator_values.row(s).noalias() = probs.transpose() * group_indicator_values;
        }
    }
    return indicator_values;
}

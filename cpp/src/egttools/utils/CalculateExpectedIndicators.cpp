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

#include <egttools/utils/CalculateExpectedIndicators.h>

egttools::Vector egttools::utils::calculate_strategies_distribution(size_t pop_size,
                                                                    size_t nb_strategies,
                                                                    SparseMatrix2D& stationary_distribution) {
    egttools::Vector strategy_distribution = egttools::Vector::Zero(static_cast<signed long>(nb_strategies));
    egttools::VectorXui state = egttools::VectorXui::Zero(static_cast<signed long>(nb_strategies));

    for (SparseMatIt it(stationary_distribution, 0); it; ++it) {
        egttools::FinitePopulations::sample_simplex(it.index(), pop_size, nb_strategies, state);
        strategy_distribution += (state.cast<double>() / pop_size) * it.value();
    }

    return strategy_distribution;
}

double egttools::utils::calculate_expected_payoff(int64_t pop_size, int64_t group_size, int64_t nb_strategies, SparseMatrix2D& stationary_distribution, Matrix2D& payoff_matrix) {
    // This function calculates the expected payoff of a population
    // This is: E[Payoff] = sum_for_all_states(P(state) *
    //                          (sum_for_all_group_configurations(P(group_config) * avg_payoff_group_config))
    double expected_payoff = 0;
    auto nb_group_configurations = egttools::starsBars<int64_t>(group_size, nb_strategies);

    egttools::VectorXui state = egttools::VectorXui::Zero(static_cast<signed long>(nb_strategies));
    std::vector<size_t> group_configuration(nb_strategies, 0);

    for (SparseMatIt it(stationary_distribution, 0); it; ++it) {
        egttools::FinitePopulations::sample_simplex(it.index(), pop_size, nb_strategies, state);

        double expected_payoff_state = 0;

        for (int64_t i = 0; i < nb_group_configurations; ++i) {
            // Update strategy counts based on the current state
            egttools::FinitePopulations::sample_simplex(i, group_size, nb_strategies, group_configuration);

            // Calculate probability of encountering the current group
            auto prob = egttools::multivariateHypergeometricPDF(pop_size, nb_strategies, group_size,
                                                                group_configuration,
                                                                state);

            expected_payoff_state += prob * payoff_matrix.col(i).mean();
        }

        expected_payoff += expected_payoff_state * it.value();
    }
    return expected_payoff;
}
//
//void egttools::utils::calculate_strategies_distribution(size_t pop_size, size_t nb_strategies,
//                                                        egttools::SparseMatrix2D& stationary_distribution,
//                                                        egttools::Vector& strategy_distribution) {
//    strategy_distribution.setZero();
//    egttools::VectorXui state = egttools::VectorXui::Zero(static_cast<signed long>(nb_strategies));
//    //#pragma omp simd
//    for (SparseMatIt it(stationary_distribution, 0); it; ++it) {
//        egttools::FinitePopulations::sample_simplex(it.index(), pop_size, nb_strategies, state);
//        strategy_distribution += (state.cast<double>() / pop_size) * it.value();
//    }
//}
//
//void egttools::utils::calculate_strategies_distribution(size_t pop_size, size_t nb_strategies,
//                                                        egttools::SparseMatrix2D& stationary_distribution,
//                                                        egttools::Vector& strategy_distribution,
//                                                        egttools::VectorXui& state) {
//    strategy_distribution.setZero();
//    state.setZero();
//    //#pragma omp simd
//    for (SparseMatIt it(stationary_distribution, 0); it; ++it) {
//        egttools::FinitePopulations::sample_simplex(it.index(), pop_size, nb_strategies, state);
//        strategy_distribution += (state.cast<double>() / pop_size) * it.value();
//    }
//}
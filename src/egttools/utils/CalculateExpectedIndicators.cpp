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
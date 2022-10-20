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

#include <egttools/Types.h>

#include <egttools/finite_populations/Utils.hpp>

//#if defined(_OPENMP)
//#include <egttools/OpenMPUtils.hpp>
//#endif

namespace egttools::utils {

    using SparseMatIt = SparseMatrix2D::InnerIterator;

    /**
     * @brief Calculates the average frequency of each strategy available in the population given the stationary distribution.
     *
     * @param pop_size : size of the population.
     * @param nb_strategies : number of strategies.
     * @param stationary_distribution : an Eigen SparseMatrix containing the frequency of each state in the system.
     * @return An Eigen::Vector containing the frequency of each strategy.
     */
    Vector calculate_strategies_distribution(size_t pop_size, size_t nb_strategies,
                                             SparseMatrix2D& stationary_distribution);

    double calculate_expected_payoff(int64_t pop_size, int64_t group_size, int64_t nb_strategies, SparseMatrix2D& stationary_distribution, Matrix2D& payoff_matrix);

//    /**
//      * @brief Calculates the average frequency of each strategy available in the population given the stationary distribution.
//      *
//      * @param pop_size : size of the population.
//      * @param nb_strategies : number of strategies.
//      * @param stationary_distribution : an Eigen SparseMatrix containing the frequency of each state in the system.
//      * @param strategy_distribution : a container Eigen::Vector which is used to sture the average frequency of each strategy.
//      * @return
//      */
//    void calculate_strategies_distribution(size_t pop_size, size_t nb_strategies,
//                                           SparseMatrix2D& stationary_distribution, Vector& strategy_distribution);
//
//    /**
//     * @brief Calculates the average frequency of each strategy available in the population given the stationary distribution.
//     *
//     * @param pop_size : size of the population.
//     * @param nb_strategies : number of strategies.
//     * @param stationary_distribution : an Eigen SparseMatrix containing the frequency of each state in the system
//     * @param strategy_distribution : a container Eigen::Vector which is used to sture the average frequency of each strategy.
//     * @param state : a container Eigen::Vector of size nb_strategies which is used to store the counts of each strategy in a given
//     *                population state.
//     * @return
//     */
//    void calculate_strategies_distribution(size_t pop_size, size_t nb_strategies,
//                                           SparseMatrix2D& stationary_distribution,
//                                           Vector& strategy_distribution, VectorXui& state);
}// namespace egttools::utils


#endif//EGTTOOLS_UTILS_CALCULATEEXPECTEDINDICATORS_H

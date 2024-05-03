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
#include <stdexcept>

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

    /**
     * @brief Calculates the expected value of an indicator in the population.
     *
     * This method will calculate the expected indicator given the stationary distribution of the population.
     * For that, this function will calculate the average indicator at any given population state and then
     * multiply it for the probability of the state (given by the stationary distribution).
     *
     * To calculate the average indicator at a given population state, this method needs to calculate the
     * cooperation rate for any possible group configuration and multiply it by the likelihood of the group
     * occurring at the given population state.
     *
     * The cooperation rate for a given group composition is given by the expected_indicator_matrix
     *
     *
     *
     * basically I need to be able to calculate on average the cooperation rate for every prevalent strategy.
     * There are mostly two scenarios:
     *
     * 1) two-player games ->
     *
     * @param pop_size
     * @param nb_strategies
     * @param stationary_distribution
     * @param expected_indicator_matrix
     * @return the expected indicator given the stationary distribution.
     */

    double calculate_expected_indicator(int64_t pop_size, int64_t nb_strategies,
                                        SparseMatrix2D& stationary_distribution,
                                        Matrix2D& expected_indicator_matrix);

//    double calculate_expected_indicator(int64_t pop_size, int64_t nb_strategies,
//                                        int64_t group_size,
//                                        SparseMatrix2D& stationary_distribution,
//                                        Matrix2D& expected_indicator_matrix);

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

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

#ifndef EGTTOOLS_FINITEPOPULATIONS_UTILS_HPP
#define EGTTOOLS_FINITEPOPULATIONS_UTILS_HPP

#include <egttools/Distributions.h>
#include <egttools/Types.h>

#include <cmath>
#include <limits>
#include <vector>

namespace egttools::FinitePopulations {
    using GroupPayoffs = egttools::Matrix2D;
    using StrategyCounts = std::vector<size_t>;

    /**
     * @brief returns the imitation probability calculated according to the fermi function.
     *
     * @param beta intensity of selection
     * @param a fitness of player A
     * @param b fitness fo player B
     * @return probability of imitation
     */
    double fermi(double beta, double a, double b);

    /**
     * @brief contest success function that compares 2 payoffs according to a payoff importance z
     *
     * This function must never be called with z = 0. This would produce a zero division error.
     * And the behaviour might be undefined.
     *
     * @param z : importance of the payoff
     * @param a : expected payoff a
     * @param b : expected payoff b
     * @return probability of a winning over b
     */
    double contest_success(double z, double a, double b);

    /**
     * @brief contest success function that compares 2 payoffs according to a payoff importance z
     *
     * This function should be used when z = 0
     *
     * @param a : expected payoff a
     * @param b : expected payoff b
     * @return probability of a winning over b
     */
    double contest_success(double a, double b);

    /**
    * @brief This function converts a vector containing counts into an index.
    *
    * This method was copied from @ebargiac
    *
    * @param group_size maximum bin size (it can also be the population size)
    * @param current_group The vector to convert.
    *
    * @return The unique index in [0, starsBars(history, group_size - 1)) representing the n-dimensional simplex.
    */
    size_t calculate_state(const size_t &group_size, const egttools::Factors &current_group);

    /**
    * @brief This function converts a vector containing counts into an index.
    *
    * This method was copied from @ebargiac
    *
    * @param group_size The sum of the values contained in data.
    * @param current_group The vector to convert.
    *
    * @return The unique index in [0, starsBars(history, data.size() - 1)) representing data.
    */
    size_t calculate_state(const size_t &group_size, const Eigen::Ref<const egttools::VectorXui> &current_group);

    /**
     * @brief Transforms and state index into a vector.
     *
     * @param i : state index
     * @param pop_size : size of the population
     * @param nb_strategies : number of strategies
     * @return vector with the sampled state
     */
    VectorXui sample_simplex(size_t i, const size_t &pop_size, const size_t &nb_strategies);

    /**
     * @brief Transforms and state index into a vector.
     *
     * @param i : state index
     * @param pop_size : size of the population
     * @param nb_strategies : number of strategies
     * @param state : container for the sampled state
     */
    void sample_simplex(size_t i, const size_t &pop_size, const size_t &nb_strategies, VectorXui &state);

    void
    sample_simplex(size_t i, const size_t &pop_size, const size_t &nb_strategies, std::vector<size_t> &state);

    template<typename G>
    void sample_simplex(size_t nb_strategies, Vector &state, std::uniform_real_distribution<double> prob_dist,
                        G &generator) {
        for (size_t i = 0; i < nb_strategies; ++i) {
            state(i) = -std::log(prob_dist(generator));
        }
        state.array() /= state.sum();
        assert(state.sum() == 1.0);
    }

    /**
     * @brief Defines the numeric limit of floating points
     */
    constexpr double_t doubleEpsilon = std::numeric_limits<double>::digits10;
}// namespace egttools::FinitePopulations

#endif//DYRWIN_FINITEPOPULATIONS_UTILS_HPP

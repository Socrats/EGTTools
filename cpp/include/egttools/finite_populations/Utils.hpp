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
#ifndef EGTTOOLS_FINITEPOPULATIONS_UTILS_HPP
#define EGTTOOLS_FINITEPOPULATIONS_UTILS_HPP

#include <egttools/Distributions.h>
#include <egttools/Sampling.h>
#include <egttools/Types.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <unordered_set>
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

    /**
     * @brief Transforms and state index into a vector.
     *
     * @param i : state index
     * @param pop_size : size of the population
     * @param nb_strategies : number of strategies
     * @param state : container for the sampled state
     */
    void
    sample_simplex(size_t i, const size_t &pop_size, const size_t &nb_strategies, std::vector<size_t> &state);

    /**
     * @brief Samples the unit n-dimensional simplex randomly uniform.
     * @tparam SizeType : type of size variables
     * @tparam G
     * @param nb_strategies
     * @param state
     * @param prob_dist
     * @param generator
     */
    template<typename SizeType, typename G>
    void sample_unit_simplex(SizeType nb_strategies, Vector &state, std::uniform_real_distribution<double> prob_dist,
                             G &generator) {
        for (int i = 0; i < static_cast<int>(nb_strategies); ++i) {
            state(i) = -std::log(prob_dist(generator));
        }
        state.array() /= state.sum();
        assert(state.sum() == 1.0);
    }

    /**
     * @brief Samples uniformly a point in the simplex.
     *
     * This algorithm has been proposed in "Sampling Uniformly the Unit
     * Simplex", from Noah A. Smith and Roy W. Tromble.
     *
     * Sample x_1, ..., x_{n−1}
     * uniformly at random from {0, 2, ..., M − 1}
     * without replacement (i.e., choose n−1 distinct values).
     * Let x_0 = 0, xn = M and order the sampled values.
     * Let y_i = x_i−x_{i−1}, ∀i∈{1,2,...,n}.
     *
     * To get a point in the unit simplex it is enough to divide the vector
     * produced by pop_size.
     *
     * @tparam SizeType : Type of the of values
     * @tparam OutputVector : Type of vector which will contain the
     *                        n-dimensional point in the simplex
     * @tparam G : Type of the random generator
     * @param nb_strategies : number of strategies
     * @param pop_size : population size
     * @param state : vector container for the state (point in the n-dimensional simplex)
     * @param generator : random generator
     */
    template<typename SizeType, typename SampleType, typename OutputVector, typename G>
    void sample_simplex_direct_method(SizeType nb_strategies, SizeType pop_size, OutputVector &state,
                                      G &generator) {
        std::vector<SampleType> samples(nb_strategies - 1);
        std::unordered_set<SampleType> container(nb_strategies - 1);
        // sample without replacement nb_strategies - 1 elements
        egttools::sampling::ordered_sample_without_replacement<SizeType, SampleType, G>(0, pop_size, nb_strategies - 1,
                                                                                        samples, container, generator);
        state(0) = samples[0];
        for (SizeType i = 1; i < nb_strategies - 1; ++i) {
            state(i) = samples[i] - samples[i - 1];
        }
        state(nb_strategies - 1) = pop_size - samples[nb_strategies - 2];
    }

    /**
     * @brief Defines the numeric limit of floating points
     */
    constexpr double_t doubleEpsilon = std::numeric_limits<double>::digits10;
}// namespace egttools::FinitePopulations

#endif//EGTTOOLS_FINITEPOPULATIONS_UTILS_HPP

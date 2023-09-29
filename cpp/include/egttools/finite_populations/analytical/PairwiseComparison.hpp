/** Copyright (c) 2019-2022  Elias Fernandez
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
#ifndef EGTTOOLS_FINITEPOPULATIONS_ANALYTICAL_PAIRWISECOMPARISON_HPP
#define EGTTOOLS_FINITEPOPULATIONS_ANALYTICAL_PAIRWISECOMPARISON_HPP

#include <egttools/Distributions.h>
#include <egttools/SeedGenerator.h>
#include <egttools/Types.h>

#include <cmath>
#include <egttools/LruCache.hpp>
#include <egttools/finite_populations/Utils.hpp>
#include <egttools/finite_populations/games/AbstractGame.hpp>
#include <egttools/finite_populations/games/Matrix2PlayerGameHolder.hpp>
#include <egttools/finite_populations/games/MatrixNPlayerGameHolder.hpp>
#include <stdexcept>
#include <tuple>

#if (HAS_BOOST)
#include <boost/multiprecision/cpp_dec_float.hpp>
#endif

#if defined(_OPENMP)
#include <egttools/OpenMPUtils.hpp>
#endif


namespace egttools::FinitePopulations::analytical {
#if (HAS_BOOST)
    using cpp_dec_float_100 = boost::multiprecision::cpp_dec_float_100;
#endif
    using Cache = egttools::Utils::LRUCache<std::string, double>;

    /**
     * @brief Provides analytical methods to study evolutionary dynamics in finite populations
     * with the Pairwise Comparison rule.
     */
    class PairwiseComparison {
    public:
        /**
         * @brief Implements methods to study evolutionary dynamics in finite populations with the
         * Pairwise Comparison rule.
         *
         * This class implements a series of analytical methods to calculate the most relevant indicators
         * used to study the evolutionary dynamics in finite populations with the Pairwise Comparison
         * rule.
         *
         * This class requires a @param population_size to indicate the size of the population in which
         * the evolutionary process takes place, as well as a @param game which must be an object
         * inheriting from `egttools.games.AbstractGame`, and which contains a method to calculate
         * the fitness of a strategy, given a population state (represented as the counts of each
         * strategy in the population).
         *
         * @note For now it is not possible not possible to update the game without instantiating
         * PairwiseComparison again. Hopefully, this will be fixed in the future
         *
         * @param population_size : size of the population
         * @param game : Game object.
         */
        PairwiseComparison(int population_size, egttools::FinitePopulations::AbstractGame &game);

        PairwiseComparison(int population_size, egttools::FinitePopulations::AbstractGame &game, size_t cache_size);

        ~PairwiseComparison() = default;

        void pre_calculate_edge_fitnesses();

        /**
         * @brief computes the transition matrix of the Markov Chain which defines the population dynamics.
         *
         * It is not advisable to use this method for very large state spaces since the memory required
         * to store the matrix might explode. In these cases you should resort to dimensional reduction
         * techniques, such as the Small Mutation Limit (SML).
         *
         * @param beta : intensity of selection
         * @param mu : mutation rate
         * @return SparseMatrix2D containing the transition probabilities from any population state to another.
         * This matrix will be of size nb_states x nb_states.
         */
        SparseMatrix2D calculate_transition_matrix(double beta, double mu);

        /**
         * @brief Calculates the gradient of selection without mutation for the given state.
         *
         * This method calculates the gradient of selection (without mutation), which is, the
         * most likely direction of evolution of the system.
         *
         * @param beta : intensity of selection
         * @param state : VectorXui containing the counts of each strategy in the population
         * @return Vector of nb_strategies dimensions containing the gradient of selection.
         */
        Vector calculate_gradient_of_selection(double beta, const Eigen::Ref<const VectorXui> &state);

        /**
         * @brief Calculates the fixation probability of an invading strategy in a population o resident strategy.
         *
         * @param index_invading_strategy : index of the invading strategy
         * @param index_resident_strategy : index of the resident strategy
         * @param beta : intensity of selection
         * @return fixation probability
         */
        double calculate_fixation_probability(int index_invading_strategy, int index_resident_strategy, double beta);

        /**
         * @brief Calculates the transition matrix of the reduced Markov Chain that emerges when assuming SML.
         *
         * By assuming the limit of small mutations (SML), we can reduce the number of states of the dynamical system
         * to those which are monomorphic, i.e., the whole population adopts the same strategy.
         *
         * Thus, the dimensions of the transition matrix in the SML is (nb_strategies, nb_strategies), and
         * the transitions are given by the normalized fixation probabilities. This means that a transition
         * where i \neq j, T[i, j] = fixation(i, j) / (nb_strategies - 1) and T[i, i] = 1 - \sum{T[i, j]}.
         *
         * This method will also return the matrix of fixation probabilities,
         * where fixation_probabilities[i, j] gives the probability that one mutant j fixates in a population
         * of i.
         *
         *
         * @param beta : intensity of selection
         * @return std::tuple<Matrix2D, Matrix2D> A tuple including the transition matrix
         *         and a matrix with the fixation probabilities.
         */
        std::tuple<Matrix2D, Matrix2D> calculate_transition_and_fixation_matrix_sml(double beta);

        //        Vector calculate_gradient_of_selection(const Eigen::Ref<const Matrix2D> &transition_matrix,
        //                                               const Eigen::Ref<const Vector> &stationary_distribution,
        //                                               const Eigen::Ref<const VectorXui> &state);

        // setters
        void update_population_size(int population_size);

        // getters
        [[nodiscard]] int nb_strategies() const;
        [[nodiscard]] int64_t nb_states() const;
        [[nodiscard]] int population_size() const;
        [[nodiscard]] const egttools::FinitePopulations::AbstractGame &game() const;

    private:
        int population_size_, nb_strategies_;
        size_t cache_size_;
        int64_t nb_states_;
        egttools::FinitePopulations::AbstractGame &game_;

        Cache cache_;

        /**
         * @brief calculates a transition probability.
         *
         * This method calculates the transition probability from the current @param state
         * to a new state containing one more @param increasing_strategy and one less
         * @param decreasing_strategy.
         *
         * @param decreasing_strategy : index of the strategy that will decrease
         * @param increasing_strategy : index of the strategy that will increase
         * @param beta : intensity of selection
         * @param state : Vector containing the counts of the strategies in the population
         * @return the transition probability
         */
        //        inline double calculate_transition_(int decreasing_strategy, int increasing_strategy, double beta, double mu, VectorXui &state);

        inline double calculate_local_gradient_(int decreasing_strategy, int increasing_strategy, double beta, VectorXui &state);

        inline double calculate_fitness_(int &strategy_index, VectorXui &state);
    };
}// namespace egttools::FinitePopulations::analytical

#endif//EGTTOOLS_FINITEPOPULATIONS_ANALYTICAL_PAIRWISECOMPARISON_HPP

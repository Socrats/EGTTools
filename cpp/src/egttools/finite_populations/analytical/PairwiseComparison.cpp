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

#include <egttools/finite_populations/analytical/PairwiseComparison.hpp>

egttools::FinitePopulations::analytical::PairwiseComparison::PairwiseComparison(int population_size,
                                                                                egttools::FinitePopulations::AbstractGame &game) : population_size_(population_size),
                                                                                                                                   game_(game) {
    if (population_size <= 0) {
        throw std::invalid_argument(
                "The size of the population must be a positive integer");
    }

    nb_strategies_ = static_cast<int>(game.nb_strategies());
    nb_states_ = egttools::starsBars(population_size_, nb_strategies_);
}

egttools::SparseMatrix2D egttools::FinitePopulations::analytical::PairwiseComparison::calculate_transition_matrix(double beta, double mu) {
    // Check if the size of the population is positive
    if (beta < 0) {
        throw std::invalid_argument(
                "The intensity of selection beta must not be negative!");
    }
    // First we initialise the container for the stationary distribution
    auto transition_matrix = SparseMatrix2D(nb_states_, nb_states_);
    // Then we sample a random population state
    VectorXui current_state = VectorXui::Zero(nb_strategies_);
    VectorXui new_state(current_state);

    double mutation_probability = mu / (nb_strategies_ - 1);

    for (int64_t current_state_index = 0; current_state_index < nb_states_; ++current_state_index) {
        double total_probability = 0.;

        // get current state
        egttools::FinitePopulations::sample_simplex(current_state_index, population_size_, nb_strategies_, current_state);

        // copy current state
        new_state = current_state;

        // First check if we are in a monomorphic state
        // If that is the case, the probability of remaining in the current
        // state is 1 - mu, and the probability of moving to any other
        // adjacent state is mu/(nb_strategies - 1)
        bool homomorphic = false;
        for (int i = 0; i < nb_strategies_; ++i) {
            if (current_state(i) == static_cast<size_t>(population_size_)) {
                homomorphic = true;
                new_state(i) -= 1;
                break;
            }
        }

        // We need now to look at all possible transitions
        // That is all possible changes of 1 individual in the population
        // This excludes strategies that are not currently present in the population
        // (their count is zero)
        for (int i = 0; i < nb_strategies_; ++i) {
            // This will happen only when we are in a monomorphic state
            if (current_state(i) == static_cast<size_t>(population_size_))
                continue;

            new_state(i) += 1;

            if (homomorphic) {
                auto new_state_index = egttools::FinitePopulations::calculate_state(population_size_,
                                                                                    new_state);
                // update transition matrix
                transition_matrix.coeffRef(current_state_index, static_cast<int64_t>(new_state_index)) = mutation_probability;
            } else {
                // calculate fitness of the increasing strategy
                current_state(i) -= 1;
                auto fitness_increase = game_.calculate_fitness(i,
                                                                population_size_,
                                                                current_state);
                current_state(i) += 1;

                for (int j = 0; j < nb_strategies_; ++j) {
                    // if the strategy to decrease already has count 0
                    // continue, or if we are tying to increase and decrease the same strategy
                    if ((i == j) || (current_state(i) == 0))
                        continue;

                    new_state(j) -= 1;
                    auto new_state_index = egttools::FinitePopulations::calculate_state(population_size_,
                                                                                        new_state);

                    // calculate fitness of the decreasing strategy
                    current_state(j) -= 1;
                    auto fitness_decrease = game_.calculate_fitness(j, population_size_, current_state);
                    current_state(j) += 1;

                    double transition_probability = (1 - mu) * (static_cast<double>(current_state(i)) / (population_size_ - 1));
                    transition_probability *= egttools::FinitePopulations::fermi(beta, fitness_decrease, fitness_increase);
                    transition_probability = (static_cast<double>(current_state(j)) / population_size_) * (transition_probability + mutation_probability);

                    // update transition matrix
                    transition_matrix.coeffRef(current_state_index, static_cast<int64_t>(new_state_index)) = transition_probability;

                    total_probability += transition_probability;
                    new_state(j) += 1;
                }
            }
            new_state(i) -= 1;
        }
        // update transition matrix with probability of remaining in the current state
        if (homomorphic) {
            transition_matrix.coeffRef(current_state_index, current_state_index) = (1 - mu);
        } else {
            transition_matrix.coeffRef(current_state_index, current_state_index) = 1 - total_probability;
        }
    }

    return transition_matrix;
}

//double egttools::FinitePopulations::analytical::PairwiseComparison::calculate_gradient_of_selection(double beta, egttools::VectorXui state) {
//    egttools::Vector probability_selecting_strategy_to_die = state.cast<double>() / population_size_;
//    egttools::Vector probability_selecting_strategy_to_reproduce = state.cast<double>() / (population_size_ - 1);
//
//
//}

void egttools::FinitePopulations::analytical::PairwiseComparison::update_population_size(int population_size) {
    // Check if the size of the population is positive
    if (population_size <= 0) {
        throw std::invalid_argument(
                "The size of the population must be a positive integer");
    }

    population_size_ = population_size;
    nb_states_ = egttools::starsBars(population_size_, nb_strategies_);
}

void egttools::FinitePopulations::analytical::PairwiseComparison::update_game(egttools::FinitePopulations::AbstractGame &game) {
    game_ = game;
    nb_strategies_ = static_cast<int>(game.nb_strategies());
    nb_states_ = egttools::starsBars(population_size_, nb_strategies_);
}

int egttools::FinitePopulations::analytical::PairwiseComparison::nb_strategies() const {
    return nb_strategies_;
}

int64_t egttools::FinitePopulations::analytical::PairwiseComparison::nb_states() const {
    return nb_states_;
}

int egttools::FinitePopulations::analytical::PairwiseComparison::population_size() const {
    return population_size_;
}

const egttools::FinitePopulations::AbstractGame &egttools::FinitePopulations::analytical::PairwiseComparison::game() const {
    return game_;
}
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
                                                                                                                                   cache_size_(100),
                                                                                                                                   game_(game),
                                                                                                                                   cache_(cache_size_) {
    if (population_size <= 0) {
        throw std::invalid_argument(
                "The size of the population must be a positive integer");
    }

    nb_strategies_ = static_cast<int>(game.nb_strategies());
    nb_states_ = egttools::starsBars(population_size_, nb_strategies_);
}

egttools::FinitePopulations::analytical::PairwiseComparison::PairwiseComparison(int population_size,
                                                                                egttools::FinitePopulations::AbstractGame &game, size_t cache_size) : population_size_(population_size),
                                                                                                                                                      cache_size_(cache_size),
                                                                                                                                                      game_(game),
                                                                                                                                                      cache_(cache_size) {
    if (population_size <= 0) {
        throw std::invalid_argument(
                "The size of the population must be a positive integer");
    }

    nb_strategies_ = static_cast<int>(game.nb_strategies());
    nb_states_ = egttools::starsBars(population_size_, nb_strategies_);
}

void egttools::FinitePopulations::analytical::PairwiseComparison::pre_calculate_edge_fitnesses() {
    Matrix2D fitnesses = Matrix2D::Zero(nb_strategies_, (population_size_ - 1) * nb_strategies_);
    int nb_elements = population_size_ - 2;

#pragma omp parallel for default(none) shared(fitnesses, nb_strategies_, population_size_, game_, nb_elements)
    for (int i = 0; i < nb_strategies_; ++i) {
        VectorXui population_state = VectorXui::Zero(nb_strategies_);
        for (int j = i; j < nb_strategies_; ++j) {
            for (int z = 1; z < population_size_; ++z) {
                population_state(i) = z;
                population_state(j) = population_size_ - z;

                // calculate fitness of invading strategy
                population_state(i) -= 1;
                fitnesses(i, j * nb_elements + (z - 1)) = game_.calculate_fitness(i, population_size_, population_state);
                population_state(i) += 1;

                population_state(j) -= 1;
                fitnesses(j, i * nb_elements + (population_size_ - z - 1)) = game_.calculate_fitness(j, population_size_, population_state);
            }
            population_state(j) = 0;
        }
    }

    std::string key1, key2;

    // Now we add the to cache
    VectorXui population_state = VectorXui::Zero(nb_strategies_);
    for (int i = 0; i < nb_strategies_; ++i) {
        for (int j = i; j < nb_strategies_; ++j) {
            for (int z = 1; z < population_size_; ++z) {
                population_state(i) = z;
                population_state(j) = population_size_ - z;
                // add fitness value to cache
                std::stringstream result;
                result << population_state;
                key1 = std::to_string(i) + result.str();
                key2 = std::to_string(j) + result.str();

                cache_.insert(key1, fitnesses(i, j * nb_elements + (z - 1)));
                cache_.insert(key2, fitnesses(j, i * nb_elements + (population_size_ - z - 1)));
            }
            population_state(j) = 0;
        }
        population_state(i) = 0;
    }
}

egttools::SparseMatrix2D egttools::FinitePopulations::analytical::PairwiseComparison::calculate_transition_matrix(double beta, double mu) {
    // Check if beta is positive
    if (beta < 0) {
        throw std::invalid_argument(
                "The intensity of selection beta must not be negative!");
    }
    // First we initialise the container for the stationary distribution
    auto transition_matrix = SparseMatrix2D(nb_states_, nb_states_);
    // Then we sample a random population state
    VectorXui current_state = VectorXui::Zero(nb_strategies_);
    VectorXui new_state(current_state);

    double not_mu = 1. - mu;
    double mutation_probability;
    if (nb_strategies_ > 2) {
        mutation_probability = mu / (nb_strategies_ - 1);
    } else {
        mutation_probability = mu;
    }

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
            if (current_state(i) == static_cast<size_t>(population_size_)) continue;

            new_state(i) += 1;

            if (homomorphic) {
                auto new_state_index = egttools::FinitePopulations::calculate_state(population_size_, new_state);
                // update transition matrix
                transition_matrix.coeffRef(current_state_index, static_cast<int64_t>(new_state_index)) = mutation_probability;
            } else {
                // If the increasing strategy has currently 0 count, it can only
                // increase through mutation
                if (current_state(i) == 0) {
                    for (int j = 0; j < nb_strategies_; ++j) {
                        // if the strategy to decrease already has count 0
                        // continue, or if we are tying to increase and decrease the same strategy
                        if ((i == j) || (current_state(j) == 0))
                            continue;

                        new_state(j) -= 1;
                        auto new_state_index = egttools::FinitePopulations::calculate_state(population_size_, new_state);

                        // calculate transition probability
                        double transition_probability = (static_cast<double>(current_state(j)) / population_size_) * mutation_probability;

                        // update transition matrix
                        transition_matrix.coeffRef(current_state_index, static_cast<int64_t>(new_state_index)) = transition_probability;

                        total_probability += transition_probability;
                        new_state(j) += 1;
                    }
                } else {
                    // calculate fitness of the increasing strategy
                    auto fitness_increase = calculate_fitness_(i, current_state);

                    for (int j = 0; j < nb_strategies_; ++j) {
                        // if the strategy to decrease already has count 0
                        // continue, or if we are tying to increase and decrease the same strategy
                        if ((i == j) || (current_state(j) == 0))
                            continue;

                        new_state(j) -= 1;
                        auto new_state_index = egttools::FinitePopulations::calculate_state(population_size_, new_state);

                        // calculate fitness of the decreasing strategy
                        auto fitness_decrease = calculate_fitness_(j, current_state);

                        // calculate transition probability
                        double transition_probability = not_mu * (static_cast<double>(current_state(i)) / (population_size_ - 1));
                        transition_probability *= egttools::FinitePopulations::fermi(beta, fitness_decrease, fitness_increase);
                        transition_probability = (static_cast<double>(current_state(j)) / population_size_) * (transition_probability + mutation_probability);

                        // update transition matrix
                        transition_matrix.coeffRef(current_state_index, static_cast<int64_t>(new_state_index)) = transition_probability;

                        total_probability += transition_probability;
                        new_state(j) += 1;
                    }
                }
            }
            new_state(i) -= 1;
        }
        // update transition matrix with probability of staying in the current state
        if (homomorphic) {
            transition_matrix.coeffRef(current_state_index, current_state_index) = not_mu;
        } else {
            transition_matrix.coeffRef(current_state_index, current_state_index) = 1 - total_probability;
        }
    }

    return transition_matrix;
}

egttools::Vector egttools::FinitePopulations::analytical::PairwiseComparison::calculate_gradient_of_selection(double beta, const Eigen::Ref<const VectorXui> &state) {
    // The gradient of selection can be calculated by summing all
    // transition incoming transition probabilities and resting all
    // outgoing transition probabilities.
    // We can do that by looping over all possible dimensions (nb_strategies)
    // adding a delta (a change possible change in the state), calculating the transition
    // probability from the new state to the current, and subtracting it from the probability
    // of transitioning from the current state to the new.

    Vector gradients = egttools::Vector::Zero(nb_strategies_);
    VectorXui current_state(state);

    for (int i = 0; i < nb_strategies_; ++i) {
        // The first loop is used to get the dimension for which
        // we calculate the gradient.
        if (current_state(i) == 0) continue;

        // Check if decreasing this strategy is possible, otherwise the gradient
        // in this direction is 0.
        for (int j = 0; j < nb_strategies_; ++j) {
            // The second loop is used to get the direction of change
            if (j == i) continue;
            if (current_state(j) == 0) continue;

            auto gradient_increase = calculate_local_gradient_(j, i, beta, current_state);
            gradients(i) += gradient_increase;
            gradients(j) -= gradient_increase;
        }
    }

    return gradients / nb_strategies_;
}

#if (HAS_BOOST)
double egttools::FinitePopulations::analytical::PairwiseComparison::calculate_fixation_probability(int index_invading_strategy, int index_resident_strategy, double beta) {
    cpp_dec_float_100 phi = 0;
    cpp_dec_float_100 prod = 1;
    cpp_dec_float_100 probability_increase, probability_decrease;

    VectorXui population_state = VectorXui::Zero(nb_strategies_);

    for (int i = 1; i < population_size_; ++i) {
        population_state(index_invading_strategy) = i;
        population_state(index_resident_strategy) = population_size_ - i;

        // calculate fitness of invading strategy
        auto fitness_invading_strategy = calculate_fitness_(index_invading_strategy, population_state);
        auto fitness_resident_strategy = calculate_fitness_(index_resident_strategy, population_state);

        // Calculate the probability that the invading strategy will increase
        probability_increase = (static_cast<double>(population_size_ - i) / population_size_) * (static_cast<double>(i) / (population_size_ - 1));
        probability_increase *= egttools::FinitePopulations::fermi(beta, fitness_resident_strategy, fitness_invading_strategy);
        probability_decrease = (static_cast<double>(i) / population_size_) * (static_cast<double>(population_size_ - i) / (population_size_ - 1));
        probability_decrease *= egttools::FinitePopulations::fermi(beta, fitness_invading_strategy, fitness_resident_strategy);

        prod *= probability_decrease / probability_increase;
        phi += prod;

        if (phi > 1e7) return 0.0;
    }

    cpp_dec_float_100 fixation_probability = 1 / (1. + phi);

    return fixation_probability.convert_to<double>();
}
#else
double egttools::FinitePopulations::analytical::PairwiseComparison::calculate_fixation_probability(int index_invading_strategy, int index_resident_strategy, double beta) {
    double phi = 0;
    double prod = 1;
    double probability_increase, probability_decrease;

    VectorXui population_state = VectorXui::Zero(nb_strategies_);

    for (int i = 1; i < population_size_; ++i) {
        population_state(index_invading_strategy) = i;
        population_state(index_resident_strategy) = population_size_ - i;

        // calculate fitness of invading strategy
        auto fitness_invading_strategy = calculate_fitness_(index_invading_strategy, population_state);
        auto fitness_resident_strategy = calculate_fitness_(index_resident_strategy, population_state);

        // Calculate the probability that the invading strategy will increase
        probability_increase = (static_cast<double>(population_state(index_resident_strategy)) / population_size_) * (static_cast<double>(i) / (population_size_ - 1));
        probability_increase *= egttools::FinitePopulations::fermi(beta, fitness_resident_strategy, fitness_invading_strategy);
        probability_decrease = (static_cast<double>(i) / population_size_) * (static_cast<double>(population_state(index_resident_strategy)) / (population_size_ - 1));
        probability_decrease *= egttools::FinitePopulations::fermi(beta, fitness_invading_strategy, fitness_resident_strategy);

        prod *= probability_decrease / probability_increase;
        phi += prod;

        if (phi > 1e7) return 0.0;
    }
    return 1 / (1. + phi);
}
#endif

std::tuple<egttools::Matrix2D, egttools::Matrix2D> egttools::FinitePopulations::analytical::PairwiseComparison::calculate_transition_and_fixation_matrix_sml(double beta) {
    Matrix2D transitions = Matrix2D::Zero(nb_strategies_, nb_strategies_);
    Matrix2D fixation_probabilities = Matrix2D::Zero(nb_strategies_, nb_strategies_);

    //#pragma omp parallel for default(none) shared(beta, nb_strategies_, population_size_, transitions, fixation_probabilities)
    for (int i = 0; i < nb_strategies_; ++i) {
        double transition_stay = 1;
        for (int j = 0; j < nb_strategies_; ++j) {
            if (i != j) {
                auto fixation_probability = calculate_fixation_probability(j, i, beta);
                fixation_probabilities(i, j) = fixation_probability;
                transitions(i, j) = fixation_probability / (nb_strategies_ - 1);
                //#pragma omp atomic update
                transition_stay -= transitions(i, j);
            }
        }
        transitions(i, i) = transition_stay;
    }

    return {transitions, fixation_probabilities};
}

void egttools::FinitePopulations::analytical::PairwiseComparison::update_population_size(int population_size) {
    // Check if the size of the population is positive
    if (population_size <= 0) {
        throw std::invalid_argument(
                "The size of the population must be a positive integer");
    }

    population_size_ = population_size;
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

//double egttools::FinitePopulations::analytical::PairwiseComparison::calculate_transition_(int decreasing_strategy, int increasing_strategy, double beta, double mu, egttools::VectorXui &state) {
//    state(increasing_strategy) -= 1;
//    auto fitness_increasing_strategy = game_.calculate_fitness(increasing_strategy, population_size_, state);
//    state(increasing_strategy) += 1;
//    state(decreasing_strategy) -= 1;
//    auto fitness_decreasing_strategy = game_.calculate_fitness(decreasing_strategy, population_size_, state);
//    state(decreasing_strategy) += 1;
//
//    // To get back from new state to current state, we need
//    // to calculate the probability that strategy i increases and j decreases
//    double mutation_probability = mu / (nb_strategies_ - 1);
//    double transition_probability = (1 - mu) * (static_cast<double>(state(increasing_strategy)) / (population_size_ - 1));
//    transition_probability *= egttools::FinitePopulations::fermi(beta, fitness_decreasing_strategy, fitness_increasing_strategy);
//
//    transition_probability = (static_cast<double>(state(decreasing_strategy)) / population_size_) * (transition_probability + mutation_probability);
//
//    return transition_probability;
//}

double egttools::FinitePopulations::analytical::PairwiseComparison::calculate_local_gradient_(int decreasing_strategy, int increasing_strategy, double beta, egttools::VectorXui &state) {
    state(increasing_strategy) -= 1;
    auto fitness_increasing_strategy = game_.calculate_fitness(increasing_strategy, population_size_, state);
    state(increasing_strategy) += 1;
    state(decreasing_strategy) -= 1;
    auto fitness_decreasing_strategy = game_.calculate_fitness(decreasing_strategy, population_size_, state);
    state(decreasing_strategy) += 1;

    double gradient = (static_cast<double>(state(decreasing_strategy)) / population_size_) * (static_cast<double>(state(increasing_strategy)) / (population_size_ - 1));
    gradient *= tanh((beta / 2) * (fitness_increasing_strategy - fitness_decreasing_strategy));

    return gradient;
}

double egttools::FinitePopulations::analytical::PairwiseComparison::calculate_fitness_(int &strategy_index, egttools::VectorXui &state) {
    double fitness;
    std::stringstream result;
    result << state;

    std::string key = std::to_string(strategy_index) + result.str();

    // First we check if fitness value is in the lookup table
    if (!cache_.exists(key)) {
        state(strategy_index) -= 1;
        fitness = game_.calculate_fitness(strategy_index, population_size_, state);
        state(strategy_index) += 1;

        // Finally we store the new fitness in the Cache. We also keep a Cache for the payoff given each group combination
        cache_.insert(key, fitness);
    } else {
        fitness = cache_.get(key);
    }

    return fitness;
}
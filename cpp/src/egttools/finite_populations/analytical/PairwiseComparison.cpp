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
    AbstractGame &game) : population_size_(population_size),
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
    AbstractGame &game, size_t cache_size) : population_size_(population_size),
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
    const int nb_elements = population_size_ - 2;

#if defined(_OPENMP) && !defined(_MSC_VER)
#pragma omp parallel for default(none) shared(fitnesses, nb_strategies_, population_size_, game_, nb_elements, Eigen::Dynamic)
#endif
    for (int i = 0; i < nb_strategies_; ++i) {
        VectorXui population_state = VectorXui::Zero(nb_strategies_);
        for (int j = i; j < nb_strategies_; ++j) {
            for (int z = 1; z < population_size_; ++z) {
                population_state(i) = z;
                population_state(j) = population_size_ - z;

                // calculate fitness of invading strategy
                population_state(i) -= 1;
                fitnesses(i, j * nb_elements + (z - 1)) = game_.
                        calculate_fitness(i, population_size_, population_state);
                population_state(i) += 1;

                population_state(j) -= 1;
                fitnesses(j, i * nb_elements + (population_size_ - z - 1)) = game_.calculate_fitness(
                    j, population_size_, population_state);
            }
            population_state(j) = 0;
        }
    }

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
                std::string key1 = std::to_string(i) + result.str();
                std::string key2 = std::to_string(j) + result.str();

                cache_.put(key1, fitnesses(i, j * nb_elements + (z - 1)));
                cache_.put(key2, fitnesses(j, i * nb_elements + (population_size_ - z - 1)));
            }
            population_state(j) = 0;
        }
        population_state(i) = 0;
    }
}

egttools::SparseMatrix2D
egttools::FinitePopulations::analytical::PairwiseComparison::calculate_transition_matrix(const double beta, const double mu) {
    if (beta < 0.0) {
        throw std::invalid_argument("beta must be >= 0");
    }
    if (mu < 0.0 || mu > 1.0) {
        throw std::invalid_argument("mu must be in [0,1]");
    }
    if (nb_strategies_ < 2)
        throw std::invalid_argument("At least 2 strategies are required");

    const int64_t S = nb_states_;
    const int k = nb_strategies_;
    const int N = population_size_;
    const double one_minus_mu = 1.0 - mu;
    const double mutation_probability = (k > 2) ? (mu / (k - 1)) : mu;

    std::vector<Eigen::Triplet<double> > trips;
    // Rough upper bound: each row ~ k*(k-1) off-diagonals + 1 diagonal.
    // (Over-estimate is fine; it only affects capacity, not correctness.)
    trips.reserve(static_cast<size_t>(S) * (k * (k - 1) + 1));

    VectorXui current(k), next(k);

    for (int64_t row = 0; row < S; ++row) {
        sample_simplex(row, N, k, current);
        next = current;

        bool monomorphic = false;
        int mono_idx = -1;
        for (int i = 0; i < k; ++i) {
            if (current(i) == static_cast<size_t>(N)) {
                monomorphic = true;
                mono_idx = i;
                break;
            }
        }

        double total_offdiag = 0.0;

        if (monomorphic) {
            // Adjacent single-mutant neighbors: increase any other strategy by 1
            for (int i = 0; i < k; ++i) {
                if (i == mono_idx) continue;
                next = current;
                next(mono_idx) -= 1;
                next(i) += 1;
                const int64_t col = static_cast<int64_t>(calculate_state(N, next));
                trips.emplace_back(row, col, mutation_probability);
                total_offdiag += mutation_probability;
            }
            const double diag = std::max(0.0, 1.0 - total_offdiag); // = 1 - mu
            trips.emplace_back(row, row, diag);
            continue;
        }

        // Non-monomorphic rows
        for (int i = 0; i < k; ++i) {
            // If we try to increase strategy i:
            if (current(i) == static_cast<size_t>(N)) continue; // already handled by monomorphic
            next = current;
            next(i) += 1;

            if (current(i) == 0) {
                // i can only increase via mutation from some j>0
                for (int j = 0; j < k; ++j) {
                    if (j == i || current(j) == 0) continue;
                    next(j) -= 1;
                    const int64_t col = static_cast<int64_t>(calculate_state(N, next));
                    const double prob =
                            (static_cast<double>(current(j)) / N) * mutation_probability;
                    if (prob > 0.0) {
                        trips.emplace_back(row, col, prob);
                        total_offdiag += prob;
                    }
                    next(j) += 1;
                }
            } else {
                // i present: selection + mutation
                const double f_i = calculate_fitness_(i, current);
                for (int j = 0; j < k; ++j) {
                    if (j == i || current(j) == 0) continue;
                    next(j) -= 1;
                    const int64_t col = static_cast<int64_t>(calculate_state(N, next));
                    const double f_j = calculate_fitness_(j, current);

                    double sel = one_minus_mu *
                                 (static_cast<double>(current(i)) / (N - 1)) *
                                 fermi(beta, f_j, f_i);
                    double prob = (static_cast<double>(current(j)) / N) *
                                  (sel + mutation_probability);

                    if (prob > 0.0) {
                        trips.emplace_back(row, col, prob);
                        total_offdiag += prob;
                    }
                    next(j) += 1;
                }
            }
            // revert next(i) done implicitly by reassigning next each loop
        }

        double diag = 1.0 - total_offdiag;
        if (diag < 0.0 && diag > -1e-12) diag = 0.0; // clamp tiny negatives
        trips.emplace_back(row, row, diag);
    }

    SparseMatrix2D P(S, S);
    // Eigen sums duplicates (if any) by default; you can also pass a combiner if you want custom behavior.
    P.setFromTriplets(trips.begin(), trips.end());
    P.makeCompressed();
    P.prune(1e-16); // optional: drop tiny entries

    return P;
}


egttools::Vector egttools::FinitePopulations::analytical::PairwiseComparison::calculate_gradient_of_selection(
    const double beta, const Eigen::Ref<const VectorXui> &state) const {
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
double egttools::FinitePopulations::analytical::PairwiseComparison::calculate_fixation_probability(
    int index_invading_strategy, int index_resident_strategy, const double beta) {
    cpp_dec_float_100 phi = 0;
    cpp_dec_float_100 prod = 1;

    VectorXui population_state = VectorXui::Zero(nb_strategies_);

    for (int i = 1; i < population_size_; ++i) {
        population_state(index_invading_strategy) = i;
        population_state(index_resident_strategy) = population_size_ - i;

        // calculate fitness of invading strategy
        const auto fitness_invading_strategy = calculate_fitness_(index_invading_strategy, population_state);
        const auto fitness_resident_strategy = calculate_fitness_(index_resident_strategy, population_state);

        // Calculate the probability that the invading strategy will increase
        cpp_dec_float_100 probability_increase = (static_cast<double>(population_size_ - i) / population_size_) * (
                                                     static_cast<double>(i) / (population_size_ - 1));
        probability_increase *= fermi(beta, fitness_resident_strategy,
                                      fitness_invading_strategy);
        cpp_dec_float_100 probability_decrease = (static_cast<double>(i) / population_size_) * (
                                                     static_cast<double>(population_size_ - i) / (
                                                         population_size_ - 1));
        probability_decrease *= fermi(beta, fitness_invading_strategy,
                                      fitness_resident_strategy);

        prod *= probability_decrease / probability_increase;
        phi += prod;

        if (phi > 1e7) return 0.0;
    }

    const cpp_dec_float_100 fixation_probability = 1 / (1. + phi);

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

std::tuple<egttools::Matrix2D, egttools::Matrix2D>
egttools::FinitePopulations::analytical::PairwiseComparison::calculate_transition_and_fixation_matrix_sml(
    const double beta) {
    Matrix2D transitions = Matrix2D::Zero(nb_strategies_, nb_strategies_);
    Matrix2D fixation_probabilities = Matrix2D::Zero(nb_strategies_, nb_strategies_);

#if defined(_OPENMP) && !defined(_MSC_VER)
#pragma omp parallel for default(none) shared(beta, nb_strategies_, population_size_, transitions, fixation_probabilities)
#endif
    for (int i = 0; i < nb_strategies_; ++i) {
        double transition_stay = 1;
        for (int j = 0; j < nb_strategies_; ++j) {
            if (i != j) {
                const auto fixation_probability = calculate_fixation_probability(j, i, beta);
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

void egttools::FinitePopulations::analytical::PairwiseComparison::update_population_size(const int population_size) {
    // Check if the size of the population is positive
    if (population_size <= 0) {
        throw std::invalid_argument(
            "The size of the population must be a positive integer");
    }

    population_size_ = population_size;
    nb_states_ = starsBars(population_size_, nb_strategies_);
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

const egttools::FinitePopulations::AbstractGame &
egttools::FinitePopulations::analytical::PairwiseComparison::game() const {
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

double egttools::FinitePopulations::analytical::PairwiseComparison::calculate_local_gradient_(
    const int decreasing_strategy, const int increasing_strategy, const double beta, VectorXui &state) const {
    state(increasing_strategy) -= 1;
    const auto fitness_increasing_strategy = game_.calculate_fitness(increasing_strategy, population_size_, state);
    state(increasing_strategy) += 1;
    state(decreasing_strategy) -= 1;
    const auto fitness_decreasing_strategy = game_.calculate_fitness(decreasing_strategy, population_size_, state);
    state(decreasing_strategy) += 1;

    double gradient = (static_cast<double>(state(decreasing_strategy)) / population_size_) * (
                          static_cast<double>(state(increasing_strategy)) / (population_size_ - 1));
    gradient *= tanh((beta / 2) * (fitness_increasing_strategy - fitness_decreasing_strategy));

    return gradient;
}

double egttools::FinitePopulations::analytical::PairwiseComparison::calculate_fitness_(
    const int &strategy_index, VectorXui &state) {
    double fitness;
    std::stringstream result;
    result << state;

    const std::string key = std::to_string(strategy_index) + result.str();

    // First we check if fitness value is in the lookup table
    if (const auto value = cache_.get(key); value) {
        fitness = *value;
    } else {
        state(strategy_index) -= 1;
        fitness = game_.calculate_fitness(strategy_index, population_size_, state);
        state(strategy_index) += 1;

        // Finally we store the new fitness in the Cache. We also keep a Cache for the payoff given each group combination
        cache_.put(key, fitness);
    }

    return fitness;
}

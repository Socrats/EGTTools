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
#include <atomic>
#include <cmath>
#include <limits>

namespace {
    inline std::uint64_t make_fitness_cache_key(const int64_t state_index,
                                                const int strategy_index) {
        return (static_cast<std::uint64_t>(state_index) << 32) |
               static_cast<std::uint32_t>(strategy_index);
    }
} // namespace

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
    const int nb_elements = population_size_ - 1;

#if defined(_OPENMP) && !defined(_MSC_VER)
#pragma omp parallel for default(shared) shared(fitnesses, nb_strategies_, population_size_, game_, nb_elements)
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
                const int64_t state_index =
                        static_cast<int64_t>(calculate_state(
                            population_size_, population_state));

                const auto key1 = make_fitness_cache_key(state_index, i);
                const auto key2 = make_fitness_cache_key(state_index, j);

                cache_.put(key1, fitnesses(i, j * nb_elements + (z - 1)));
                cache_.put(key2, fitnesses(j, i * nb_elements + (population_size_ - z - 1)));
            }
            population_state(j) = 0;
        }
        population_state(i) = 0;
    }
}

egttools::Matrix2D
egttools::FinitePopulations::analytical::PairwiseComparison::compute_fitness_matrix() {
    // Pre-compute fitness for every (strategy, state) pair in a serial loop.
    // This method calls game_.calculate_fitness() — which may call into Python —
    // and is therefore intentionally serial: the caller must hold the Python GIL
    // if the game has Python callbacks.
    Matrix2D fitness = Matrix2D::Zero(nb_strategies_, nb_states_);
    VectorXui state(nb_strategies_);
    for (int64_t s = 0; s < nb_states_; ++s) {
        sample_simplex(s, population_size_, nb_strategies_, state);
        for (int i = 0; i < nb_strategies_; ++i) {
            if (state(i) > 0) {
                fitness(i, s) = calculate_fitness_(i, state, s);
            }
        }
    }
    return fitness;
}

egttools::SparseMatrix2D
egttools::FinitePopulations::analytical::PairwiseComparison::assemble_transition_matrix_from_fitness(
    const double beta,
    const double mu,
    const Eigen::Ref<const Matrix2D> &fitness_matrix) {

    using Triplet = Eigen::Triplet<double>;
    const double row_tol = 1e-10;

    const int64_t S = nb_states_;
    const int k = nb_strategies_;
    const int N = population_size_;

    const double one_minus_mu = 1.0 - mu;
    const double mutation_probability =
            (k > 2) ? (mu / static_cast<double>(k - 1)) : mu;

    const double inv_N = 1.0 / static_cast<double>(N);
    const double inv_Nm1 = 1.0 / static_cast<double>(N - 1);

    // Allocate one Triplet bucket per OpenMP thread so that each thread writes
    // to its own vector without any synchronisation during the hot loop.
#if defined(_OPENMP) && !defined(_MSC_VER)
    const int max_threads = omp_get_max_threads();
#else
    const int max_threads = 1;
#endif
    const size_t trips_per_thread =
            static_cast<size_t>((S + max_threads - 1) / max_threads) *
            static_cast<size_t>(k * (k - 1) + 1);

    std::vector<std::vector<Triplet>> thread_trips(max_threads);
    for (auto &v: thread_trips) v.reserve(trips_per_thread);

    // Throwing C++ exceptions from inside an OpenMP parallel region is undefined
    // behaviour. Instead we record the first offending row and throw after the loop.
    std::atomic<int64_t> row_sum_error_row{-1};

#if defined(_OPENMP) && !defined(_MSC_VER)
#pragma omp parallel for schedule(dynamic, 32) default(shared) \
    shared(thread_trips, row_sum_error_row, fitness_matrix, S, k, N, beta, \
           one_minus_mu, mutation_probability, inv_N, inv_Nm1, row_tol)
#endif
    for (int64_t row = 0; row < S; ++row) {
#if defined(_OPENMP) && !defined(_MSC_VER)
        auto &local_trips = thread_trips[omp_get_thread_num()];
#else
        auto &local_trips = thread_trips[0];
#endif

        // All per-row working variables are declared here so they are thread-private.
        VectorXui current(k);
        std::vector<int> present;
        present.reserve(k);
        std::vector<double> fitness(k, 0.0);

        sample_simplex(row, N, k, current);

        for (int i = 0; i < k; ++i) {
            if (current(i) > 0) {
                present.push_back(i);
            }
        }

        double total_offdiag = 0.0;

        if (present.size() == 1) {
            const int mono_idx = present[0];

            for (int i = 0; i < k; ++i) {
                if (i == mono_idx) continue;

                current(mono_idx) -= 1;
                current(i) += 1;

                const int64_t col = static_cast<int64_t>(calculate_state(N, current));

#ifndef NDEBUG
                if (col < 0 || col >= S) {
                    throw std::runtime_error(
                        "Calculated next-state index out of bounds in monomorphic row " +
                        std::to_string(row) + ", col=" + std::to_string(col));
                }
#endif

                current(i) -= 1;
                current(mono_idx) += 1;

                local_trips.emplace_back(row, col, mutation_probability);
                total_offdiag += mutation_probability;
            }
        } else {
            // Read pre-computed fitness values (no Python calls here).
            for (const int i: present) {
                fitness[i] = fitness_matrix(i, row);
#ifndef NDEBUG
                if (!std::isfinite(fitness[i])) {
                    throw std::runtime_error(
                        "Non-finite fitness at row " + std::to_string(row) +
                        ", strategy=" + std::to_string(i));
                }
#endif
            }

            for (int i = 0; i < k; ++i) {
                if (current(i) == 0) {
                    // Strategy i is absent: it can only increase via mutation from a present strategy j.
                    current(i) += 1;

                    for (const int j: present) {
                        if (j == i) continue;

                        current(j) -= 1;
                        const int64_t col = static_cast<int64_t>(calculate_state(N, current));
                        current(j) += 1;

#ifndef NDEBUG
                        if (col < 0 || col >= S) {
                            throw std::runtime_error(
                                "Calculated next-state index out of bounds at row " +
                                std::to_string(row) + ", col=" + std::to_string(col));
                        }
#endif

                        const double prob =
                                static_cast<double>(current(j)) * inv_N * mutation_probability;

#ifndef NDEBUG
                        if (!std::isfinite(prob) || prob < 0.0) {
                            throw std::runtime_error(
                                "Invalid mutation probability at row " + std::to_string(row) +
                                ", i=" + std::to_string(i) +
                                ", j=" + std::to_string(j) +
                                ", prob=" + std::to_string(prob));
                        }
#endif

                        if (prob > 0.0) {
                            local_trips.emplace_back(row, col, prob);
                            total_offdiag += prob;
                        }
                    }

                    current(i) -= 1;
                } else {
                    // Strategy i is present: it can increase via mutation and selection.
                    const double f_i = fitness[i];
                    const double selection_prefactor =
                            one_minus_mu * static_cast<double>(current(i)) * inv_Nm1;

                    current(i) += 1;

                    for (const int j: present) {
                        if (j == i) continue;

                        current(j) -= 1;
                        const int64_t col = static_cast<int64_t>(calculate_state(N, current));
                        current(j) += 1;

#ifndef NDEBUG
                        if (col < 0 || col >= S) {
                            throw std::runtime_error(
                                "Calculated next-state index out of bounds at row " +
                                std::to_string(row) + ", col=" + std::to_string(col));
                        }
#endif

                        const double selection_probability =
                                selection_prefactor * fermi(beta, fitness[j], f_i);

                        const double prob =
                                static_cast<double>(current(j)) * inv_N *
                                (selection_probability + mutation_probability);

#ifndef NDEBUG
                        if (!std::isfinite(prob) || prob < 0.0) {
                            throw std::runtime_error(
                                "Invalid transition probability at row " + std::to_string(row) +
                                ", i=" + std::to_string(i) +
                                ", j=" + std::to_string(j) +
                                ", prob=" + std::to_string(prob));
                        }
#endif

                        if (prob > 0.0) {
                            local_trips.emplace_back(row, col, prob);
                            total_offdiag += prob;
                        }
                    }

                    current(i) -= 1;
                }
            }
        }

        if (total_offdiag > 1.0 + row_tol) {
            // Record the first offending row; we will throw after the parallel region.
            int64_t expected = -1;
            row_sum_error_row.compare_exchange_strong(expected, row);
        }

        const double diag = std::max(0.0, 1.0 - total_offdiag);
        local_trips.emplace_back(row, row, diag);
    }

    if (row_sum_error_row.load() >= 0) {
        throw std::runtime_error(
            "Transition matrix row sum exceeded 1 at row " +
            std::to_string(row_sum_error_row.load()));
    }

    // Merge per-thread buckets into a single flat vector.
    std::vector<Triplet> trips;
    {
        size_t total = 0;
        for (const auto &v: thread_trips) total += v.size();
        trips.reserve(total);
        for (auto &v: thread_trips)
            trips.insert(trips.end(),
                         std::make_move_iterator(v.begin()),
                         std::make_move_iterator(v.end()));
    }

    SparseMatrix2D transition_matrix(S, S);
    transition_matrix.setFromTriplets(trips.begin(), trips.end());
    transition_matrix.makeCompressed();

    return transition_matrix;
}

egttools::SparseMatrix2D
egttools::FinitePopulations::analytical::PairwiseComparison::calculate_transition_matrix(
    const double beta,
    const double mu) {
    if (beta < 0.0) {
        throw std::invalid_argument("beta must be >= 0");
    }
    if (mu < 0.0 || mu > 1.0) {
        throw std::invalid_argument("mu must be in [0,1]");
    }
    if (nb_strategies_ < 2) {
        throw std::invalid_argument("At least 2 strategies are required");
    }
    if (population_size_ < 2) {
        throw std::invalid_argument("Population size must be >= 2");
    }

    // Pre-compute all fitness values serially, then assemble the matrix in parallel.
    return assemble_transition_matrix_from_fitness(beta, mu, compute_fitness_matrix());
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

double egttools::FinitePopulations::analytical::PairwiseComparison::effective_mutation_probability_(
    const double mu) const {
    return (nb_strategies_ > 2)
               ? (mu / static_cast<double>(nb_strategies_ - 1))
               : mu;
}

egttools::Vector
egttools::FinitePopulations::analytical::PairwiseComparison::calculate_gradient_of_selection_with_mutation(
    const double beta,
    const double mu,
    const Eigen::Ref<const VectorXui> &state) const {
    if (beta < 0.0) {
        throw std::invalid_argument("beta must be >= 0");
    }
    if (mu < 0.0 || mu > 1.0) {
        throw std::invalid_argument("mu must be in [0,1]");
    }

    // Fast path: no mutation -> reuse the existing implementation exactly.
    if (mu == 0.0) {
        return calculate_gradient_of_selection(beta, state);
    }

    Vector gradients = Vector::Zero(nb_strategies_);

    // Selection contribution.
    // This is by far the expensive part because it requires repeated fitness evaluations.
    // If mu == 1, selection disappears completely, so we skip it.
    if (mu < 1.0) {
        gradients = (1.0 - mu) * calculate_gradient_of_selection(beta, state);
    }

    // Mutation contribution.
    //
    // Under the current transition convention, the mutation-only contribution for strategy i is
    //
    //   (1 / nb_strategies_) * sum_{j != i} [ x_j / Z * m_eff - x_i / Z * m_eff ]
    //
    // where
    //   m_eff = mu / (nb_strategies - 1)    if nb_strategies > 2
    //   m_eff = mu                           if nb_strategies == 2
    //
    // which simplifies to
    //
    //   m_eff / (nb_strategies * Z) * (Z - nb_strategies * x_i).
    //
    const double mutation_probability = effective_mutation_probability_(mu);
    const double inv_population_size = 1.0 / static_cast<double>(population_size_);
    const double inv_nb_strategies = 1.0 / static_cast<double>(nb_strategies_);
    const double mutation_prefactor =
            mutation_probability * inv_population_size * inv_nb_strategies;

    for (int i = 0; i < nb_strategies_; ++i) {
        gradients(i) += mutation_prefactor *
        (static_cast<double>(population_size_) -
         static_cast<double>(nb_strategies_) * static_cast<double>(state(i)));
    }

    return gradients;
}

// ---------------------------------------------------------------------------
// Private helper: streaming log-sum-exp yielding log(φ)
// ---------------------------------------------------------------------------
double egttools::FinitePopulations::analytical::PairwiseComparison::calculate_log_phi_(
    int index_invading_strategy, int index_resident_strategy, double beta) {
    double log_prod = 0.0;
    double max_log  = -std::numeric_limits<double>::infinity();
    double sum_exp  = 0.0;

    VectorXui population_state = VectorXui::Zero(nb_strategies_);

    for (int i = 1; i < population_size_; ++i) {
        population_state(index_invading_strategy) = i;
        population_state(index_resident_strategy) = population_size_ - i;

        const int64_t state_index =
                static_cast<int64_t>(egttools::FinitePopulations::calculate_state(
                    population_size_, population_state));

        const auto f_inv = calculate_fitness_(index_invading_strategy, population_state, state_index);
        const auto f_res = calculate_fitness_(index_resident_strategy, population_state, state_index);

        // log(p- / p+) = beta * (f_resident - f_invading): always finite
        log_prod += beta * (f_res - f_inv);

        if (log_prod > max_log) {
            sum_exp = sum_exp * std::exp(max_log - log_prod) + 1.0;
            max_log = log_prod;
        } else {
            sum_exp += std::exp(log_prod - max_log);
        }
    }

    // sum_exp == 0 only when pop_size == 1 (no iterations): φ = 0 → log φ = -∞
    if (sum_exp == 0.0) return -std::numeric_limits<double>::infinity();
    return max_log + std::log(sum_exp);
}

// ---------------------------------------------------------------------------
// Public: fixation probability ρ (double precision)
// ---------------------------------------------------------------------------
double egttools::FinitePopulations::analytical::PairwiseComparison::calculate_fixation_probability(
    int index_invading_strategy, int index_resident_strategy, double beta) {
    const double log_phi = calculate_log_phi_(index_invading_strategy, index_resident_strategy, beta);
    // 1/(1+exp(log_phi)): exp returns +∞ for large log_phi → result is 0.0, which is correct
    return 1.0 / (1.0 + std::exp(log_phi));
}

// ---------------------------------------------------------------------------
// Public: log fixation probability log(ρ) — always a finite double
// ---------------------------------------------------------------------------
double egttools::FinitePopulations::analytical::PairwiseComparison::calculate_log_fixation_probability(
    int index_invading_strategy, int index_resident_strategy, double beta) {
    const double log_phi = calculate_log_phi_(index_invading_strategy, index_resident_strategy, beta);

    if (log_phi == -std::numeric_limits<double>::infinity()) return 0.0;  // ρ = 1

    // log(ρ) = -softplus(log_phi) = -log(1 + exp(log_phi)), numerically stable:
    //   log_phi >= 0: -(log_phi + log1p(exp(-log_phi)))
    //   log_phi <  0: -log1p(exp(log_phi))
    return log_phi >= 0.0
               ? -(log_phi + std::log1p(std::exp(-log_phi)))
               : -std::log1p(std::exp(log_phi));
}

// ---------------------------------------------------------------------------
// Public (Boost): fixation probability using 50-digit decimal arithmetic
// ---------------------------------------------------------------------------
#if (HAS_BOOST)
double egttools::FinitePopulations::analytical::PairwiseComparison::calculate_fixation_probability_boost(
    int index_invading_strategy, int index_resident_strategy, double beta) {
    using Scalar = cpp_dec_float_50;
    using boost::multiprecision::exp;
    using boost::multiprecision::log;

    Scalar log_prod = 0;
    Scalar max_log  = std::numeric_limits<Scalar>::lowest();  // effectively −∞
    Scalar sum_exp  = 0;

    VectorXui population_state = VectorXui::Zero(nb_strategies_);

    for (int i = 1; i < population_size_; ++i) {
        population_state(index_invading_strategy) = i;
        population_state(index_resident_strategy) = population_size_ - i;

        const int64_t state_index =
                static_cast<int64_t>(egttools::FinitePopulations::calculate_state(
                    population_size_, population_state));

        const auto f_inv = calculate_fitness_(index_invading_strategy, population_state, state_index);
        const auto f_res = calculate_fitness_(index_resident_strategy, population_state, state_index);

        log_prod += Scalar(beta) * (Scalar(f_res) - Scalar(f_inv));

        if (log_prod > max_log) {
            sum_exp = sum_exp * exp(max_log - log_prod) + Scalar(1);
            max_log = log_prod;
        } else {
            sum_exp += exp(log_prod - max_log);
        }
    }

    if (sum_exp == 0) return 1.0;
    const Scalar log_phi = max_log + log(sum_exp);
    const Scalar rho     = Scalar(1) / (Scalar(1) + exp(log_phi));
    return rho.convert_to<double>();
}
#endif

// ---------------------------------------------------------------------------
// Public: SML transition matrix + log-fixation matrix
// ---------------------------------------------------------------------------
std::tuple<egttools::Matrix2D, egttools::Matrix2D>
egttools::FinitePopulations::analytical::PairwiseComparison::calculate_transition_and_log_fixation_matrix_sml(
    const double beta) {
    Matrix2D log_rho    = Matrix2D::Constant(nb_strategies_, nb_strategies_, 0.0);
    Matrix2D transitions = Matrix2D::Zero(nb_strategies_, nb_strategies_);

    // --- Step 1: compute all log-ρ values ---
#if defined(_OPENMP) && !defined(_MSC_VER)
#pragma omp parallel for default(shared) shared(beta, nb_strategies_, log_rho)
#endif
    for (int i = 0; i < nb_strategies_; ++i) {
        for (int j = 0; j < nb_strategies_; ++j) {
            if (i != j)
                log_rho(i, j) = calculate_log_fixation_probability(j, i, beta);
        }
    }

    // --- Step 2: find global max off-diagonal log-ρ for scaling ---
    double max_log_rho = -std::numeric_limits<double>::infinity();
    for (int i = 0; i < nb_strategies_; ++i)
        for (int j = 0; j < nb_strategies_; ++j)
            if (i != j) max_log_rho = std::max(max_log_rho, log_rho(i, j));

    // --- Step 3: build transition matrix scaled by exp(-max_log_rho) ---
    // Scaling all off-diagonal entries by the same constant preserves the
    // stationary distribution while guaranteeing a valid stochastic matrix.
    for (int i = 0; i < nb_strategies_; ++i) {
        double transition_stay = 1.0;
        for (int j = 0; j < nb_strategies_; ++j) {
            if (i != j) {
                transitions(i, j) = std::exp(log_rho(i, j) - max_log_rho) / (nb_strategies_ - 1);
                transition_stay -= transitions(i, j);
            }
        }
        transitions(i, i) = transition_stay;
    }

    return {transitions, log_rho};
}

std::tuple<egttools::Matrix2D, egttools::Matrix2D>
egttools::FinitePopulations::analytical::PairwiseComparison::calculate_transition_and_fixation_matrix_sml(
    const double beta) {
    Matrix2D transitions = Matrix2D::Zero(nb_strategies_, nb_strategies_);
    Matrix2D fixation_probabilities = Matrix2D::Zero(nb_strategies_, nb_strategies_);

#if defined(_OPENMP) && !defined(_MSC_VER)
#pragma omp parallel for default(shared) shared(beta, nb_strategies_, population_size_, transitions, fixation_probabilities)
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
    const int strategy_index,
    const VectorXui &state,
    const int64_t state_index) {
    const auto key = make_fitness_cache_key(state_index, strategy_index);

    if (const auto value = cache_.get(key); value) {
        return *value;
    }

    VectorXui tmp(state);
    tmp(strategy_index) -= 1;

    const double fitness =
            game_.calculate_fitness(strategy_index, population_size_, tmp);

    cache_.put(key, fitness);
    return fitness;
}

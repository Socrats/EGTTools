/** Copyright (c) 2024  Elias Fernandez
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
#ifndef EGTTOOLS_FINITEPOPULATIONS_NETWORKMCESTIMATOR_HPP
#define EGTTOOLS_FINITEPOPULATIONS_NETWORKMCESTIMATOR_HPP

#include <egttools/LruCache.hpp>
#include <egttools/SeedGenerator.h>
#include <egttools/Types.h>
#include <egttools/finite_populations/games/AbstractSpatialGame.hpp>
#include <egttools/finite_populations/structure/AbstractNetworkStructure.hpp>
#include <egttools/finite_populations/update_rules/PairwiseComparison.hpp>

#include <algorithm>
#include <cmath>
#include <functional>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#if defined(_OPENMP)
#include <egttools/OpenMPExtensions.hpp>
#endif

namespace egttools::FinitePopulations {

    using AdjacencyList = structure::AdjacencyList;
    using NodeDictionary = structure::NodeDictionary;
    using AbstractSpatialGame = games::AbstractSpatialGame;

    /**
     * @brief Monte Carlo estimator for evolutionary games on networks.
     *
     * Provides a high-level interface analogous to PairwiseComparisonNumerical,
     * but for network-structured populations. The update rule (pairwise comparison,
     * birth-death, death-birth, time-dependent variants) is a template parameter
     * for zero-overhead dispatch.
     *
     * The topology is stored as a contiguous AdjacencyList (vector<vector<int>>)
     * for O(1) neighbour lookup. Constructors accept both AdjacencyList and the
     * legacy NodeDictionary (map<int,vector<int>>) for Python/NetworkX interop.
     *
     * All stochastic estimation methods use OpenMP to parallelise over independent
     * runs. Each thread has its own RNG and LRU fitness cache, avoiding contention.
     *
     * Batch-based L1 / L∞ convergence checking (identical to
     * PairwiseComparisonNumerical): estimation stops early when the per-batch
     * change in the running estimate falls below `tolerance`. Set `tolerance = 0`
     * (default) to run all `nb_runs` without early stopping.
     *
     * @tparam UpdateRule   Policy struct/class providing `step()` and
     *                      `compute_exact_gradient()`. See update_rules/.
     * @tparam CacheType    Fitness cache (default: LRUCache<string,double>).
     */
    template<class UpdateRule = update_rules::PairwiseComparison,
             class CacheType = egttools::Utils::LRUCache<std::string, double>>
    class NetworkMCEstimator {
    public:
        /**
         * @param game         Spatial game used to evaluate node fitness.
         * @param topology     Network adjacency list (0-indexed nodes).
         * @param nb_strategies Number of distinct strategies.
         * @param beta         Selection intensity (Fermi parameter or exp-weight base).
         * @param mu           Mutation probability per time step.
         * @param cache_size   Per-thread LRU cache size for fitness values.
         * @param update_rule  Update rule instance (default-constructed for stateless rules).
         */
        NetworkMCEstimator(AbstractSpatialGame &game,
                           AdjacencyList topology,
                           int nb_strategies,
                           double beta,
                           double mu,
                           int cache_size = 100000,
                           UpdateRule update_rule = UpdateRule{});

        /**
         * Convenience constructor accepting a NodeDictionary (Python/NetworkX output).
         * Converts internally to AdjacencyList.
         */
        NetworkMCEstimator(AbstractSpatialGame &game,
                           const NodeDictionary &topology,
                           int nb_strategies,
                           double beta,
                           double mu,
                           int cache_size = 100000,
                           UpdateRule update_rule = UpdateRule{});

        // ------------------------------------------------------------------
        // Numerically exact methods (deterministic given population state)
        // ------------------------------------------------------------------

        /**
         * Numerically exact average gradient of selection for a given per-node
         * strategy assignment.
         *
         * The formula is rule-specific; see the UpdateRule documentation.
         *
         * @param population  Per-node strategy indices, length N.
         * @return  Gradient vector of length nb_strategies.
         */
        [[nodiscard]] Vector calculate_gradient_of_selection(
            const std::vector<int> &population);

        // ------------------------------------------------------------------
        // Numerical estimators (stochastic, return mean ± SE)
        // ------------------------------------------------------------------

        /**
         * Estimate fixation probability of a single invader placed on a
         * uniformly random node in an otherwise resident population.
         *
         * A run is counted as fixation if the invader strategy takes over the
         * entire population before nb_generations time-steps elapse.
         *
         * @param invader       Strategy index of the invader.
         * @param resident      Strategy index of the resident.
         * @param nb_runs       Number of independent invasion trials.
         * @param nb_generations Maximum time-steps per trial.
         * @return  Estimated fixation probability in [0, 1].
         */
        [[nodiscard]] double estimate_fixation_probability(
            int invader, int resident,
            int64_t nb_runs, int64_t nb_generations);

        /**
         * Estimate time-averaged strategy frequencies (fractions of nodes
         * adopting each strategy) after the transitory period.
         *
         * Runs are parallelised over OpenMP threads. Convergence is checked
         * in batches of `check_every` runs using the L1 norm of the change in
         * the running estimate.
         *
         * @return  Pair {mean_frequencies, standard_errors}, each of length nb_strategies.
         */
        [[nodiscard]] std::pair<Vector, Vector> estimate_strategy_distribution(
            int64_t nb_runs, int64_t nb_generations, int64_t transitory,
            double tolerance = 0.0, int64_t check_every = 0);

        /**
         * Estimate time-averaged values of user-defined indicator functions.
         *
         * Each indicator receives the full per-node strategy vector and the
         * adjacency list, allowing spatial metrics (clustering, homophily, etc.)
         * to be estimated alongside strategy frequencies.
         *
         * @param indicators  Vector of callables; each maps
         *                    (population, network) -> Vector of indicator values.
         * @return  Pair {means, ses} each of shape (nb_indicators,).
         */
        [[nodiscard]] std::pair<Vector, Vector> estimate_stationary_indicators(
            int64_t nb_runs, int64_t nb_generations, int64_t transitory,
            const std::vector<std::function<Vector(const std::vector<int> &,
                                                   const AdjacencyList &)>> &indicators,
            double tolerance = 0.0, int64_t check_every = 0);

        // ------------------------------------------------------------------
        // Trajectory methods
        // ------------------------------------------------------------------

        /**
         * Run a single trajectory and return aggregate strategy counts for each
         * generation after `transitory`.
         *
         * One "generation" = N asynchronous time-steps (N = population size).
         *
         * @param nb_generations  Total generations to simulate.
         * @param transitory      Burn-in generations to discard.
         * @param init_state      Initial strategy counts (length nb_strategies).
         * @return  Matrix of shape (nb_generations - transitory, nb_strategies).
         */
        [[nodiscard]] MatrixXui2D run(int64_t nb_generations, int64_t transitory,
                                      const VectorXui &init_state);

        /**
         * Run a single trajectory and invoke `callback(t, population)` every
         * `snapshot_interval` generations after `transitory`.
         *
         * The callback receives the generation index and a const-reference to the
         * per-node strategy vector. This design avoids storing all snapshots in
         * memory; the caller controls how data is collected (e.g. into a Python
         * generator, a file, or a pre-allocated array).
         *
         * @param snapshot_interval  Call callback every this many generations (post-transitory).
         * @param callback           Invoked as callback(generation, population).
         */
        void run_snapshots(int64_t nb_generations, int64_t transitory,
                           int64_t snapshot_interval,
                           const VectorXui &init_state,
                           const std::function<void(int64_t, const std::vector<int> &)> &callback);

        // ------------------------------------------------------------------
        // Average Gradient of Selection (AGoS) estimators
        // ------------------------------------------------------------------

        /**
         * Estimate the time-independent Average Gradient of Selection G^A(j).
         *
         * Runs `nb_runs` independent trajectories. At each post-transitory
         * generation, computes the numerically exact gradient for the current
         * per-node population and bins the result by cooperator count j
         * (= count of strategy 0). Parallelised over runs with OpenMP;
         * per-thread caches avoid contention.
         *
         * @param nb_runs         Independent trajectories.
         * @param nb_generations  Generations per trajectory.
         * @param transitory      Burn-in generations discarded from binning.
         * @return Pair (mean_G, se_G) each of shape (N+1, nb_strategies).
         *         Row j = mean/SE gradient at cooperator count j.
         *         Rows 0 and N are zero (absorbing states).
         */
        /**
         * When `runs_per_j > 0` the paper's sampling scheme is used: for each
         * initial cooperator count j0 ∈ {1, …, N-1} exactly `runs_per_j`
         * independent trajectories are started with j0 cooperators placed on
         * uniformly random nodes.  Total runs = runs_per_j × (N-1).
         * When `runs_per_j == 0` (default) `nb_runs` trajectories are started
         * from a uniformly random strategy assignment.
         */
        [[nodiscard]] std::pair<Matrix2D, Matrix2D> estimate_agos(
            int64_t nb_runs, int64_t nb_generations, int64_t transitory = 0,
            int64_t runs_per_j = 0);

        /**
         * Estimate the time-dependent Average Gradient of Selection G^A(j, t).
         *
         * Same as estimate_agos but keeps the generation index, allowing study
         * of how the gradient landscape evolves from the initial transient to
         * the stationary regime.
         *
         * @param nb_runs         Independent trajectories.
         * @param nb_generations  Generations per trajectory (all recorded, no transitory).
         * @return Pair (mean_G_t, se_G_t):
         *         - mean_G_t[t] is a (N+1, nb_strategies) matrix at generation t.
         *         - se_G_t[t]   is the corresponding standard-error matrix.
         *         Both are Matrix3D = std::vector<Matrix2D> of length nb_generations.
         */
        [[nodiscard]] std::pair<Matrix3D, Matrix3D> estimate_agos_time_dependent(
            int64_t nb_runs, int64_t nb_generations);

        // ------------------------------------------------------------------
        // Accessors
        // ------------------------------------------------------------------

        [[nodiscard]] int population_size() const { return population_size_; }
        [[nodiscard]] int nb_strategies() const { return nb_strategies_; }
        [[nodiscard]] double beta() const { return beta_; }
        [[nodiscard]] double mu() const { return mu_; }
        [[nodiscard]] const AdjacencyList &topology() const { return network_; }

        void set_beta(double beta) { beta_ = beta; }
        void set_mu(double mu) { mu_ = mu; }

        // ------------------------------------------------------------------
        // Step-by-step simulation (interactive / manual use)
        // ------------------------------------------------------------------

        /** Initialise a simulation session with the given strategy counts. */
        void initialize(const VectorXui &init_state);

        /** Initialise a simulation session with a uniformly random strategy assignment. */
        void initialize();

        /**
         * Advance the session by one generation.
         * For async rules: N individual update steps.
         * For sync rules (e.g. LinearProportional): one full simultaneous sweep.
         * Throws std::runtime_error if initialize() has not been called first.
         */
        void step();

        /** Per-node strategy assignments (length = population_size). */
        [[nodiscard]] const std::vector<int> &population_strategies() const;

        /** Strategy count vector (length = nb_strategies). */
        [[nodiscard]] const VectorXui &mean_population_state() const;

    private:
        AbstractSpatialGame &game_;
        AdjacencyList network_;
        int population_size_, nb_strategies_, cache_size_;
        double beta_, mu_;
        UpdateRule update_rule_;

        // --- step-by-step session state (used by initialize / step) ---
        std::vector<int> session_population_;
        VectorXui        session_mean_state_;
        std::mt19937_64  session_gen_;
        CacheType        session_cache_;
        VectorXui        session_nbuf_;
        int64_t          session_step_count_{0};
        bool             session_initialized_{false};

        // ------------------------------------------------------------------
        // Internal helpers
        // ------------------------------------------------------------------

        void initialize_state_(std::vector<int> &population, VectorXui &mean_state,
                                const VectorXui &target_counts, std::mt19937_64 &gen) const;

        // Place exactly one invader on a random node in an all-resident population.
        void initialize_invasion_(std::vector<int> &population, VectorXui &mean_state,
                                  int invader, int resident, std::mt19937_64 &gen) const;

        // Advance by one generation: N async steps or one full sync sweep.
        void do_generation_(std::vector<int> &population, VectorXui &mean_state,
                            CacheType &cache, VectorXui &nbuf,
                            std::mt19937_64 &gen, UpdateRule &rule, int64_t gen_idx);

        // Check whether the population is monomorphic.
        static bool is_monomorphic_(const VectorXui &mean_state);
    };

    // ======================================================================
    // Implementation
    // ======================================================================

    template<class UR, class CT>
    NetworkMCEstimator<UR, CT>::NetworkMCEstimator(AbstractSpatialGame &game,
                                                   AdjacencyList topology,
                                                   int nb_strategies,
                                                   double beta,
                                                   double mu,
                                                   int cache_size,
                                                   UR update_rule)
        : game_(game),
          network_(std::move(topology)),
          population_size_(static_cast<int>(network_.size())),
          nb_strategies_(nb_strategies),
          cache_size_(cache_size),
          beta_(beta),
          mu_(mu),
          update_rule_(std::move(update_rule)),
          session_cache_(cache_size) {}

    template<class UR, class CT>
    NetworkMCEstimator<UR, CT>::NetworkMCEstimator(AbstractSpatialGame &game,
                                                   const NodeDictionary &topology,
                                                   int nb_strategies,
                                                   double beta,
                                                   double mu,
                                                   int cache_size,
                                                   UR update_rule)
        : NetworkMCEstimator(game,
                             structure::dict_to_adjacency_list(topology),
                             nb_strategies, beta, mu, cache_size,
                             std::move(update_rule)) {}

    // ------------------------------------------------------------------
    // Numerically exact gradient
    // ------------------------------------------------------------------

    template<class UR, class CT>
    Vector NetworkMCEstimator<UR, CT>::calculate_gradient_of_selection(
        const std::vector<int> &population) {
        CT cache(cache_size_);
        VectorXui nbuf = VectorXui::Zero(nb_strategies_);
        return update_rule_.compute_exact_gradient(
            population, network_, game_, cache, nbuf, nb_strategies_, beta_);
    }

    // ------------------------------------------------------------------
    // Fixation probability
    // ------------------------------------------------------------------

    template<class UR, class CT>
    double NetworkMCEstimator<UR, CT>::estimate_fixation_probability(
        int invader, int resident, int64_t nb_runs, int64_t nb_generations) {
        if (invader < 0 || invader >= nb_strategies_)
            throw std::invalid_argument("invader must be in [0, nb_strategies)");
        if (resident < 0 || resident >= nb_strategies_)
            throw std::invalid_argument("resident must be in [0, nb_strategies)");
        if (invader == resident)
            throw std::invalid_argument("invader and resident must differ");

        long int fixations = 0;
        long int extinctions = 0;

#if defined(_OPENMP) && !defined(_MSC_VER)
#pragma omp parallel for reduction(+ : fixations, extinctions) default(none) \
    shared(invader, resident, nb_runs, nb_generations)
#endif
        for (int64_t run = 0; run < nb_runs; ++run) {
            std::mt19937_64 gen(egttools::Random::SeedGenerator::getInstance().getSeed());
            CT cache(cache_size_);
            VectorXui nbuf = VectorXui::Zero(nb_strategies_);
            UR local_rule = update_rule_;

            std::vector<int> population(population_size_, resident);
            VectorXui mean_state = VectorXui::Zero(nb_strategies_);
            mean_state(resident) = population_size_;

            // Place one invader on a random node
            std::uniform_int_distribution<int> node_dist(0, population_size_ - 1);
            int seed_node = node_dist(gen);
            mean_state(resident) -= 1;
            mean_state(invader) += 1;
            population[seed_node] = invader;

            for (int64_t t = 0; t < nb_generations; ++t) {
                do_generation_(population, mean_state, cache, nbuf, gen, local_rule, t);
                if (static_cast<int>(mean_state(invader)) == population_size_) { ++fixations; break; }
                if (mean_state(invader) == 0u) { ++extinctions; break; }
            }
        }

        long int decided = fixations + extinctions;
        if (decided == 0) return 0.0;
        return static_cast<double>(fixations) / static_cast<double>(decided);
    }

    // ------------------------------------------------------------------
    // Strategy distribution
    // ------------------------------------------------------------------

    template<class UR, class CT>
    std::pair<Vector, Vector> NetworkMCEstimator<UR, CT>::estimate_strategy_distribution(
        int64_t nb_runs, int64_t nb_generations, int64_t transitory,
        double tolerance, int64_t check_every) {
        if (transitory >= nb_generations)
            throw std::invalid_argument("transitory must be < nb_generations");

        const int64_t counting_gens = nb_generations - transitory;
        const int64_t batch = (check_every > 0) ? check_every
                                                 : std::max<int64_t>(1, nb_runs / 10);

        Vector sum_freq = Vector::Zero(nb_strategies_);
        Vector sum_freq2 = Vector::Zero(nb_strategies_);
        Vector prev_estimate = Vector::Zero(nb_strategies_);
        int64_t runs_done = 0;

        while (runs_done < nb_runs) {
            const int64_t this_batch = std::min(batch, nb_runs - runs_done);
            Vector batch_sum = Vector::Zero(nb_strategies_);

#if defined(_OPENMP) && !defined(_MSC_VER)
#pragma omp parallel for reduction(+ : batch_sum) default(none) \
    shared(this_batch, nb_generations, transitory, counting_gens)
#endif
            for (int64_t run = 0; run < this_batch; ++run) {
                std::mt19937_64 gen(egttools::Random::SeedGenerator::getInstance().getSeed());
                CT cache(cache_size_);
                VectorXui nbuf = VectorXui::Zero(nb_strategies_);
                UR local_rule = update_rule_;

                // Random initial state
                std::vector<int> population(population_size_);
                VectorXui mean_state = VectorXui::Zero(nb_strategies_);
                std::uniform_int_distribution<int> s_dist(0, nb_strategies_ - 1);
                for (int i = 0; i < population_size_; ++i) {
                    population[i] = s_dist(gen);
                    mean_state(population[i]) += 1;
                }

                Vector run_sum = Vector::Zero(nb_strategies_);

                for (int64_t g = 0; g < nb_generations; ++g) {
                    do_generation_(population, mean_state, cache, nbuf, gen, local_rule, g);
                    if (g >= transitory) {
                        for (int s = 0; s < nb_strategies_; ++s)
                            run_sum(s) += static_cast<double>(mean_state(s)) / population_size_;
                    }
                }
                batch_sum += run_sum / static_cast<double>(counting_gens);
            }

            sum_freq += batch_sum;
            sum_freq2 += batch_sum.array().square().matrix();
            runs_done += this_batch;

            if (tolerance > 0.0 && runs_done > 0) {
                Vector current_estimate = sum_freq / static_cast<double>(runs_done);
                if ((current_estimate - prev_estimate).lpNorm<1>() < tolerance) break;
                prev_estimate = current_estimate;
            }
        }

        Vector mean = sum_freq / static_cast<double>(runs_done);
        Vector variance = (sum_freq2 / static_cast<double>(runs_done)) - mean.array().square().matrix();
        // Clamp numerical noise
        variance = variance.cwiseMax(0.0);
        Vector se = (variance / static_cast<double>(runs_done)).cwiseSqrt();
        return {mean, se};
    }

    // ------------------------------------------------------------------
    // Stationary indicators
    // ------------------------------------------------------------------

    template<class UR, class CT>
    std::pair<Vector, Vector> NetworkMCEstimator<UR, CT>::estimate_stationary_indicators(
        int64_t nb_runs, int64_t nb_generations, int64_t transitory,
        const std::vector<std::function<Vector(const std::vector<int> &,
                                               const AdjacencyList &)>> &indicators,
        double tolerance, int64_t check_every) {
        if (transitory >= nb_generations)
            throw std::invalid_argument("transitory must be < nb_generations");
        if (indicators.empty())
            throw std::invalid_argument("at least one indicator function required");

        const int nb_ind = static_cast<int>(indicators.size());
        const int64_t counting_gens = nb_generations - transitory;
        const int64_t batch = (check_every > 0) ? check_every
                                                 : std::max<int64_t>(1, nb_runs / 10);

        Vector sum_ind = Vector::Zero(nb_ind);
        Vector sum_ind2 = Vector::Zero(nb_ind);
        Vector prev_estimate = Vector::Zero(nb_ind);
        int64_t runs_done = 0;

        while (runs_done < nb_runs) {
            const int64_t this_batch = std::min(batch, nb_runs - runs_done);
            Vector batch_sum = Vector::Zero(nb_ind);

#if defined(_OPENMP) && !defined(_MSC_VER)
#pragma omp parallel for reduction(+ : batch_sum) default(none) \
    shared(this_batch, nb_generations, transitory, counting_gens, indicators, nb_ind)
#endif
            for (int64_t run = 0; run < this_batch; ++run) {
                std::mt19937_64 gen(egttools::Random::SeedGenerator::getInstance().getSeed());
                CT cache(cache_size_);
                VectorXui nbuf = VectorXui::Zero(nb_strategies_);
                UR local_rule = update_rule_;

                std::vector<int> population(population_size_);
                VectorXui mean_state = VectorXui::Zero(nb_strategies_);
                std::uniform_int_distribution<int> s_dist(0, nb_strategies_ - 1);
                for (int i = 0; i < population_size_; ++i) {
                    population[i] = s_dist(gen);
                    mean_state(population[i]) += 1;
                }

                Vector run_sum = Vector::Zero(nb_ind);

                for (int64_t g = 0; g < nb_generations; ++g) {
                    do_generation_(population, mean_state, cache, nbuf, gen, local_rule, g);
                    if (g >= transitory) {
                        for (int k = 0; k < nb_ind; ++k) {
                            Vector vals = indicators[k](population, network_);
                            run_sum(k) += (vals.size() == 1) ? vals(0)
                                                              : vals.sum() / vals.size();
                        }
                    }
                }
                batch_sum += run_sum / static_cast<double>(counting_gens);
            }

            sum_ind += batch_sum;
            sum_ind2 += batch_sum.array().square().matrix();
            runs_done += this_batch;

            if (tolerance > 0.0 && runs_done > 0) {
                Vector current_estimate = sum_ind / static_cast<double>(runs_done);
                if ((current_estimate - prev_estimate).lpNorm<1>() < tolerance) break;
                prev_estimate = current_estimate;
            }
        }

        Vector mean = sum_ind / static_cast<double>(runs_done);
        Vector variance = (sum_ind2 / static_cast<double>(runs_done)) - mean.array().square().matrix();
        variance = variance.cwiseMax(0.0);
        Vector se = (variance / static_cast<double>(runs_done)).cwiseSqrt();
        return {mean, se};
    }

    // ------------------------------------------------------------------
    // Trajectory: aggregate counts
    // ------------------------------------------------------------------

    template<class UR, class CT>
    MatrixXui2D NetworkMCEstimator<UR, CT>::run(int64_t nb_generations, int64_t transitory,
                                                const VectorXui &init_state) {
        if (transitory >= nb_generations)
            throw std::invalid_argument("transitory must be < nb_generations");

        const int64_t recording_gens = nb_generations - transitory;
        MatrixXui2D trajectory(recording_gens, nb_strategies_);

        std::mt19937_64 gen(egttools::Random::SeedGenerator::getInstance().getSeed());
        CT cache(cache_size_);
        VectorXui nbuf = VectorXui::Zero(nb_strategies_);

        std::vector<int> population(population_size_);
        VectorXui mean_state(nb_strategies_);
        initialize_state_(population, mean_state, init_state, gen);

        int64_t row = 0;
        for (int64_t g = 0; g < nb_generations; ++g) {
            do_generation_(population, mean_state, cache, nbuf, gen, update_rule_, g);
            if (g >= transitory) {
                trajectory.row(row++) = mean_state;
            }
        }
        return trajectory;
    }

    // ------------------------------------------------------------------
    // Trajectory: per-node snapshots via callback
    // ------------------------------------------------------------------

    template<class UR, class CT>
    void NetworkMCEstimator<UR, CT>::run_snapshots(
        int64_t nb_generations, int64_t transitory,
        int64_t snapshot_interval,
        const VectorXui &init_state,
        const std::function<void(int64_t, const std::vector<int> &)> &callback) {
        if (transitory >= nb_generations)
            throw std::invalid_argument("transitory must be < nb_generations");
        if (snapshot_interval <= 0)
            throw std::invalid_argument("snapshot_interval must be > 0");

        std::mt19937_64 gen(egttools::Random::SeedGenerator::getInstance().getSeed());
        CT cache(cache_size_);
        VectorXui nbuf = VectorXui::Zero(nb_strategies_);

        std::vector<int> population(population_size_);
        VectorXui mean_state(nb_strategies_);
        initialize_state_(population, mean_state, init_state, gen);

        int64_t post_transitory_gen = 0;
        for (int64_t g = 0; g < nb_generations; ++g) {
            do_generation_(population, mean_state, cache, nbuf, gen, update_rule_, g);
            if (g >= transitory) {
                if (post_transitory_gen % snapshot_interval == 0) {
                    callback(post_transitory_gen, population);
                }
                ++post_transitory_gen;
            }
        }
    }

    // ------------------------------------------------------------------
    // Private helpers
    // ------------------------------------------------------------------

    template<class UR, class CT>
    void NetworkMCEstimator<UR, CT>::initialize_state_(
        std::vector<int> &population, VectorXui &mean_state,
        const VectorXui &target_counts, std::mt19937_64 &gen) const {
        mean_state = target_counts;
        int idx = 0;
        for (int s = 0; s < nb_strategies_; ++s) {
            for (Eigen::Index c = 0; c < static_cast<Eigen::Index>(target_counts(s)); ++c) {
                population[idx++] = s;
            }
        }
        std::shuffle(population.begin(), population.end(), gen);
    }

    template<class UR, class CT>
    bool NetworkMCEstimator<UR, CT>::is_monomorphic_(const VectorXui &mean_state) {
        int non_zero = 0;
        for (Eigen::Index i = 0; i < mean_state.size(); ++i)
            if (mean_state(i) > 0) ++non_zero;
        return non_zero <= 1;
    }

    // ------------------------------------------------------------------
    // do_generation_: sync/async dispatch via if constexpr
    // ------------------------------------------------------------------

    template<class UR, class CT>
    void NetworkMCEstimator<UR, CT>::do_generation_(
        std::vector<int> &population, VectorXui &mean_state,
        CT &cache, VectorXui &nbuf,
        std::mt19937_64 &gen, UR &rule, int64_t gen_idx) {
        if constexpr (UR::synchronous) {
            rule.step(population, mean_state, network_, game_, cache, nbuf,
                      nb_strategies_, beta_, mu_, gen, gen_idx);
        } else {
            for (int s = 0; s < population_size_; ++s) {
                rule.step(population, mean_state, network_, game_, cache, nbuf,
                          nb_strategies_, beta_, mu_, gen,
                          gen_idx * population_size_ + s);
            }
        }
    }

    // ------------------------------------------------------------------
    // Step-by-step session
    // ------------------------------------------------------------------

    template<class UR, class CT>
    void NetworkMCEstimator<UR, CT>::initialize(const VectorXui &init_state) {
        session_population_.resize(population_size_);
        session_mean_state_.resize(nb_strategies_);
        session_gen_.seed(egttools::Random::SeedGenerator::getInstance().getSeed());
        session_cache_ = CT(cache_size_);
        session_nbuf_  = VectorXui::Zero(nb_strategies_);
        initialize_state_(session_population_, session_mean_state_, init_state, session_gen_);
        session_step_count_    = 0;
        session_initialized_   = true;
    }

    template<class UR, class CT>
    void NetworkMCEstimator<UR, CT>::initialize() {
        session_population_.resize(population_size_);
        session_mean_state_ = VectorXui::Zero(nb_strategies_);
        session_gen_.seed(egttools::Random::SeedGenerator::getInstance().getSeed());
        session_cache_ = CT(cache_size_);
        session_nbuf_  = VectorXui::Zero(nb_strategies_);
        std::uniform_int_distribution<int> s_dist(0, nb_strategies_ - 1);
        for (int i = 0; i < population_size_; ++i) {
            session_population_[i] = s_dist(session_gen_);
            session_mean_state_(session_population_[i]) += 1;
        }
        session_step_count_  = 0;
        session_initialized_ = true;
    }

    template<class UR, class CT>
    void NetworkMCEstimator<UR, CT>::step() {
        if (!session_initialized_)
            throw std::runtime_error("NetworkMCEstimator: call initialize() before step()");
        do_generation_(session_population_, session_mean_state_,
                       session_cache_, session_nbuf_, session_gen_,
                       update_rule_, session_step_count_++);
    }

    template<class UR, class CT>
    const std::vector<int> &NetworkMCEstimator<UR, CT>::population_strategies() const {
        if (!session_initialized_)
            throw std::runtime_error("NetworkMCEstimator: call initialize() before accessing population_strategies()");
        return session_population_;
    }

    template<class UR, class CT>
    const VectorXui &NetworkMCEstimator<UR, CT>::mean_population_state() const {
        if (!session_initialized_)
            throw std::runtime_error("NetworkMCEstimator: call initialize() before accessing mean_population_state()");
        return session_mean_state_;
    }

    // ------------------------------------------------------------------
    // AGoS: time-independent
    // ------------------------------------------------------------------

    template<class UR, class CT>
    std::pair<Matrix2D, Matrix2D> NetworkMCEstimator<UR, CT>::estimate_agos(
        int64_t nb_runs, int64_t nb_generations, int64_t transitory,
        int64_t runs_per_j) {
        if (transitory >= nb_generations)
            throw std::invalid_argument("transitory must be < nb_generations");

        const int N = population_size_;
        const int K = nb_strategies_;

        // When runs_per_j > 0 use the paper's sampling scheme: for each initial
        // cooperator count j0 in {1, ..., N-1} run exactly runs_per_j simulations
        // starting from j0 cooperators placed on uniformly random nodes.
        // Total runs = runs_per_j × (N-1).
        // When runs_per_j == 0 fall through to uniform-random initialisation
        // with nb_runs total.
        const bool per_j_mode   = (runs_per_j > 0);
        const int64_t total_runs = per_j_mode
                                   ? runs_per_j * static_cast<int64_t>(N - 1)
                                   : nb_runs;

#if defined(_OPENMP) && !defined(_MSC_VER)
        const int nb_threads = omp_get_max_threads();
#else
        const int nb_threads = 1;
#endif

        std::vector<Matrix2D> thr_sum(nb_threads, Matrix2D::Zero(N + 1, K));
        std::vector<Matrix2D> thr_sum2(nb_threads, Matrix2D::Zero(N + 1, K));
        std::vector<VectorXi> thr_cnt(nb_threads, VectorXi::Zero(N + 1));

#if defined(_OPENMP) && !defined(_MSC_VER)
#pragma omp parallel num_threads(nb_threads) default(none) \
    shared(total_runs, nb_generations, transitory, thr_sum, thr_sum2, thr_cnt, \
           N, K, per_j_mode, runs_per_j)
        {
            const int tid = omp_get_thread_num();
#pragma omp for schedule(dynamic)
            for (int64_t run = 0; run < total_runs; ++run) {
#else
        {
            const int tid = 0;
            for (int64_t run = 0; run < total_runs; ++run) {
#endif
                std::mt19937_64 gen(egttools::Random::SeedGenerator::getInstance().getSeed());
                CT dyn_cache(cache_size_);
                CT grad_cache(cache_size_);
                VectorXui nbuf  = VectorXui::Zero(K);
                VectorXui nbuf2 = VectorXui::Zero(K);
                UR local_rule = update_rule_;

                std::vector<int> population(N, 1);   // all defectors initially
                VectorXui mean_state = VectorXui::Zero(K);

                if (per_j_mode) {
                    // Start from exactly j0 cooperators (paper's scheme)
                    const int j0 = static_cast<int>(run / runs_per_j) + 1; // j0 in [1, N-1]
                    // Shuffle node indices and assign first j0 as cooperators
                    std::vector<int> nodes(N);
                    std::iota(nodes.begin(), nodes.end(), 0);
                    std::shuffle(nodes.begin(), nodes.end(), gen);
                    for (int i = 0; i < j0; ++i) population[nodes[i]] = 0;
                    mean_state(0) = static_cast<size_t>(j0);
                    mean_state(1) = static_cast<size_t>(N - j0);
                } else {
                    // Uniform-random initialisation
                    std::uniform_int_distribution<int> s_dist(0, K - 1);
                    for (int i = 0; i < N; ++i) {
                        population[i] = s_dist(gen);
                        mean_state(population[i]) += 1;
                    }
                }

                for (int64_t g = 0; g < nb_generations; ++g) {
                    do_generation_(population, mean_state, dyn_cache, nbuf, gen, local_rule, g);
                    if (g >= transitory) {
                        const int j = static_cast<int>(mean_state(0));
                        if (j > 0 && j < N) {
                            Vector grad = local_rule.compute_exact_gradient(
                                population, network_, game_, grad_cache, nbuf2, K, beta_);
                            thr_sum[tid].row(j)  += grad.transpose();
                            thr_sum2[tid].row(j) += grad.array().square().matrix().transpose();
                            thr_cnt[tid](j)      += 1;
                        }
                    }
                }
            }
        }

        // Merge per-thread results
        Matrix2D sum_G  = Matrix2D::Zero(N + 1, K);
        Matrix2D sum_G2 = Matrix2D::Zero(N + 1, K);
        VectorXi count  = VectorXi::Zero(N + 1);
        for (int t = 0; t < nb_threads; ++t) {
            sum_G  += thr_sum[t];
            sum_G2 += thr_sum2[t];
            count  += thr_cnt[t];
        }

        Matrix2D mean_G = Matrix2D::Zero(N + 1, K);
        Matrix2D se_G   = Matrix2D::Zero(N + 1, K);
        for (int j = 1; j < N; ++j) {
            if (count(j) > 0) {
                mean_G.row(j) = sum_G.row(j) / static_cast<double>(count(j));
                if (count(j) > 1) {
                    Matrix2D var = (sum_G2.row(j) / static_cast<double>(count(j)))
                                   - mean_G.row(j).array().square().matrix();
                    se_G.row(j)  = (var.cwiseMax(0.0) / static_cast<double>(count(j))).cwiseSqrt();
                }
            }
        }
        return {mean_G, se_G};
    }

    // ------------------------------------------------------------------
    // AGoS: time-dependent
    // ------------------------------------------------------------------

    template<class UR, class CT>
    std::pair<Matrix3D, Matrix3D> NetworkMCEstimator<UR, CT>::estimate_agos_time_dependent(
        int64_t nb_runs, int64_t nb_generations) {
        if (nb_generations <= 0)
            throw std::invalid_argument("nb_generations must be > 0");

        const int N  = population_size_;
        const int K  = nb_strategies_;
        const int T  = static_cast<int>(nb_generations);

#if defined(_OPENMP) && !defined(_MSC_VER)
        const int nb_threads = omp_get_max_threads();
#else
        const int nb_threads = 1;
#endif

        // thr_sum[thread][generation] is a (N+1, K) matrix.
        std::vector<std::vector<Matrix2D>> thr_sum(
            nb_threads, std::vector<Matrix2D>(T, Matrix2D::Zero(N + 1, K)));
        std::vector<std::vector<Matrix2D>> thr_sum2(
            nb_threads, std::vector<Matrix2D>(T, Matrix2D::Zero(N + 1, K)));
        std::vector<std::vector<VectorXi>> thr_cnt(
            nb_threads, std::vector<VectorXi>(T, VectorXi::Zero(N + 1)));

#if defined(_OPENMP) && !defined(_MSC_VER)
#pragma omp parallel num_threads(nb_threads) default(none) \
    shared(nb_runs, nb_generations, thr_sum, thr_sum2, thr_cnt, N, K, T)
        {
            const int tid = omp_get_thread_num();
#pragma omp for schedule(dynamic)
            for (int64_t run = 0; run < nb_runs; ++run) {
#else
        {
            const int tid = 0;
            for (int64_t run = 0; run < nb_runs; ++run) {
#endif
                std::mt19937_64 gen(egttools::Random::SeedGenerator::getInstance().getSeed());
                CT dyn_cache(cache_size_);
                CT grad_cache(cache_size_);
                VectorXui nbuf  = VectorXui::Zero(K);
                VectorXui nbuf2 = VectorXui::Zero(K);
                UR local_rule = update_rule_;

                std::vector<int> population(N);
                VectorXui mean_state = VectorXui::Zero(K);
                std::uniform_int_distribution<int> s_dist(0, K - 1);
                for (int i = 0; i < N; ++i) {
                    population[i] = s_dist(gen);
                    mean_state(population[i]) += 1;
                }

                for (int g = 0; g < T; ++g) {
                    do_generation_(population, mean_state, dyn_cache, nbuf, gen, local_rule, static_cast<int64_t>(g));
                    const int j = static_cast<int>(mean_state(0));
                    if (j > 0 && j < N) {
                        Vector grad = local_rule.compute_exact_gradient(
                            population, network_, game_, grad_cache, nbuf2, K, beta_);
                        thr_sum[tid][g].row(j)  += grad.transpose();
                        thr_sum2[tid][g].row(j) += grad.array().square().matrix().transpose();
                        thr_cnt[tid][g](j)      += 1;
                    }
                }
            }
        }

        // Merge and compute mean/SE
        Matrix3D mean_G_t(T, Matrix2D::Zero(N + 1, K));
        Matrix3D se_G_t(T, Matrix2D::Zero(N + 1, K));

        for (int g = 0; g < T; ++g) {
            Matrix2D sum_G  = Matrix2D::Zero(N + 1, K);
            Matrix2D sum_G2 = Matrix2D::Zero(N + 1, K);
            VectorXi count  = VectorXi::Zero(N + 1);
            for (int t = 0; t < nb_threads; ++t) {
                sum_G  += thr_sum[t][g];
                sum_G2 += thr_sum2[t][g];
                count  += thr_cnt[t][g];
            }
            for (int j = 1; j < N; ++j) {
                if (count(j) > 0) {
                    mean_G_t[g].row(j) = sum_G.row(j) / static_cast<double>(count(j));
                    if (count(j) > 1) {
                        Matrix2D var = (sum_G2.row(j) / static_cast<double>(count(j)))
                                       - mean_G_t[g].row(j).array().square().matrix();
                        se_G_t[g].row(j) = (var.cwiseMax(0.0) / static_cast<double>(count(j))).cwiseSqrt();
                    }
                }
            }
        }
        return {mean_G_t, se_G_t};
    }

}// namespace egttools::FinitePopulations

#endif//EGTTOOLS_FINITEPOPULATIONS_NETWORKMCESTIMATOR_HPP

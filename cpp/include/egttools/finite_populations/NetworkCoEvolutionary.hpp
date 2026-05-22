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
#ifndef EGTTOOLS_FINITEPOPULATIONS_NETWORKCOEVOLUTIONARY_HPP
#define EGTTOOLS_FINITEPOPULATIONS_NETWORKCOEVOLUTIONARY_HPP

#include <egttools/LruCache.hpp>
#include <egttools/SeedGenerator.h>
#include <egttools/Types.h>
#include <egttools/finite_populations/games/AbstractSpatialGame.hpp>
#include <egttools/finite_populations/structure/AbstractNetworkStructure.hpp>
#include <egttools/finite_populations/structure/rewiring/RandomRewiring.hpp>
#include <egttools/finite_populations/update_rules/PairwiseComparison.hpp>

#include <algorithm>
#include <cmath>
#include <functional>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#if defined(_OPENMP)
#include <egttools/OpenMPExtensions.hpp>
#endif

namespace egttools::FinitePopulations {

    /**
     * @brief Monte Carlo estimator for co-evolutionary networks.
     *
     * In each time step, with probability `rewiring_probability`, the selected
     * focal node invokes the RewiringRule to sever one edge and form a new one.
     * With complementary probability `1 - rewiring_probability`, the focal node
     * undergoes a strategy update via the UpdateRule.
     *
     * The rewiring probability models adaptive social ties: players can leave
     * unfavourable connections and seek new partners (Santos et al. 2006,
     * Borges et al. 2023).
     *
     * Unlike NetworkMCEstimator, the topology changes at every step so we
     * maintain a per-run mutable copy of the adjacency list. The topology()
     * accessor returns the initial (fixed) topology; current state is only
     * available through the callback in run_snapshots_full().
     *
     * @tparam UpdateRule    Strategy update policy (PairwiseComparison, BirthDeath, ...).
     * @tparam RewiringRule  Edge rewiring policy (RandomRewiring, HomophilicRewiring, ...).
     * @tparam CacheType     Fitness cache.
     */
    template<class UpdateRule = update_rules::PairwiseComparison,
             class RewiringRule = structure::rewiring::RandomRewiring,
             class CacheType = egttools::Utils::LRUCache<std::string, double>>
    class NetworkCoEvolutionary {
    public:
        using AdjacencyList = structure::AdjacencyList;
        using NodeDictionary = structure::NodeDictionary;
        using AbstractSpatialGame = games::AbstractSpatialGame;

        NetworkCoEvolutionary(AbstractSpatialGame &game,
                              AdjacencyList topology,
                              int nb_strategies,
                              double beta,
                              double mu,
                              double rewiring_probability,
                              int cache_size = 100000,
                              UpdateRule update_rule = UpdateRule{},
                              RewiringRule rewiring_rule = RewiringRule{});

        NetworkCoEvolutionary(AbstractSpatialGame &game,
                              const NodeDictionary &topology,
                              int nb_strategies,
                              double beta,
                              double mu,
                              double rewiring_probability,
                              int cache_size = 100000,
                              UpdateRule update_rule = UpdateRule{},
                              RewiringRule rewiring_rule = RewiringRule{});

        // ------------------------------------------------------------------
        // Numerical estimators
        // ------------------------------------------------------------------

        /**
         * Estimate time-averaged strategy frequencies and mean edge homophily
         * (fraction of edges linking same-strategy nodes) after the transitory.
         *
         * @return {mean_frequencies, se_frequencies, mean_homophily, se_homophily}
         */
        [[nodiscard]] std::tuple<Vector, Vector, double, double>
        estimate_strategy_distribution(int64_t nb_runs, int64_t nb_generations,
                                       int64_t transitory,
                                       double tolerance = 0.0,
                                       int64_t check_every = 0);

        /**
         * Estimate fixation probability of a single invader in a resident population,
         * starting from the initial topology.
         */
        [[nodiscard]] double estimate_fixation_probability(
                int invader, int resident,
                int64_t nb_runs, int64_t nb_generations);

        /**
         * Run a single trajectory and return aggregate strategy counts per generation
         * (after transitory). The topology evolves during the run; only strategy
         * counts are recorded.
         */
        [[nodiscard]] MatrixXui2D run(int64_t nb_generations, int64_t transitory,
                                      const VectorXui &init_state);

        /**
         * Run a single trajectory and call:
         *   strategy_callback(generation, population)       — every snapshot_interval gens
         *   topology_callback(generation, network)          — every topology_interval gens
         *                                                     (0 = never)
         * Both callbacks fire only after the transitory period.
         */
        void run_snapshots(
                int64_t nb_generations, int64_t transitory,
                int64_t snapshot_interval,
                const VectorXui &init_state,
                const std::function<void(int64_t, const std::vector<int> &)> &strategy_callback,
                int64_t topology_interval = 0,
                const std::function<void(int64_t, const AdjacencyList &)> &topology_callback = nullptr);

        // ------------------------------------------------------------------
        // Accessors
        // ------------------------------------------------------------------

        [[nodiscard]] int population_size() const { return population_size_; }
        [[nodiscard]] int nb_strategies() const { return nb_strategies_; }
        [[nodiscard]] double beta() const { return beta_; }
        [[nodiscard]] double mu() const { return mu_; }
        [[nodiscard]] double rewiring_probability() const { return rewiring_probability_; }
        [[nodiscard]] const AdjacencyList &initial_topology() const { return network_; }

        void set_beta(double beta) { beta_ = beta; }
        void set_mu(double mu) { mu_ = mu; }
        void set_rewiring_probability(double p) { rewiring_probability_ = p; }

    private:
        AbstractSpatialGame &game_;
        AdjacencyList network_;// initial topology (immutable across runs)
        int population_size_, nb_strategies_, cache_size_;
        double beta_, mu_, rewiring_probability_;
        UpdateRule update_rule_;
        RewiringRule rewiring_rule_;

        // One asynchronous step: either rewire or strategy update.
        void do_coevolutionary_step_(std::vector<int> &population,
                                     VectorXui &mean_state,
                                     AdjacencyList &current_net,
                                     CacheType &cache,
                                     VectorXui &nbuf,
                                     std::mt19937_64 &gen,
                                     int64_t t = 0);

        void initialize_state_(std::vector<int> &population, VectorXui &mean_state,
                                const VectorXui &target_counts,
                                std::mt19937_64 &gen) const;

        static double edge_homophily_(const std::vector<int> &population,
                                      const AdjacencyList &net);
    };

    // ======================================================================
    // Implementation
    // ======================================================================

    template<class UR, class RR, class CT>
    NetworkCoEvolutionary<UR, RR, CT>::NetworkCoEvolutionary(
            AbstractSpatialGame &game, AdjacencyList topology,
            int nb_strategies, double beta, double mu, double rewiring_probability,
            int cache_size, UR update_rule, RR rewiring_rule)
        : game_(game),
          network_(std::move(topology)),
          population_size_(static_cast<int>(network_.size())),
          nb_strategies_(nb_strategies),
          cache_size_(cache_size),
          beta_(beta),
          mu_(mu),
          rewiring_probability_(rewiring_probability),
          update_rule_(std::move(update_rule)),
          rewiring_rule_(std::move(rewiring_rule)) {
        if (rewiring_probability_ < 0.0 || rewiring_probability_ > 1.0)
            throw std::invalid_argument("rewiring_probability must be in [0, 1]");
    }

    template<class UR, class RR, class CT>
    NetworkCoEvolutionary<UR, RR, CT>::NetworkCoEvolutionary(
            AbstractSpatialGame &game, const NodeDictionary &topology,
            int nb_strategies, double beta, double mu, double rewiring_probability,
            int cache_size, UR update_rule, RR rewiring_rule)
        : NetworkCoEvolutionary(game, structure::dict_to_adjacency_list(topology),
                                nb_strategies, beta, mu, rewiring_probability,
                                cache_size, std::move(update_rule), std::move(rewiring_rule)) {}

    // ------------------------------------------------------------------
    // Private: one co-evolutionary step
    // ------------------------------------------------------------------

    template<class UR, class RR, class CT>
    void NetworkCoEvolutionary<UR, RR, CT>::do_coevolutionary_step_(
            std::vector<int> &population, VectorXui &mean_state,
            AdjacencyList &current_net, CT &cache, VectorXui &nbuf,
            std::mt19937_64 &gen, int64_t t) {
        std::uniform_real_distribution<double> real_dist(0.0, 1.0);

        if (real_dist(gen) < rewiring_probability_) {
            // Rewiring step: pick a random focal node
            std::uniform_int_distribution<int> node_dist(0, population_size_ - 1);
            int focal = node_dist(gen);
            rewiring_rule_.rewire(focal, current_net, population, gen);
        } else {
            // Strategy update step (uses current_net, not network_)
            update_rule_.step(population, mean_state, current_net, game_, cache,
                              nbuf, nb_strategies_, beta_, mu_, gen, t);
        }
    }

    // ------------------------------------------------------------------
    // Fixation probability
    // ------------------------------------------------------------------

    template<class UR, class RR, class CT>
    double NetworkCoEvolutionary<UR, RR, CT>::estimate_fixation_probability(
            int invader, int resident, int64_t nb_runs, int64_t nb_generations) {
        if (invader < 0 || invader >= nb_strategies_)
            throw std::invalid_argument("invader must be in [0, nb_strategies)");
        if (resident < 0 || resident >= nb_strategies_)
            throw std::invalid_argument("resident must be in [0, nb_strategies)");
        if (invader == resident)
            throw std::invalid_argument("invader and resident must differ");

        long int fixations = 0, extinctions = 0;

#if defined(_OPENMP) && !defined(_MSC_VER)
#pragma omp parallel for reduction(+ : fixations, extinctions) default(none) \
    shared(invader, resident, nb_runs, nb_generations)
#endif
        for (int64_t run = 0; run < nb_runs; ++run) {
            std::mt19937_64 gen(egttools::Random::SeedGenerator::getInstance().getSeed());
            CT cache(cache_size_);
            VectorXui nbuf = VectorXui::Zero(nb_strategies_);
            UR local_update = update_rule_;
            RR local_rewire = rewiring_rule_;
            AdjacencyList current_net = network_;// mutable copy per run

            std::vector<int> population(population_size_, resident);
            VectorXui mean_state = VectorXui::Zero(nb_strategies_);
            mean_state(resident) = population_size_;

            std::uniform_int_distribution<int> node_dist(0, population_size_ - 1);
            int seed_node = node_dist(gen);
            mean_state(resident) -= 1;
            mean_state(invader) += 1;
            population[seed_node] = invader;

            std::uniform_real_distribution<double> real_dist(0.0, 1.0);

            for (int64_t t = 0; t < nb_generations; ++t) {
                for (int step = 0; step < population_size_; ++step) {
                    int64_t ts = t * population_size_ + step;
                    if (real_dist(gen) < rewiring_probability_) {
                        std::uniform_int_distribution<int> nd(0, population_size_ - 1);
                        int focal = nd(gen);
                        local_rewire.rewire(focal, current_net, population, gen);
                    } else {
                        local_update.step(population, mean_state, current_net, game_, cache,
                                          nbuf, nb_strategies_, beta_, mu_, gen, ts);
                    }
                }
                if (static_cast<int>(mean_state(invader)) == population_size_) { ++fixations; break; }
                if (mean_state(invader) == 0u) { ++extinctions; break; }
            }
        }

        long int decided = fixations + extinctions;
        if (decided == 0) return 0.0;
        return static_cast<double>(fixations) / static_cast<double>(decided);
    }

    // ------------------------------------------------------------------
    // Strategy distribution + homophily
    // ------------------------------------------------------------------

    template<class UR, class RR, class CT>
    std::tuple<Vector, Vector, double, double>
    NetworkCoEvolutionary<UR, RR, CT>::estimate_strategy_distribution(
            int64_t nb_runs, int64_t nb_generations, int64_t transitory,
            double tolerance, int64_t check_every) {
        if (transitory >= nb_generations)
            throw std::invalid_argument("transitory must be < nb_generations");

        const int64_t counting_gens = nb_generations - transitory;
        const int64_t batch = (check_every > 0) ? check_every
                                                 : std::max<int64_t>(1, nb_runs / 10);

        Vector sum_freq = Vector::Zero(nb_strategies_);
        Vector sum_freq2 = Vector::Zero(nb_strategies_);
        double sum_hom = 0.0, sum_hom2 = 0.0;
        Vector prev_estimate = Vector::Zero(nb_strategies_);
        int64_t runs_done = 0;

        while (runs_done < nb_runs) {
            const int64_t this_batch = std::min(batch, nb_runs - runs_done);
            Vector batch_sum = Vector::Zero(nb_strategies_);
            double batch_hom = 0.0;

#if defined(_OPENMP) && !defined(_MSC_VER)
#pragma omp parallel for reduction(+ : batch_sum, batch_hom) default(none) \
    shared(this_batch, nb_generations, transitory, counting_gens)
#endif
            for (int64_t run = 0; run < this_batch; ++run) {
                std::mt19937_64 gen(egttools::Random::SeedGenerator::getInstance().getSeed());
                CT cache(cache_size_);
                VectorXui nbuf = VectorXui::Zero(nb_strategies_);
                UR local_update = update_rule_;
                RR local_rewire = rewiring_rule_;
                AdjacencyList current_net = network_;

                std::vector<int> population(population_size_);
                VectorXui mean_state = VectorXui::Zero(nb_strategies_);
                std::uniform_int_distribution<int> s_dist(0, nb_strategies_ - 1);
                for (int i = 0; i < population_size_; ++i) {
                    population[i] = s_dist(gen);
                    mean_state(population[i]) += 1;
                }

                std::uniform_real_distribution<double> real_dist(0.0, 1.0);
                Vector run_freq = Vector::Zero(nb_strategies_);
                double run_hom = 0.0;

                for (int64_t g = 0; g < nb_generations; ++g) {
                    for (int step = 0; step < population_size_; ++step) {
                        int64_t ts = g * population_size_ + step;
                        if (real_dist(gen) < rewiring_probability_) {
                            std::uniform_int_distribution<int> nd(0, population_size_ - 1);
                            local_rewire.rewire(nd(gen), current_net, population, gen);
                        } else {
                            local_update.step(population, mean_state, current_net, game_, cache,
                                              nbuf, nb_strategies_, beta_, mu_, gen, ts);
                        }
                    }
                    if (g >= transitory) {
                        for (int s = 0; s < nb_strategies_; ++s)
                            run_freq(s) += static_cast<double>(mean_state(s)) / population_size_;
                        run_hom += edge_homophily_(population, current_net);
                    }
                }
                batch_sum += run_freq / static_cast<double>(counting_gens);
                batch_hom += run_hom / static_cast<double>(counting_gens);
            }

            sum_freq += batch_sum;
            sum_freq2 += batch_sum.array().square().matrix();
            sum_hom += batch_hom;
            sum_hom2 += batch_hom * batch_hom;
            runs_done += this_batch;

            if (tolerance > 0.0 && runs_done > 0) {
                Vector cur = sum_freq / static_cast<double>(runs_done);
                if ((cur - prev_estimate).lpNorm<1>() < tolerance) break;
                prev_estimate = cur;
            }
        }

        double n = static_cast<double>(runs_done);
        Vector mean_freq = sum_freq / n;
        Vector var_freq = (sum_freq2 / n) - mean_freq.array().square().matrix();
        var_freq = var_freq.cwiseMax(0.0);
        Vector se_freq = (var_freq / n).cwiseSqrt();

        double mean_hom = sum_hom / n;
        double var_hom = std::max(0.0, sum_hom2 / n - mean_hom * mean_hom);
        double se_hom = std::sqrt(var_hom / n);

        return {mean_freq, se_freq, mean_hom, se_hom};
    }

    // ------------------------------------------------------------------
    // Trajectory: aggregate counts
    // ------------------------------------------------------------------

    template<class UR, class RR, class CT>
    MatrixXui2D NetworkCoEvolutionary<UR, RR, CT>::run(
            int64_t nb_generations, int64_t transitory, const VectorXui &init_state) {
        if (transitory >= nb_generations)
            throw std::invalid_argument("transitory must be < nb_generations");

        std::mt19937_64 gen(egttools::Random::SeedGenerator::getInstance().getSeed());
        CT cache(cache_size_);
        VectorXui nbuf = VectorXui::Zero(nb_strategies_);
        AdjacencyList current_net = network_;

        std::vector<int> population(population_size_);
        VectorXui mean_state(nb_strategies_);
        initialize_state_(population, mean_state, init_state, gen);

        const int64_t recording_gens = nb_generations - transitory;
        MatrixXui2D trajectory(recording_gens, nb_strategies_);

        std::uniform_real_distribution<double> real_dist(0.0, 1.0);
        int64_t row = 0;

        for (int64_t g = 0; g < nb_generations; ++g) {
            for (int step = 0; step < population_size_; ++step) {
                int64_t ts = g * population_size_ + step;
                if (real_dist(gen) < rewiring_probability_) {
                    std::uniform_int_distribution<int> nd(0, population_size_ - 1);
                    rewiring_rule_.rewire(nd(gen), current_net, population, gen);
                } else {
                    update_rule_.step(population, mean_state, current_net, game_, cache,
                                      nbuf, nb_strategies_, beta_, mu_, gen, ts);
                }
            }
            if (g >= transitory) trajectory.row(row++) = mean_state;
        }
        return trajectory;
    }

    // ------------------------------------------------------------------
    // Trajectory: snapshot callbacks
    // ------------------------------------------------------------------

    template<class UR, class RR, class CT>
    void NetworkCoEvolutionary<UR, RR, CT>::run_snapshots(
            int64_t nb_generations, int64_t transitory,
            int64_t snapshot_interval, const VectorXui &init_state,
            const std::function<void(int64_t, const std::vector<int> &)> &strategy_callback,
            int64_t topology_interval,
            const std::function<void(int64_t, const AdjacencyList &)> &topology_callback) {
        if (transitory >= nb_generations)
            throw std::invalid_argument("transitory must be < nb_generations");
        if (snapshot_interval <= 0)
            throw std::invalid_argument("snapshot_interval must be > 0");

        std::mt19937_64 gen(egttools::Random::SeedGenerator::getInstance().getSeed());
        CT cache(cache_size_);
        VectorXui nbuf = VectorXui::Zero(nb_strategies_);
        AdjacencyList current_net = network_;

        std::vector<int> population(population_size_);
        VectorXui mean_state(nb_strategies_);
        initialize_state_(population, mean_state, init_state, gen);

        std::uniform_real_distribution<double> real_dist(0.0, 1.0);
        int64_t post_t = 0;

        for (int64_t g = 0; g < nb_generations; ++g) {
            for (int step = 0; step < population_size_; ++step) {
                int64_t ts = g * population_size_ + step;
                if (real_dist(gen) < rewiring_probability_) {
                    std::uniform_int_distribution<int> nd(0, population_size_ - 1);
                    rewiring_rule_.rewire(nd(gen), current_net, population, gen);
                } else {
                    update_rule_.step(population, mean_state, current_net, game_, cache,
                                      nbuf, nb_strategies_, beta_, mu_, gen, ts);
                }
            }
            if (g >= transitory) {
                if (post_t % snapshot_interval == 0)
                    strategy_callback(post_t, population);
                if (topology_interval > 0 && topology_callback
                    && post_t % topology_interval == 0)
                    topology_callback(post_t, current_net);
                ++post_t;
            }
        }
    }

    // ------------------------------------------------------------------
    // Private helpers
    // ------------------------------------------------------------------

    template<class UR, class RR, class CT>
    void NetworkCoEvolutionary<UR, RR, CT>::initialize_state_(
            std::vector<int> &population, VectorXui &mean_state,
            const VectorXui &target_counts, std::mt19937_64 &gen) const {
        mean_state = target_counts;
        int idx = 0;
        for (int s = 0; s < nb_strategies_; ++s)
            for (Eigen::Index c = 0; c < static_cast<Eigen::Index>(target_counts(s)); ++c)
                population[idx++] = s;
        std::shuffle(population.begin(), population.end(), gen);
    }

    template<class UR, class RR, class CT>
    double NetworkCoEvolutionary<UR, RR, CT>::edge_homophily_(
            const std::vector<int> &population, const AdjacencyList &net) {
        int same = 0, total = 0;
        for (int i = 0; i < static_cast<int>(net.size()); ++i) {
            for (int j : net[i]) {
                if (j > i) {// count each undirected edge once
                    ++total;
                    if (population[i] == population[j]) ++same;
                }
            }
        }
        return total > 0 ? static_cast<double>(same) / total : 0.0;
    }

}// namespace egttools::FinitePopulations

#endif//EGTTOOLS_FINITEPOPULATIONS_NETWORKCOEVOLUTIONARY_HPP

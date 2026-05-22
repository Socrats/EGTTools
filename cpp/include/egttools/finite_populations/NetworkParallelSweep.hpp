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
#ifndef EGTTOOLS_FINITEPOPULATIONS_NETWORKPARALLELSWEEP_HPP
#define EGTTOOLS_FINITEPOPULATIONS_NETWORKPARALLELSWEEP_HPP

#include <egttools/LruCache.hpp>
#include <egttools/SeedGenerator.h>
#include <egttools/Types.h>
#include <egttools/finite_populations/games/AbstractSpatialGame.hpp>
#include <egttools/finite_populations/structure/AbstractNetworkStructure.hpp>
#include <egttools/finite_populations/update_rules/PairwiseComparison.hpp>

#include <algorithm>
#include <random>
#include <vector>

#if defined(_OPENMP)
#include <egttools/OpenMPExtensions.hpp>
#endif

namespace egttools::FinitePopulations {

    /**
     * @brief Parallel parameter sweep over a grid of games × topologies (OpenMP).
     *
     * Runs nb_runs independent trajectories for each (game, topology) combination
     * using OpenMP threads. Topologies are held by const reference — never copied —
     * so even an O(N²) complete-graph adjacency list is stored only once.
     *
     * Each thread owns its RNG, LRU cache, and working buffers. Games must be C++
     * instances (e.g. NormalFormNetworkGame): their calculate_fitness is called
     * without holding the GIL, so Python subclasses are not safe here.
     *
     * @tparam UpdateRule  Policy struct providing step() with a synchronous flag.
     * @tparam CacheType   Fitness cache type.
     *
     * @param games         One AbstractSpatialGame* per parameter combination.
     * @param betas         One beta (selection intensity / D_>) per game.
     * @param topologies    Network adjacency lists. Shared read-only across threads.
     * @param nb_strategies Number of strategies.
     * @param mu            Mutation probability per time step.
     * @param nb_runs       Independent runs per (game, topology) combination.
     * @param avg_gens      Generations to average over after the transitory.
     * @param transitory    Burn-in generations (discarded).
     * @param init_state    Strategy counts at t=0 (must sum to population_size).
     * @param cache_size    Per-thread LRU cache capacity.
     * @param update_rule   Update rule instance (default-constructed for stateless rules).
     *
     * @return Matrix2D of shape (n_games, n_topologies) with mean strategy-0 fraction
     *         (cooperation frequency) averaged over runs × averaging generations.
     */
    template<class UpdateRule = update_rules::PairwiseComparison,
             class CacheType = egttools::Utils::LRUCache<std::string, double>>
    egttools::Matrix2D run_network_sweep(
        const std::vector<games::AbstractSpatialGame *> &games,
        const std::vector<double>                       &betas,
        const std::vector<structure::AdjacencyList>     &topologies,
        int              nb_strategies,
        double           mu,
        int64_t          nb_runs,
        int64_t          avg_gens,
        int64_t          transitory,
        const VectorXui &init_state,
        int              cache_size  = 100000,
        UpdateRule       update_rule = UpdateRule{}) {

        const int n_games = static_cast<int>(games.size());
        const int n_topos = static_cast<int>(topologies.size());

        egttools::Matrix2D result = egttools::Matrix2D::Zero(n_games, n_topos);

#if defined(_OPENMP) && !defined(_MSC_VER)
#pragma omp parallel for collapse(2) default(none)              \
    shared(games, betas, topologies, nb_strategies, mu,         \
           nb_runs, avg_gens, transitory, init_state,           \
           cache_size, update_rule, result, n_games, n_topos)
#endif
        for (int gi = 0; gi < n_games; ++gi) {
            for (int ti = 0; ti < n_topos; ++ti) {

                const structure::AdjacencyList &topo    = topologies[ti];
                const int                       pop_size = static_cast<int>(topo.size());
                const double                    beta     = betas[static_cast<std::size_t>(gi)];

                // Thread-local simulation state
                std::mt19937_64 gen(egttools::Random::SeedGenerator::getInstance().getSeed());
                CacheType       cache(cache_size);
                VectorXui       nbuf(nb_strategies);
                UpdateRule      local_rule = update_rule;
                std::vector<int> population(pop_size);
                VectorXui        mean_state(nb_strategies);

                double coop_sum = 0.0;

                for (int64_t run = 0; run < nb_runs; ++run) {
                    // --- Initialise population ---
                    int node = 0;
                    for (int s = 0; s < nb_strategies; ++s)
                        for (uint64_t k = 0; k < init_state(s); ++k)
                            population[static_cast<std::size_t>(node++)] = s;
                    std::shuffle(population.begin(), population.end(), gen);
                    mean_state.setZero();
                    for (int s : population) mean_state(static_cast<Eigen::Index>(s))++;

                    // --- Run generations ---
                    double run_coop = 0.0;
                    for (int64_t g = 0; g < transitory + avg_gens; ++g) {
                        if constexpr (UpdateRule::synchronous) {
                            local_rule.step(population, mean_state, topo,
                                            *games[static_cast<std::size_t>(gi)],
                                            cache, nbuf, nb_strategies, beta, mu, gen, g);
                        } else {
                            for (int s = 0; s < pop_size; ++s)
                                local_rule.step(population, mean_state, topo,
                                                *games[static_cast<std::size_t>(gi)],
                                                cache, nbuf, nb_strategies, beta, mu, gen,
                                                g * static_cast<int64_t>(pop_size) + s);
                        }
                        if (g >= transitory)
                            run_coop += static_cast<double>(mean_state(0));
                    }
                    coop_sum += run_coop / (static_cast<double>(avg_gens) *
                                            static_cast<double>(pop_size));
                }

                result(gi, ti) = coop_sum / static_cast<double>(nb_runs);
            }
        }

        return result;
    }

}// namespace egttools::FinitePopulations

#endif//EGTTOOLS_FINITEPOPULATIONS_NETWORKPARALLELSWEEP_HPP

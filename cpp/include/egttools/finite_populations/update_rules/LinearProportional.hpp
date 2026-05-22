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
#ifndef EGTTOOLS_FINITEPOPULATIONS_UPDATERULES_LINEARPROPORTIONAL_HPP
#define EGTTOOLS_FINITEPOPULATIONS_UPDATERULES_LINEARPROPORTIONAL_HPP

#include <egttools/Types.h>
#include <egttools/finite_populations/update_rules/AbstractUpdateRule.hpp>
#include <algorithm>
#include <random>
#include <string>
#include <vector>

namespace egttools::FinitePopulations::update_rules {

    /**
     * @brief Linear-proportional imitation rule with synchronous updating.
     *
     * Implements the update rule from Santos, Pacheco & Lenaerts (2006) PNAS:
     *
     *   For each node x simultaneously:
     *     1. Compute accumulated payoff P_x by playing the spatial game against
     *        all neighbours (Phase 1, done simultaneously for all nodes).
     *     2. Pick a random neighbour y of x.
     *     3. If P_y > P_x: copy y's strategy with probability
     *          p = (P_y - P_x) / (k_> * D_>)
     *        where k_> = max(degree(x), degree(y)) and D_> (passed as `beta`)
     *        is the maximum possible payoff difference = max(T,1) - min(S,0).
     *     4. Else: keep current strategy.
     *   All updates applied simultaneously at the end of Phase 2.
     *
     * NOTE: `beta` is reused as D_> (payoff normalisation constant). Callers
     * constructing a NetworkMCEstimator for this rule should pass
     *   beta = max(T, 1.0) - min(S, 0.0)
     * as the selection intensity parameter.
     *
     * One call to step() constitutes one complete synchronous generation.
     */
    struct LinearProportional {

        static constexpr bool synchronous = true;

        [[nodiscard]] static std::string name() { return "LinearProportional"; }

        // ---------- single synchronous generation ----------

        template<class GameType, class CacheType>
        static void step(std::vector<int> &population,
                         VectorXui &mean_state,
                         const AdjacencyList &network,
                         GameType &game,
                         CacheType & /*cache*/,
                         VectorXui &nbuf,
                         int nb_strategies,
                         double beta,   // D_> = max(T,1) - min(S,0)
                         double mu,
                         std::mt19937_64 &gen,
                         int64_t /*t*/ = 0) {
            const int N = static_cast<int>(population.size());

            std::uniform_real_distribution<double> real_dist(0.0, 1.0);
            std::uniform_int_distribution<int> strat_dist(0, nb_strategies - 1);

            // Phase 1: compute accumulated payoffs for all nodes simultaneously.
            std::vector<double> payoffs(N);
            for (int x = 0; x < N; ++x) {
                nbuf.setZero();
                for (int nb : network[x]) nbuf(population[nb]) += 1;
                payoffs[x] = game.calculate_fitness(population[x], nbuf);
            }

            // Phase 2: determine new strategies for all nodes simultaneously.
            std::vector<int> new_population = population;

            for (int x = 0; x < N; ++x) {
                // Mutation
                if (real_dist(gen) < mu) {
                    int new_s = strat_dist(gen);
                    while (new_s == population[x]) new_s = strat_dist(gen);
                    new_population[x] = new_s;
                    continue;
                }

                const auto &neighbors = network[x];
                if (neighbors.empty()) continue;

                std::uniform_int_distribution<int> nb_dist(0, static_cast<int>(neighbors.size()) - 1);
                const int y = neighbors[nb_dist(gen)];

                if (payoffs[y] > payoffs[x]) {
                    const int k_greater = std::max(
                        static_cast<int>(network[x].size()),
                        static_cast<int>(network[y].size()));
                    const double denom = static_cast<double>(k_greater) * beta;
                    const double prob  = (denom > 0.0)
                                         ? std::min(1.0, (payoffs[y] - payoffs[x]) / denom)
                                         : 1.0;
                    if (real_dist(gen) < prob) {
                        new_population[x] = population[y];
                    }
                }
            }

            // Phase 3: apply all updates and recompute mean_state.
            mean_state.setZero();
            for (int x = 0; x < N; ++x) {
                population[x] = new_population[x];
                mean_state(population[x]) += 1;
            }
        }

        // ---------- numerically exact gradient (stub — not applicable for sync rule) ----------

        template<class GameType, class CacheType>
        static Vector compute_exact_gradient(const std::vector<int> & /*population*/,
                                             const AdjacencyList & /*network*/,
                                             GameType & /*game*/,
                                             CacheType & /*cache*/,
                                             VectorXui & /*nbuf*/,
                                             int nb_strategies,
                                             double /*beta*/,
                                             int64_t /*t*/ = 0) {
            return Vector::Zero(nb_strategies);
        }
    };

}// namespace egttools::FinitePopulations::update_rules

#endif//EGTTOOLS_FINITEPOPULATIONS_UPDATERULES_LINEARPROPORTIONAL_HPP

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
#ifndef EGTTOOLS_FINITEPOPULATIONS_UPDATERULES_PAIRWISECOMPARISON_HPP
#define EGTTOOLS_FINITEPOPULATIONS_UPDATERULES_PAIRWISECOMPARISON_HPP

#include <egttools/Types.h>
#include <egttools/finite_populations/Utils.hpp>
#include <egttools/finite_populations/update_rules/AbstractUpdateRule.hpp>
#include <random>
#include <sstream>
#include <string>
#include <vector>

namespace egttools::FinitePopulations::update_rules {

    /**
     * @brief Pairwise-comparison (Fermi imitation) update rule.
     *
     * At each time step:
     *   1. Select a focal node i uniformly at random from {0, ..., N-1}.
     *   2. With probability mu, the focal node adopts a uniformly random
     *      different strategy (mutation).
     *   3. Otherwise, select a neighbour j of i uniformly at random.
     *      If s_i != s_j, the focal node copies j's strategy with probability
     *        p = 1 / (1 + exp(beta * (f_i - f_j)))  [Fermi function]
     *
     * Numerically exact gradient formula (Pinheiro, Pacheco & Santos 2012):
     *
     *   G(s_k) = (1/N) sum_i (1/d_i) [
     *       sum_{j in N(i), s_j=k, s_i!=k}  Fermi(beta, f_i, f_j)   [T+]
     *     - sum_{j in N(i), s_j!=k, s_i=k}  Fermi(beta, f_i, f_j)   [T-]
     *   ]
     *
     * where d_i = degree of node i, N = population size,
     * and 1 generation = N asynchronous time-steps.
     */
    struct PairwiseComparison {

        static constexpr bool synchronous = false;

        [[nodiscard]] static std::string name() { return "PairwiseComparison"; }

        // ---------- fitness helper (shared by step and compute_exact_gradient) ----------

        template<class GameType, class CacheType>
        static double fitness(int node,
                              const std::vector<int> &population,
                              const AdjacencyList &network,
                              GameType &game,
                              CacheType &cache,
                              VectorXui &nbuf) {
            nbuf.setZero();
            for (int nb : network[node]) nbuf(population[nb]) += 1;

            std::ostringstream oss;
            oss << nbuf;
            std::string key = std::to_string(population[node]) + oss.str();

            if (!cache.exists(key)) {
                double f = game.calculate_fitness(population[node], nbuf);
                cache.insert(key, f);
                return f;
            }
            return cache.get(key);
        }

        // ---------- single update step ----------

        template<class GameType, class CacheType>
        static void step(std::vector<int> &population,
                         VectorXui &mean_state,
                         const AdjacencyList &network,
                         GameType &game,
                         CacheType &cache,
                         VectorXui &nbuf,
                         int nb_strategies,
                         double beta,
                         double mu,
                         std::mt19937_64 &gen,
                         int64_t /*t*/ = 0) {
            const int N = static_cast<int>(population.size());

            std::uniform_int_distribution<int> node_dist(0, N - 1);
            std::uniform_real_distribution<double> real_dist(0.0, 1.0);
            std::uniform_int_distribution<int> strat_dist(0, nb_strategies - 1);

            int focal = node_dist(gen);

            // Mutation
            if (real_dist(gen) < mu) {
                int new_s = strat_dist(gen);
                while (new_s == population[focal]) new_s = strat_dist(gen);
                mean_state(population[focal]) -= 1;
                mean_state(new_s) += 1;
                population[focal] = new_s;
                return;
            }

            const auto &neighbors = network[focal];
            if (neighbors.empty()) return;

            std::uniform_int_distribution<int> nb_dist(0, static_cast<int>(neighbors.size()) - 1);
            int neighbor = neighbors[nb_dist(gen)];

            if (population[focal] == population[neighbor]) return;

            double ff = fitness(focal, population, network, game, cache, nbuf);
            double fn = fitness(neighbor, population, network, game, cache, nbuf);

            if (real_dist(gen) < egttools::FinitePopulations::fermi(beta, ff, fn)) {
                mean_state(population[focal]) -= 1;
                mean_state(population[neighbor]) += 1;
                population[focal] = population[neighbor];
            }
        }

        // ---------- numerically exact gradient ----------

        template<class GameType, class CacheType>
        static Vector compute_exact_gradient(const std::vector<int> &population,
                                             const AdjacencyList &network,
                                             GameType &game,
                                             CacheType &cache,
                                             VectorXui &nbuf,
                                             int nb_strategies,
                                             double beta,
                                             int64_t /*t*/ = 0) {
            const int N = static_cast<int>(population.size());
            Vector transitions_plus = Vector::Zero(nb_strategies);
            Vector transitions_minus = Vector::Zero(nb_strategies);
            Vector t_prob = Vector::Zero(nb_strategies);

            for (int i = 0; i < N; ++i) {
                const auto degree = static_cast<double>(network[i].size());
                if (degree == 0.0) continue;

                double fi = fitness(i, population, network, game, cache, nbuf);
                double t_unconditional = 0.0;
                t_prob.setZero();

                for (int j : network[i]) {
                    if (population[j] == population[i]) continue;
                    double fj = fitness(j, population, network, game, cache, nbuf);
                    double p = egttools::FinitePopulations::fermi(beta, fi, fj);
                    t_unconditional += p;
                    t_prob(population[j]) += p;
                }

                transitions_plus += t_prob / degree;
                transitions_minus(population[i]) += t_unconditional / degree;
            }

            return (transitions_plus - transitions_minus) / N;
        }
    };

}// namespace egttools::FinitePopulations::update_rules

#endif//EGTTOOLS_FINITEPOPULATIONS_UPDATERULES_PAIRWISECOMPARISON_HPP

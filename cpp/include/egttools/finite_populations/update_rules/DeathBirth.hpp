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
#ifndef EGTTOOLS_FINITEPOPULATIONS_UPDATERULES_DEATHBIRTH_HPP
#define EGTTOOLS_FINITEPOPULATIONS_UPDATERULES_DEATHBIRTH_HPP

#include <egttools/Types.h>
#include <egttools/finite_populations/Utils.hpp>
#include <egttools/finite_populations/update_rules/AbstractUpdateRule.hpp>
#include <random>
#include <sstream>
#include <string>
#include <vector>

namespace egttools::FinitePopulations::update_rules {

    /**
     * @brief Death-Birth (DB) update rule.
     *
     * At each time step:
     *   1. Select a node i to die uniformly at random.
     *   2. Select a neighbour j of i to reproduce, proportional to fitness:
     *      p(j reproduces) = exp(beta * f_j) / sum_{k in N(i)} exp(beta * f_k)
     *   3. j's strategy fills i's vacated spot.
     *
     * Mutation: with probability mu, the dead node is replaced by a uniformly
     * random strategy instead.
     *
     * DB suppresses selection on heterogeneous networks where BD amplifies it,
     * making it valuable for studying whether network structure promotes or
     * hinders cooperation.
     *
     * Numerically exact gradient formula for DB:
     *
     *   T+(k) = (1/N) sum_{i: s_i != k}
     *                 sum_{j in N(i): s_j = k} exp(beta*f_j) / W_i
     *   T-(k) = (1/N) sum_{i: s_i = k}
     *                 sum_{j in N(i): s_j != k} exp(beta*f_j) / W_i
     *
     *   where W_i = sum_{j in N(i)} exp(beta * f_j)
     *
     *   G(s_k) = T+(k) - T-(k)
     */
    struct DeathBirth {

        static constexpr bool synchronous = false;

        [[nodiscard]] static std::string name() { return "DeathBirth"; }

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

            int victim = node_dist(gen);

            // Mutation
            if (real_dist(gen) < mu) {
                int new_s = strat_dist(gen);
                while (new_s == population[victim]) new_s = strat_dist(gen);
                mean_state(population[victim]) -= 1;
                mean_state(new_s) += 1;
                population[victim] = new_s;
                return;
            }

            const auto &neighbors = network[victim];
            if (neighbors.empty()) return;

            // Sample parent from neighbours proportional to exp(beta * f)
            std::vector<double> nb_weights(neighbors.size());
            double total = 0.0;
            for (size_t k = 0; k < neighbors.size(); ++k) {
                nb_weights[k] = std::exp(beta * fitness(neighbors[k], population, network, game, cache, nbuf));
                total += nb_weights[k];
            }

            double r = real_dist(gen) * total;
            int parent = neighbors[0];
            double cumsum = 0.0;
            for (size_t k = 0; k < neighbors.size(); ++k) {
                cumsum += nb_weights[k];
                if (cumsum >= r) { parent = neighbors[k]; break; }
            }

            if (population[victim] == population[parent]) return;

            mean_state(population[victim]) -= 1;
            mean_state(population[parent]) += 1;
            population[victim] = population[parent];
        }

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

            for (int i = 0; i < N; ++i) {
                const auto &neighbors = network[i];
                if (neighbors.empty()) continue;

                // Compute fitness weights for each neighbour
                double W_i = 0.0;
                std::vector<double> wj(neighbors.size());
                for (size_t k = 0; k < neighbors.size(); ++k) {
                    wj[k] = std::exp(beta * fitness(neighbors[k], population, network, game, cache, nbuf));
                    W_i += wj[k];
                }
                if (W_i == 0.0) continue;

                for (size_t k = 0; k < neighbors.size(); ++k) {
                    int j = neighbors[k];
                    if (population[j] == population[i]) continue;
                    double p = wj[k] / W_i / N;
                    // j reproduces into i's spot
                    transitions_plus(population[j]) += p;
                    transitions_minus(population[i]) += p;
                }
            }

            return transitions_plus - transitions_minus;
        }
    };

}// namespace egttools::FinitePopulations::update_rules

#endif//EGTTOOLS_FINITEPOPULATIONS_UPDATERULES_DEATHBIRTH_HPP

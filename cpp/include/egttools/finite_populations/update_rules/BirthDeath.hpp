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
#ifndef EGTTOOLS_FINITEPOPULATIONS_UPDATERULES_BIRTHDEATH_HPP
#define EGTTOOLS_FINITEPOPULATIONS_UPDATERULES_BIRTHDEATH_HPP

#include <egttools/Types.h>
#include <egttools/finite_populations/Utils.hpp>
#include <egttools/finite_populations/update_rules/AbstractUpdateRule.hpp>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <vector>

namespace egttools::FinitePopulations::update_rules {

    /**
     * @brief Birth-Death (BD) update rule.
     *
     * At each time step:
     *   1. Select a node i to reproduce with probability proportional to its fitness.
     *      (With weak selection: proportional to 1 + delta * f_i, delta -> 0.)
     *      With strong selection (beta > 0): proportional to exp(beta * f_i).
     *   2. Select a neighbour j of i uniformly at random; j is replaced by a copy of i.
     *
     * Mutation is applied before birth-selection: with probability mu each node
     * is assigned a random different strategy instead of competing normally.
     *
     * BD rule is known to amplify selection on star graphs and suppress it on cycles,
     * in contrast to DB.
     *
     * Numerically exact gradient formula for BD:
     *
     *   Let w_i = exp(beta * f_i) (unnormalized fitness weights).
     *   W = sum_j w_j (total weight).
     *
     *   T+(k) = sum_{i: s_i != k} (w_i / W) * |{j in N(i) : s_j = k}| / d_i
     *   T-(k) = sum_{i: s_i = k} (w_i / W) * |{j in N(i) : s_j != k}| / d_i
     *
     *   G(s_k) = T+(k) - T-(k)
     */
    struct BirthDeath {

        [[nodiscard]] static std::string name() { return "BirthDeath"; }

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
            std::uniform_real_distribution<double> real_dist(0.0, 1.0);
            std::uniform_int_distribution<int> strat_dist(0, nb_strategies - 1);

            // Mutation: with prob mu each step, a random node mutates
            if (real_dist(gen) < mu) {
                std::uniform_int_distribution<int> node_dist(0, N - 1);
                int focal = node_dist(gen);
                int new_s = strat_dist(gen);
                while (new_s == population[focal]) new_s = strat_dist(gen);
                mean_state(population[focal]) -= 1;
                mean_state(new_s) += 1;
                population[focal] = new_s;
                return;
            }

            // Compute fitness weights exp(beta * f_i)
            std::vector<double> weights(N);
            double total_weight = 0.0;
            for (int i = 0; i < N; ++i) {
                weights[i] = std::exp(beta * fitness(i, population, network, game, cache, nbuf));
                total_weight += weights[i];
            }

            // Sample parent proportional to fitness
            double r = real_dist(gen) * total_weight;
            int parent = 0;
            double cumsum = 0.0;
            for (int i = 0; i < N; ++i) {
                cumsum += weights[i];
                if (cumsum >= r) { parent = i; break; }
            }

            const auto &neighbors = network[parent];
            if (neighbors.empty()) return;

            // Replace a random neighbour
            std::uniform_int_distribution<int> nb_dist(0, static_cast<int>(neighbors.size()) - 1);
            int victim = neighbors[nb_dist(gen)];

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

            std::vector<double> w(N);
            double W = 0.0;
            for (int i = 0; i < N; ++i) {
                w[i] = std::exp(beta * fitness(i, population, network, game, cache, nbuf));
                W += w[i];
            }

            Vector transitions_plus = Vector::Zero(nb_strategies);
            Vector transitions_minus = Vector::Zero(nb_strategies);

            for (int i = 0; i < N; ++i) {
                const auto degree = static_cast<double>(network[i].size());
                if (degree == 0.0) continue;
                double wi_over_W = w[i] / W;

                for (int j : network[i]) {
                    if (population[j] == population[i]) continue;
                    // i reproduces into j's spot
                    transitions_plus(population[i]) += wi_over_W / degree;
                    transitions_minus(population[j]) += wi_over_W / degree;
                }
            }

            return transitions_plus - transitions_minus;
        }
    };

}// namespace egttools::FinitePopulations::update_rules

#endif//EGTTOOLS_FINITEPOPULATIONS_UPDATERULES_BIRTHDEATH_HPP

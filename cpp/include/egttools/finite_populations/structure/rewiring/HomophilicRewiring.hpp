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
#ifndef EGTTOOLS_FINITEPOPULATIONS_STRUCTURE_REWIRING_HOMOPHILICREWIRING_HPP
#define EGTTOOLS_FINITEPOPULATIONS_STRUCTURE_REWIRING_HOMOPHILICREWIRING_HPP

#include <egttools/finite_populations/structure/rewiring/AbstractRewiring.hpp>
#include <algorithm>
#include <random>
#include <string>
#include <vector>

namespace egttools::FinitePopulations::structure::rewiring {

    /**
     * @brief Homophilic rewiring rule (Borges et al. 2023).
     *
     * When a focal node is selected for rewiring:
     *   1. Pick a random neighbour k of the focal node.
     *   2. If k has the same strategy as focal, keep the edge (no change).
     *   3. Otherwise, sever the edge (focal, k) and reconnect focal to a
     *      uniformly random non-neighbour j that shares focal's strategy.
     *      If no such node exists, fall back to a uniformly random non-neighbour.
     *
     * This rule drives the network toward same-strategy clusters, promoting
     * polarisation and social influence dynamics.
     *
     * Reference: Borges et al. (2023), "Social rewiring under polarisation".
     */
    struct HomophilicRewiring {

        [[nodiscard]] static std::string name() { return "HomophilicRewiring"; }

        static void rewire(int focal_node,
                           AdjacencyList &network,
                           const std::vector<int> &population,
                           std::mt19937_64 &gen) {
            const int N = static_cast<int>(network.size());
            auto &focal_neighbors = network[focal_node];
            if (focal_neighbors.empty()) return;

            std::uniform_int_distribution<int> nb_dist(
                    0, static_cast<int>(focal_neighbors.size()) - 1);
            int drop_idx = nb_dist(gen);
            int k = focal_neighbors[drop_idx];

            // If k is same-strategy, keep the edge.
            if (population[k] == population[focal_node]) return;

            // Candidates: non-neighbors, same strategy as focal.
            int focal_strat = population[focal_node];
            std::vector<int> same_strat, any_cand;
            for (int j = 0; j < N; ++j) {
                if (j == focal_node) continue;
                if (std::find(focal_neighbors.begin(), focal_neighbors.end(), j)
                    != focal_neighbors.end()) continue;
                any_cand.push_back(j);
                if (population[j] == focal_strat) same_strat.push_back(j);
            }

            std::vector<int> &pool = same_strat.empty() ? any_cand : same_strat;
            if (pool.empty()) return;

            std::uniform_int_distribution<int> pool_dist(
                    0, static_cast<int>(pool.size()) - 1);
            int j = pool[pool_dist(gen)];

            // Remove focal→k and k→focal
            focal_neighbors.erase(focal_neighbors.begin() + drop_idx);
            auto &k_neighbors = network[k];
            k_neighbors.erase(
                std::remove(k_neighbors.begin(), k_neighbors.end(), focal_node),
                k_neighbors.end());

            // Add focal→j and j→focal
            focal_neighbors.push_back(j);
            network[j].push_back(focal_node);
        }
    };

}// namespace egttools::FinitePopulations::structure::rewiring

#endif//EGTTOOLS_FINITEPOPULATIONS_STRUCTURE_REWIRING_HOMOPHILICREWIRING_HPP

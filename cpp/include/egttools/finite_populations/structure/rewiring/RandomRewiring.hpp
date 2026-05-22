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
#ifndef EGTTOOLS_FINITEPOPULATIONS_STRUCTURE_REWIRING_RANDOMREWIRING_HPP
#define EGTTOOLS_FINITEPOPULATIONS_STRUCTURE_REWIRING_RANDOMREWIRING_HPP

#include <egttools/finite_populations/structure/rewiring/AbstractRewiring.hpp>
#include <algorithm>
#include <random>
#include <string>
#include <vector>

namespace egttools::FinitePopulations::structure::rewiring {

    /**
     * @brief Random rewiring rule (Santos et al. 2006).
     *
     * When a focal node is selected for rewiring:
     *   1. Pick a random neighbour k of the focal node.
     *   2. Sever the edge (focal, k).
     *   3. Reconnect the focal node to a uniformly random non-neighbour j ≠ focal.
     *
     * The degree of the focal node is conserved (one edge is removed and one added).
     * The degree of k decreases by 1; the degree of j increases by 1.
     *
     * Reference: Santos & Pacheco (2006), "A new route to the evolution of cooperation",
     * Journal of Evolutionary Biology.
     */
    struct RandomRewiring {

        [[nodiscard]] static std::string name() { return "RandomRewiring"; }

        static void rewire(int focal_node,
                           AdjacencyList &network,
                           const std::vector<int> & /*population*/,
                           std::mt19937_64 &gen) {
            const int N = static_cast<int>(network.size());
            auto &focal_neighbors = network[focal_node];
            if (focal_neighbors.empty()) return;

            std::uniform_int_distribution<int> nb_dist(
                    0, static_cast<int>(focal_neighbors.size()) - 1);
            int drop_idx = nb_dist(gen);
            int k = focal_neighbors[drop_idx];

            // Collect candidates: non-neighbors of focal that are not focal itself.
            std::vector<int> candidates;
            candidates.reserve(static_cast<size_t>(N));
            for (int j = 0; j < N; ++j) {
                if (j == focal_node) continue;
                if (std::find(focal_neighbors.begin(), focal_neighbors.end(), j)
                    != focal_neighbors.end()) continue;
                candidates.push_back(j);
            }
            if (candidates.empty()) return;

            std::uniform_int_distribution<int> cand_dist(
                    0, static_cast<int>(candidates.size()) - 1);
            int j = candidates[cand_dist(gen)];

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

#endif//EGTTOOLS_FINITEPOPULATIONS_STRUCTURE_REWIRING_RANDOMREWIRING_HPP

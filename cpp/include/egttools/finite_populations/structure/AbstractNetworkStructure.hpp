//
// Created by Elias Fernandez on 04/01/2023.
//
#pragma once
#ifndef EGTTOOLS_FINITEPOPULATIONS_STRUCTURE_ABSTRACTNETWORKSTRUCTURE_HPP
#define EGTTOOLS_FINITEPOPULATIONS_STRUCTURE_ABSTRACTNETWORKSTRUCTURE_HPP

#include <egttools/Types.h>

#include <egttools/finite_populations/structure/AbstractStructure.hpp>
#include <map>
#include <vector>

namespace egttools::FinitePopulations::structure {
    // Legacy input type: accepted by constructors for Python/NetworkX compatibility.
    using NodeDictionary = std::map<int, std::vector<int>>;

    // Internal representation: O(1) node lookup, cache-friendly for large N.
    using AdjacencyList = std::vector<std::vector<int>>;

    // Converts a NodeDictionary (keyed 0..N-1) to a contiguous AdjacencyList.
    inline AdjacencyList dict_to_adjacency_list(const NodeDictionary &dict) {
        if (dict.empty()) return {};
        AdjacencyList adj(dict.size());
        for (const auto &[node, neighbors] : dict) {
            adj[static_cast<size_t>(node)] = neighbors;
        }
        return adj;
    }

    class AbstractNetworkStructure : public virtual AbstractStructure {
    public:
        /**
         * Initializes the network given a state vector.
         *
         * @param state: counts of each strategy in the population.
         */
        virtual void initialize_state(VectorXui &state) = 0;

        virtual void update_node(int node) = 0;

        /**
         * Calculates the numerically exact average gradient of selection for the
         * current per-node strategy assignment.
         *
         * @return average gradient of selection for each strategy.
         */
        virtual Vector &calculate_average_gradient_of_selection() = 0;

        virtual Vector &calculate_average_gradient_of_selection_and_update_population() = 0;

        /** @return number of nodes in the network. */
        [[nodiscard]] virtual int population_size() = 0;

        /** @return the adjacency list defining the network topology. */
        [[nodiscard]] virtual const AdjacencyList &network() = 0;
    };
}// namespace egttools::FinitePopulations::structure

#endif//EGTTOOLS_FINITEPOPULATIONS_STRUCTURE_ABSTRACTNETWORKSTRUCTURE_HPP

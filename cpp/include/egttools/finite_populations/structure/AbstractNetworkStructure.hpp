//
// Created by Elias Fernandez on 04/01/2023.
//
#pragma once
#ifndef EGTTOOLS_FINITEPOPULATIONS_STRUCTURE_ABSTRACTNETWORKSTRUCTURE_HPP
#define EGTTOOLS_FINITEPOPULATIONS_STRUCTURE_ABSTRACTNETWORKSTRUCTURE_HPP

#include <egttools/Types.h>

#include <egttools/finite_populations/structure/AbstractStructure.hpp>

namespace egttools::FinitePopulations::structure {
    using NodeDictionary = std::map<int, std::vector<int>>;

    class AbstractNetworkStructure : public virtual AbstractStructure {
    public:
        /**
         * Initializes the network given a state vector.
         *
         * This method should initialize the every node of the network given the counts
         * of each strategy in the population.
         *
         * @param state: a vector containing the counts of each strategy in the population.
         */
        virtual void initialize_state(VectorXui &state) = 0;

        virtual void update_node(int node) = 0;

        /**
         * Calculates the average gradient of selection given the current state of the network
         *
         * This method runs a single trial.
         *
         * @return the average gradient of selection for the current network
         */
        virtual Vector &calculate_average_gradient_of_selection() = 0;

        virtual Vector &calculate_average_gradient_of_selection_and_update_population() = 0;

        /**
         *
         * @return the the number of nodes of the network
         */
        [[nodiscard]] virtual int population_size() = 0;

        /**
         *
         * @return the mapping of nodes and neighbours that defines the network
         */
        [[nodiscard]] virtual NodeDictionary &network() = 0;
    };
}// namespace egttools::FinitePopulations::structure

#endif//EGTTOOLS_FINITEPOPULATIONS_STRUCTURE_ABSTRACTNETWORKSTRUCTURE_HPP

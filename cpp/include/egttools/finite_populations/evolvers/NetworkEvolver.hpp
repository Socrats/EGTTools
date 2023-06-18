//
// Created by Elias Fernandez on 08/01/2023.
//
#pragma once
#ifndef EGTTOOLS_FINITEPOPULATIONS_EVOLVERS_NETWORKEVOLVER_HPP
#define EGTTOOLS_FINITEPOPULATIONS_EVOLVERS_NETWORKEVOLVER_HPP

#include <egttools/SeedGenerator.h>
#include <egttools/Types.h>

#include <egttools/finite_populations/structure/AbstractNetworkStructure.hpp>
#include <map>
#include <memory>

#if defined(_OPENMP)
#include <egttools/OpenMPUtils.hpp>
#endif

namespace egttools::FinitePopulations::evolvers {
    using AbstractNetworkStructure = egttools::FinitePopulations::structure::AbstractNetworkStructure;
    using NodeDictionary = std::map<int, std::vector<int>>;

    class NetworkEvolver {
    public:
        /**
         * Evolves the population in structure for `nb_generations`.
         *
         * This method only returns the last total counts of strategies in the population.
         *
         * @param nb_generations : the number of generations for which evolution is run.
         * @param network : the network structure to evolve
         *
         * @return the final count of strategies in the population.
         */
        static VectorXui evolve(int_fast64_t nb_generations, AbstractNetworkStructure &network);

        static VectorXui evolve(int_fast64_t nb_generations, VectorXui &initial_state, AbstractNetworkStructure &network);

        /**
         * Evolves the population in structure for `nb_generations` and returns all states.
         *
         * This method returns the counts of strategies in the population for every generation
         * after the transitory period has passed.
         *
         * @param nb_generations : the number of generations for which evolution is run.
         * @param transitory : the transitory period.
         * @param network : the network structure to evolve
         *
         * @return the count of strategies of the population for every generation after `transitory`.
         */
        static MatrixXui2D run(int_fast64_t nb_generations, int_fast64_t transitory, AbstractNetworkStructure &network);

        static MatrixXui2D run(int_fast64_t nb_generations, int_fast64_t transitory, VectorXui &initial_state, AbstractNetworkStructure &network);

        /**
         * Calculates the average gradient of selection for a single network at a given state.
         * @param state
         * @param nb_simulations
         * @param nb_generations
         * @param network : the network structure to evolve
         *
         * @return avg. gradient of selection
         */
        static Vector calculate_average_gradient_of_selection(VectorXui &state, int_fast64_t nb_simulations, int_fast64_t nb_generations, AbstractNetworkStructure &network);

        /**
         * Calculates the average gradient of selection for multiple networks at a given state.
         *
         * This method is intended to average the gradient of multiple simulations with multiple networks.
         * You must specify the network type since this method will instantiate several Network classes
         * to run many simulations in parallel. Take into account that this might require a lot of memory.
         *
         * @param state
         * @param nb_simulations
         * @param nb_generations
         * @param networks : a list of networks classes to be used in the simulations
         * @return
         */
        static Vector calculate_average_gradient_of_selection(VectorXui &state, int_fast64_t nb_simulations,
                                                              int_fast64_t nb_generations, std::vector<AbstractNetworkStructure *> networks);
    };
}// namespace egttools::FinitePopulations::evolvers

#endif//EGTTOOLS_FINITEPOPULATIONS_EVOLVERS_NETWORKEVOLVER_HPP

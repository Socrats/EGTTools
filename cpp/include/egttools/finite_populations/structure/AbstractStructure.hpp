//
// Created by Elias Fernandez on 04/01/2023.
//
#pragma once
#ifndef EGTTOOLS_FINITEPOPULATIONS_STRUCTURE_ABSTRACTSTRUCTURE_HPP
#define EGTTOOLS_FINITEPOPULATIONS_STRUCTURE_ABSTRACTSTRUCTURE_HPP

#include <egttools/Types.h>

namespace egttools::FinitePopulations::structure {
    class AbstractStructure {
    public:
        virtual ~AbstractStructure() = default;

        /**
         * Initializes each element of the structure. In Evolutionary
         * games, this means that each individual in the structure is
         * assigned a strategy according to some algorithm (generally
         * it will be a random assignment). It is recommended
         * that subclasses which wish to implement other assignment
         * types, create different methods with more concrete name,
         * e.g., initialize_all_black, would initialize each individual
         * with the black strategy.
         */
        virtual void initialize() = 0;

        /**
         * Updates the strategy of an individual inside of the population.
         */
        virtual void update_population() = 0;

        /**
         *
         * @return the mean population state
         */
        [[nodiscard]] virtual VectorXui &mean_population_state() = 0;

        /**
         *
         * @return the number of strategies in the population
         */
        [[nodiscard]] virtual int nb_strategies() = 0;
    };
}// namespace egttools::FinitePopulations::structure

#endif//EGTTOOLS_FINITEPOPULATIONS_STRUCTURE_ABSTRACTSTRUCTURE_HPP

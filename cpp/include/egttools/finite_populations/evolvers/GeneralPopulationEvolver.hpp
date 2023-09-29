//
// Created by Elias Fernandez on 08/01/2023.
//
#pragma once
#ifndef EGTTOOLS_FINITEPOPULATIONS_EVOLVERS_GENERALPOPULATIONEVOLVER_HPP
#define EGTTOOLS_FINITEPOPULATIONS_EVOLVERS_GENERALPOPULATIONEVOLVER_HPP

#include <egttools/SeedGenerator.h>
#include <egttools/Types.h>

#include <egttools/finite_populations/structure/AbstractStructure.hpp>
#include <memory>

namespace egttools::FinitePopulations::evolvers {
    using AbstractStructure = egttools::FinitePopulations::structure::AbstractStructure;

    class GeneralPopulationEvolver {
    public:
        explicit GeneralPopulationEvolver(AbstractStructure &structure);

        /**
         * Evolves the population in structure for `nb_generations`.
         *
         * This method only returns the last total counts of strategies in the population.
         *
         * @param nb_generations : the number of generations for which evolution is run.
         * @return the final count of strategies in the population.
         */
        VectorXui evolve(int_fast64_t nb_generations);

        /**
         * Evolves the population in structure for `nb_generations` and returns all states.
         *
         * This method returns the counts of strategies in the population for every generation
         * after the transitory period has passed.
         *
         * @param nb_generations : the number of generations for which evolution is run.
         * @param transitory : the transitory period.
         * @return the count of strategies of the population for every generation after `transitory`.
         */
        MatrixXui2D run(int_fast64_t nb_generations, int_fast64_t transitory);

        [[nodiscard]] std::shared_ptr<AbstractStructure> structure();

    private:
        AbstractStructure &structure_;
    };
}// namespace egttools::FinitePopulations::evolvers

#endif//EGTTOOLS_FINITEPOPULATIONS_EVOLVERS_GENERALPOPULATIONEVOLVER_HPP

//
// Created by Elias Fernandez on 08/01/2023.
//
#include <egttools/finite_populations/evolvers/GeneralPopulationEvolver.hpp>

egttools::FinitePopulations::evolvers::GeneralPopulationEvolver::GeneralPopulationEvolver(AbstractStructure &structure) : structure_(structure) {
}

egttools::VectorXui egttools::FinitePopulations::evolvers::GeneralPopulationEvolver::evolve(int_fast64_t nb_generations) {
    // First initialize the structure
    structure_.initialize();

    // Then iterate for all the generations and return the final mean state
    for (int_fast64_t i = 0; i < nb_generations; ++i) {
        structure_.update_population();
    }

    return structure_.mean_population_state();
}

egttools::MatrixXui2D egttools::FinitePopulations::evolvers::GeneralPopulationEvolver::run(int_fast64_t nb_generations, int_fast64_t transitory) {
    if (nb_generations <= transitory)
        throw std::invalid_argument("The transitory period must be strictly smaller than the total number of generations.");

    // Create matrix of results
    MatrixXui2D results = MatrixXui2D::Zero(nb_generations - transitory, structure_.nb_strategies());

    // First initialize the structure
    structure_.initialize();

    for (int_fast64_t i = 0; i < transitory; ++i) {
        structure_.update_population();
    }

    results.row(0).array() = structure_.mean_population_state().array();

    // Then iterate for all the generations and return the final mean state
    for (int_fast64_t i = 0; i < nb_generations - transitory; ++i) {
        structure_.update_population();
        results.row(i).array() = structure_.mean_population_state().array();
    }

    return results;
}

std::shared_ptr<egttools::FinitePopulations::evolvers::AbstractStructure> egttools::FinitePopulations::evolvers::GeneralPopulationEvolver::structure() {
    return std::shared_ptr<egttools::FinitePopulations::evolvers::AbstractStructure>(&structure_);
}
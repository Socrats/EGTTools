//
// Created by Elias Fernandez on 12/06/2023.
//
#include <egttools/finite_populations/evolvers/NetworkEvolver.hpp>

egttools::FinitePopulations::evolvers::NetworkEvolver::NetworkEvolver(egttools::FinitePopulations::evolvers::AbstractNetworkStructure &structure) : structure_(structure) {
}

egttools::VectorXui egttools::FinitePopulations::evolvers::NetworkEvolver::evolve(int_fast64_t nb_generations) {
    // First initialize the structure
    structure_.initialize();

    // Then iterate for all the generations and return the final mean state
    for (int_fast64_t i = 0; i < nb_generations; ++i) {
        structure_.update_population();
    }

    return structure_.mean_population_state();
}
egttools::MatrixXui2D egttools::FinitePopulations::evolvers::NetworkEvolver::run(int_fast64_t nb_generations, int_fast64_t transitory) {
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
egttools::Vector egttools::FinitePopulations::evolvers::NetworkEvolver::calculate_average_gradient_of_selection(egttools::VectorXui &state, int_fast64_t nb_simulations, int_fast64_t nb_generations) {
    if (static_cast<int>(state.sum()) != structure_.population_size())
        throw std::invalid_argument("state must sum to population_size.");
    if (nb_simulations < 1)
        throw std::invalid_argument("the number of simulations must at least be 1.");
    if (nb_generations < 1)
        throw std::invalid_argument("the number of generations must at least be 1.");

    Vector average_gradient_of_selection = Vector::Zero(state.size());

    for (int_fast64_t i = 0; i < nb_simulations; ++i) {
        // Initialize the structure
        structure_.initialize_state(state);

        for (int_fast64_t j = 0; j < nb_generations; ++j) {
            // Calculate average gradient at the current generation
            average_gradient_of_selection += structure_.calculate_average_gradient_of_selection();
        }
    }


    return average_gradient_of_selection / (nb_simulations * nb_generations);
}
egttools::Vector egttools::FinitePopulations::evolvers::NetworkEvolver::calculate_average_gradient_of_selection(egttools::VectorXui &state, int_fast64_t nb_simulations, int_fast64_t nb_generations, std::vector<AbstractNetworkStructure *> networks) {
    if (static_cast<int>(state.sum()) != structure_.population_size())
        throw std::invalid_argument("state must sum to population_size.");
    if (nb_simulations < 1)
        throw std::invalid_argument("the number of simulations must at least be 1.");
    if (nb_generations < 1)
        throw std::invalid_argument("the number of generations must at least be 1.");

    Vector average_gradient_of_selection = Vector::Zero(state.size());

#pragma omp parallel for reduction(+ : average_gradient_of_selection) default(none) shared(networks, nb_generations, nb_simulations, state)
    for (auto & network : networks) {
        for (int_fast64_t i = 0; i < nb_simulations; ++i) {
            // Initialize the structure
            network->initialize_state(state);

            for (int_fast64_t j = 0; j < nb_generations; ++j) {
                // Calculate average gradient at the current generation
                average_gradient_of_selection += network->calculate_average_gradient_of_selection();
            }
        }
    }


    return average_gradient_of_selection / (nb_simulations * networks.size() * nb_generations);
}
std::shared_ptr<egttools::FinitePopulations::evolvers::AbstractNetworkStructure> egttools::FinitePopulations::evolvers::NetworkEvolver::structure() {
    return std::shared_ptr<egttools::FinitePopulations::evolvers::AbstractNetworkStructure>(&structure_);
}

//
// Created by Elias Fernandez on 12/06/2023.
//
#include <egttools/finite_populations/evolvers/NetworkEvolver.hpp>

egttools::VectorXui egttools::FinitePopulations::evolvers::NetworkEvolver::evolve(int_fast64_t nb_generations, AbstractNetworkStructure &network) {
    // First initialize the structure
    network.initialize();

    // Then iterate for all the generations and return the final mean state
    for (int_fast64_t i = 0; i < nb_generations; ++i) {
        network.update_population();
    }

    return network.mean_population_state();
}
egttools::VectorXui egttools::FinitePopulations::evolvers::NetworkEvolver::evolve(int_fast64_t nb_generations, egttools::VectorXui &initial_state, AbstractNetworkStructure &network) {
    // First initialize the structure
    network.initialize_state(initial_state);

    // Then iterate for all the generations and return the final mean state
    for (int_fast64_t i = 0; i < nb_generations; ++i) {
        network.update_population();
    }

    return network.mean_population_state();
}
egttools::MatrixXui2D egttools::FinitePopulations::evolvers::NetworkEvolver::run(int_fast64_t nb_generations, int_fast64_t transitory, AbstractNetworkStructure &network) {
    if (nb_generations <= transitory)
        throw std::invalid_argument("The transitory period must be strictly smaller than the total number of generations.");

    // Create matrix of results
    MatrixXui2D results = MatrixXui2D::Zero(nb_generations - transitory, network.nb_strategies());

    // First initialize the structure
    network.initialize();

    for (int_fast64_t i = 0; i < transitory; ++i) {
        network.update_population();
    }

    results.row(0).array() = network.mean_population_state().array();

    // Then iterate for all the generations and return the final mean state
    for (int_fast64_t i = 1; i < nb_generations - transitory; ++i) {
        network.update_population();
        results.row(i).array() = network.mean_population_state().array();
    }

    return results;
}
egttools::MatrixXui2D egttools::FinitePopulations::evolvers::NetworkEvolver::run(int_fast64_t nb_generations, int_fast64_t transitory, egttools::VectorXui &initial_state, AbstractNetworkStructure &network) {
    if (nb_generations <= transitory)
        throw std::invalid_argument("The transitory period must be strictly smaller than the total number of generations.");

    // Create matrix of results
    MatrixXui2D results = MatrixXui2D::Zero(nb_generations - transitory, network.nb_strategies());

    // First initialize the structure
    network.initialize_state(initial_state);

    for (int_fast64_t i = 0; i < transitory; ++i) {
        network.update_population();
    }

    results.row(0).array() = network.mean_population_state().array();

    // Then iterate for all the generations and return the final mean state
    for (int_fast64_t i = 1; i < nb_generations - transitory; ++i) {
        network.update_population();
        results.row(i).array() = network.mean_population_state().array();
    }

    return results;
}
egttools::Vector egttools::FinitePopulations::evolvers::NetworkEvolver::calculate_average_gradient_of_selection(egttools::VectorXui &state, int_fast64_t nb_simulations, int_fast64_t nb_generations, AbstractNetworkStructure &network) {
    if (static_cast<int>(state.sum()) != network.population_size())
        throw std::invalid_argument("state must sum to population_size.");
    if (nb_simulations < 1)
        throw std::invalid_argument("the number of simulations must at least be 1.");
    if (nb_generations < 1)
        throw std::invalid_argument("the number of generations must at least be 1.");

    Vector average_gradient_of_selection = Vector::Zero(state.size());

    for (int_fast64_t i = 0; i < nb_simulations; ++i) {
        // Initialize the structure
        network.initialize_state(state);

        for (int_fast64_t j = 0; j < nb_generations; ++j) {
            // Calculate average gradient at the current generation
            average_gradient_of_selection += network.calculate_average_gradient_of_selection();
            network.update_population();
        }
    }


    return average_gradient_of_selection / (nb_simulations * nb_generations);
}
egttools::Vector egttools::FinitePopulations::evolvers::NetworkEvolver::calculate_average_gradient_of_selection(egttools::VectorXui &state, int_fast64_t nb_simulations, int_fast64_t nb_generations, std::vector<AbstractNetworkStructure *> networks) {
    if (nb_simulations < 1)
        throw std::invalid_argument("the number of simulations must at least be 1.");
    if (nb_generations < 1)
        throw std::invalid_argument("the number of generations must at least be 1.");

    Vector average_gradient_of_selection = Vector::Zero(state.size());

#pragma omp parallel for reduction(+ : average_gradient_of_selection) default(none) shared(networks, nb_generations, nb_simulations, state)
    for (auto &network : networks) {
        for (int_fast64_t i = 0; i < nb_simulations; ++i) {
            // Initialize the structure
            network->initialize_state(state);

            for (int_fast64_t j = 0; j < nb_generations; ++j) {
                // Calculate average gradient at the current generation
                average_gradient_of_selection += network->calculate_average_gradient_of_selection();
                network->update_population();
            }
        }
    }


    return average_gradient_of_selection / (nb_simulations * networks.size() * nb_generations);
}

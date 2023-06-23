//
// Created by Elias Fernandez on 12/06/2023.
//
#include <tqdm/tqdm.h>

#include <egttools/finite_populations/evolvers/NetworkEvolver.hpp>

egttools::VectorXui egttools::FinitePopulations::evolvers::NetworkEvolver::evolve(int_fast64_t nb_generations,
                                                                                  AbstractNetworkStructure &network) {
    // First initialize the structure
    network.initialize();

    // Then iterate for all the generations and return the final mean state
    for (int_fast64_t i = 0; i < nb_generations; ++i) {
        network.update_population();
    }

    return network.mean_population_state();
}
egttools::VectorXui egttools::FinitePopulations::evolvers::NetworkEvolver::evolve(int_fast64_t nb_generations,
                                                                                  egttools::VectorXui &initial_state,
                                                                                  AbstractNetworkStructure &network) {
    // First initialize the structure
    network.initialize_state(initial_state);

    // Then iterate for all the generations and return the final mean state
    for (int_fast64_t i = 0; i < nb_generations; ++i) {
        network.update_population();
    }

    return network.mean_population_state();
}
egttools::MatrixXui2D egttools::FinitePopulations::evolvers::NetworkEvolver::run(int_fast64_t nb_generations,
                                                                                 int_fast64_t transitory,
                                                                                 AbstractNetworkStructure &network) {
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
egttools::MatrixXui2D egttools::FinitePopulations::evolvers::NetworkEvolver::run(int_fast64_t nb_generations,
                                                                                 int_fast64_t transitory,
                                                                                 egttools::VectorXui &initial_state,
                                                                                 AbstractNetworkStructure &network) {
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
egttools::Vector egttools::FinitePopulations::evolvers::NetworkEvolver::calculate_average_gradient_of_selection(egttools::VectorXui &state,
                                                                                                                int_fast64_t nb_simulations,
                                                                                                                int_fast64_t nb_generations,
                                                                                                                AbstractNetworkStructure &network) {
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
            average_gradient_of_selection += network.calculate_average_gradient_of_selection_and_update_population();
        }
    }


    return average_gradient_of_selection / (nb_simulations * nb_generations);
}
egttools::Vector egttools::FinitePopulations::evolvers::NetworkEvolver::calculate_average_gradient_of_selection(egttools::VectorXui &state,
                                                                                                                int_fast64_t nb_simulations,
                                                                                                                int_fast64_t nb_generations,
                                                                                                                std::vector<AbstractNetworkStructure *> networks) {
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
                average_gradient_of_selection += network->calculate_average_gradient_of_selection_and_update_population();
            }
        }
    }


    return average_gradient_of_selection / (nb_simulations * networks.size() * nb_generations);
}
egttools::Matrix2D egttools::FinitePopulations::evolvers::NetworkEvolver::calculate_average_gradients_of_selection(std::vector<egttools::VectorXui> &states,
                                                                                                                   int_fast64_t nb_simulations,
                                                                                                                   int_fast64_t nb_generations,
                                                                                                                   AbstractNetworkStructure &network) {
    if (nb_simulations < 1)
        throw std::invalid_argument("the number of simulations must at least be 1.");
    if (nb_generations < 1)
        throw std::invalid_argument("the number of generations must at least be 1.");

    auto nb_initial_states = static_cast<int_fast64_t>(states.size());
    egttools::Matrix2D average_gradients_of_selection = egttools::Matrix2D::Zero(nb_initial_states, states[0].size());

    // setup tqdm
    auto state_iterator = tq::trange(nb_initial_states);
    state_iterator.set_prefix("Iterating over states: ");

    for (int_fast64_t initial_state_index : state_iterator) {
        for (int_fast64_t i = 0; i < nb_simulations; ++i) {
            // Initialize the structure
            network.initialize_state(states[initial_state_index]);

            for (int_fast64_t j = 0; j < nb_generations; ++j) {
                // Calculate average gradient at the current generation
                average_gradients_of_selection.row(initial_state_index) += network.calculate_average_gradient_of_selection_and_update_population();
            }
        }
    }


    return average_gradients_of_selection / (nb_simulations * nb_generations);
}
egttools::Matrix2D egttools::FinitePopulations::evolvers::NetworkEvolver::calculate_average_gradients_of_selection(std::vector<egttools::VectorXui> &states,
                                                                                                                   int_fast64_t nb_simulations,
                                                                                                                   int_fast64_t nb_generations,
                                                                                                                   std::vector<AbstractNetworkStructure *> networks) {
    if (nb_simulations < 1)
        throw std::invalid_argument("the number of simulations must at least be 1.");
    if (nb_generations < 1)
        throw std::invalid_argument("the number of generations must at least be 1.");
    if (networks.empty())
        throw std::invalid_argument("There must be at least one network.");
    if (states.empty())
        throw std::invalid_argument("There must be at least one state.");

    auto nb_initial_states = static_cast<int_fast64_t>(states.size());
    egttools::Matrix2D average_gradients_of_selection = egttools::Matrix2D::Zero(nb_initial_states, states[0].size());

    // setup tqdm
    auto state_iterator = tq::trange(nb_initial_states);
    state_iterator.set_prefix("Iterating over simulations: ");

    for (auto state_index : state_iterator) {
        Vector average_gradient_of_selection = Vector::Zero(states[state_index].size());

#pragma omp parallel for reduction(+ : average_gradient_of_selection) default(none) shared(networks, nb_generations, nb_simulations, states, state_index)
        for (auto &network : networks) {
            for (int_fast64_t simulation = 0; simulation < nb_simulations; ++simulation) {
                // Initialize the structure
                network->initialize_state(states[state_index]);

                // run simulation
                for (int_fast64_t i = 0; i < nb_generations; ++i) {
                    average_gradient_of_selection += network->calculate_average_gradient_of_selection_and_update_population();
                }
            }
        }

        average_gradients_of_selection.row(state_index) = average_gradient_of_selection;
    }

    return average_gradients_of_selection / (nb_simulations * nb_generations * networks.size());
}
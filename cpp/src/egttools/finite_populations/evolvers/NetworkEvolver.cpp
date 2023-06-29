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
egttools::Matrix2D egttools::FinitePopulations::evolvers::NetworkEvolver::estimate_time_dependent_average_gradients_of_selection(std::vector<VectorXui> &initial_states,
                                                                                                                                 int_fast64_t nb_simulations,
                                                                                                                                 int_fast64_t generation_start,
                                                                                                                                 int_fast64_t generation_stop,
                                                                                                                                 AbstractNetworkStructure &network) {

    for (auto &state : initial_states)
        if (static_cast<int>(state.sum()) != network.population_size())
            throw std::invalid_argument("each state must sum to population_size.");
    if (nb_simulations < 1)
        throw std::invalid_argument("the number of simulations must at least be 1.");
    if (generation_start < 0)
        throw std::invalid_argument("the generation_start must be at least 0.");
    if (generation_stop <= generation_start)
        throw std::invalid_argument("generation_stop must be > generation_start.");

    auto nb_initial_states = static_cast<int_fast64_t>(initial_states.size());
    auto nb_population_states = egttools::starsBars(network.population_size(), network.nb_strategies());
    egttools::Matrix2D average_gradients_of_selection = egttools::Matrix2D::Zero(nb_population_states, network.nb_strategies());


    // setup tqdm
    auto state_iterator = tq::trange(nb_initial_states);
    state_iterator.set_prefix("Iterating over states: ");

    for (auto state_index : state_iterator) {
        for (int_fast64_t i = 0; i < nb_simulations; ++i) {
            // Initialize the structure
            network.initialize_state(initial_states[state_index]);

            // Evolve until generation_start
            for (int_fast64_t j = 0; j < generation_start; ++j) {
                network.update_population();
            }

            for (int_fast64_t j = generation_start; j < generation_stop; ++j) {
                // calculate current state
                auto current_state_index = static_cast<int_fast64_t>(egttools::FinitePopulations::calculate_state(network.population_size(),
                                                                                                                  network.mean_population_state()));

                // Compute the gradient of selection and update population
                average_gradients_of_selection.row(current_state_index) += network.calculate_average_gradient_of_selection_and_update_population();
            }
        }
    }

    return average_gradients_of_selection / (nb_simulations * (generation_stop - generation_start) * nb_initial_states);
}
egttools::Matrix2D egttools::FinitePopulations::evolvers::NetworkEvolver::estimate_time_dependent_average_gradients_of_selection(std::vector<VectorXui> &initial_states,
                                                                                                                                 int_fast64_t nb_simulations,
                                                                                                                                 int_fast64_t generation_start,
                                                                                                                                 int_fast64_t generation_stop,
                                                                                                                                 std::vector<AbstractNetworkStructure *> networks) {

    auto population_size = networks[0]->population_size();
    auto nb_strategies = networks[0]->nb_strategies();
    for (auto &network : networks) {
        if (network->population_size() != population_size)
            throw std::invalid_argument("All networks must have the same population size!");
        if (network->nb_strategies() != nb_strategies)
            throw std::invalid_argument("All networks must have the same number of strategies!");
    }
    for (auto &state : initial_states)
        if (static_cast<int>(state.sum()) != population_size)
            throw std::invalid_argument("each state must sum to population_size.");
    if (nb_simulations < 1)
        throw std::invalid_argument("the number of simulations must at least be 1.");
    if (generation_start < 0)
        throw std::invalid_argument("the generation_start must be at least 0.");
    if (generation_stop <= generation_start)
        throw std::invalid_argument("generation_stop must be > generation_start.");

    auto nb_initial_states = static_cast<int_fast64_t>(initial_states.size());
    auto nb_population_states = egttools::starsBars(population_size, nb_strategies);
    egttools::Matrix2D average_gradients_of_selection = egttools::Matrix2D::Zero(nb_population_states, nb_strategies);

    // setup tqdm
    auto state_iterator = tq::trange(nb_initial_states);
    state_iterator.set_prefix("Iterating over states: ");

    for (auto state_index : state_iterator) {
#pragma omp parallel for reduction(+ : average_gradients_of_selection) default(none) shared(initial_states, networks, nb_simulations, generation_start, generation_stop, state_index)
        for (auto &network : networks) {
            for (int_fast64_t i = 0; i < nb_simulations; ++i) {
                // Initialize the structure
                network->initialize_state(initial_states[state_index]);

                // Evolve until generation_start
                for (int_fast64_t j = 0; j < generation_start; ++j) {
                    network->update_population();
                }

                for (int_fast64_t j = generation_start; j < generation_stop; ++j) {
                    // calculate current state
                    auto current_state_index = static_cast<int_fast64_t>(egttools::FinitePopulations::calculate_state(network->population_size(),
                                                                                                                      network->mean_population_state()));

                    // Compute the gradient of selection and update population
                    average_gradients_of_selection.row(current_state_index) += network->calculate_average_gradient_of_selection_and_update_population();
                }
            }
        }
    }

    return average_gradients_of_selection / (nb_simulations * (generation_stop - generation_start) * networks.size() * nb_initial_states);
}
egttools::Matrix2D egttools::FinitePopulations::evolvers::NetworkEvolver::estimate_time_independent_average_gradients_of_selection(std::vector<egttools::VectorXui> &initial_states,
                                                                                                                                   int_fast64_t nb_simulations,
                                                                                                                                   int_fast64_t nb_generations,
                                                                                                                                   AbstractNetworkStructure &network) {
    for (auto &state : initial_states)
        if (static_cast<int>(state.sum()) != network.population_size())
            throw std::invalid_argument("each state must sum to population_size.");
    if (nb_simulations < 1)
        throw std::invalid_argument("the number of simulations must at least be 1.");
    if (nb_generations < 1)
        throw std::invalid_argument("the number of generations must be at least 1.");

    auto nb_initial_states = static_cast<int_fast64_t>(initial_states.size());
    auto nb_population_states = egttools::starsBars(network.population_size(), network.nb_strategies());
    egttools::Matrix2D average_gradients_of_selection = egttools::Matrix2D::Zero(nb_population_states, network.nb_strategies());

    // setup tqdm
    auto state_iterator = tq::trange(nb_initial_states);
    state_iterator.set_prefix("Iterating over states: ");

    for (auto state_index : state_iterator) {
        for (int_fast64_t i = 0; i < nb_simulations; ++i) {
            // Initialize the structure
            network.initialize_state(initial_states[state_index]);

            for (int_fast64_t j = 0; j < nb_generations; ++j) {
                // calculate current state
                auto current_state_index = static_cast<int_fast64_t>(egttools::FinitePopulations::calculate_state(network.population_size(),
                                                                                                                  network.mean_population_state()));

                // Compute the gradient of selection and update population
                average_gradients_of_selection.row(current_state_index) += network.calculate_average_gradient_of_selection_and_update_population();
            }
        }
    }


    return average_gradients_of_selection / (nb_simulations * nb_generations * nb_initial_states);
}
egttools::Matrix2D egttools::FinitePopulations::evolvers::NetworkEvolver::estimate_time_independent_average_gradients_of_selection(std::vector<egttools::VectorXui> &initial_states,
                                                                                                                                   int_fast64_t nb_simulations,
                                                                                                                                   int_fast64_t nb_generations,
                                                                                                                                   std::vector<AbstractNetworkStructure *> networks) {
    auto population_size = networks[0]->population_size();
    auto nb_strategies = networks[0]->nb_strategies();
    for (auto &network : networks) {
        if (network->population_size() != population_size)
            throw std::invalid_argument("All networks must have the same population size!");
        if (network->nb_strategies() != nb_strategies)
            throw std::invalid_argument("All networks must have the same number of strategies!");
    }
    for (auto &state : initial_states)
        if (static_cast<int>(state.sum()) != population_size)
            throw std::invalid_argument("each state must sum to population_size.");
    if (nb_simulations < 1)
        throw std::invalid_argument("the number of simulations must at least be 1.");
    if (nb_generations < 1)
        throw std::invalid_argument("the number of generations must be at least 1.");

    auto nb_initial_states = static_cast<int_fast64_t>(initial_states.size());
    auto nb_population_states = egttools::starsBars(population_size, nb_strategies);
    egttools::Matrix2D average_gradients_of_selection = egttools::Matrix2D::Zero(nb_population_states, nb_strategies);

    // setup tqdm
    auto state_iterator = tq::trange(nb_initial_states);
    state_iterator.set_prefix("Iterating over simulations: ");

    for (auto state_index : state_iterator) {
#pragma omp parallel for reduction(+ : average_gradients_of_selection) default(none) shared(initial_states, networks, nb_simulations, nb_generations, state_index)
        for (auto &network : networks) {
            for (int_fast64_t i = 0; i < nb_simulations; ++i) {
                // Initialize the structure
                network->initialize_state(initial_states[state_index]);

                for (int_fast64_t j = 0; j < nb_generations; ++j) {
                    // calculate current state
                    auto current_state_index = static_cast<int_fast64_t>(egttools::FinitePopulations::calculate_state(network->population_size(),
                                                                                                                      network->mean_population_state()));

                    // Compute the gradient of selection and update population
                    average_gradients_of_selection.row(current_state_index) += network->calculate_average_gradient_of_selection_and_update_population();
                }
            }
        }
    }

    return average_gradients_of_selection / (nb_simulations * nb_generations * networks.size() * nb_initial_states);
}
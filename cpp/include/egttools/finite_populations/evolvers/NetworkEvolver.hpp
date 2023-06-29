//
// Created by Elias Fernandez on 08/01/2023.
//
#pragma once
#ifndef EGTTOOLS_FINITEPOPULATIONS_EVOLVERS_NETWORKEVOLVER_HPP
#define EGTTOOLS_FINITEPOPULATIONS_EVOLVERS_NETWORKEVOLVER_HPP

#include <egttools/Distributions.h>
#include <egttools/SeedGenerator.h>
#include <egttools/Types.h>

#include <egttools/finite_populations/Utils.hpp>
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
         * Estimates the time-dependant gradient of selection
         *
         * This method will first evolve the population until generation_start. Afterwards,
         * it will average the gradient of selection observed for each state the population goes though
         * until generation_stop. The gradient will be computed for each time-step
         * and averaged over other simulations in which the population has gone through the same aggregated
         * state.
         *
         * @note We recommend only using this method with asynchronous updates.
         *
         * @param states : a vector containing the starting states of every simulation
         * @param nb_simulations : the number of simulations to run
         * @param generation_start : the generation at which we start to calculate the average gradient of selection
         * @param generation_stop : the final generation of the simulation
         * @param network : the network to evolve
         * @return the time-dependent average gradient of selection
         */
        static Matrix2D estimate_time_dependent_average_gradients_of_selection(std::vector<VectorXui> &states, int_fast64_t nb_simulations,
                                                                               int_fast64_t generation_start, int_fast64_t generation_stop,
                                                                               AbstractNetworkStructure &network);

        /**
         * Estimates the time-dependant gradient of selection
         *
         * This method will first evolve the population for (generation - 1) generations. Afterwards,
         * it will average the gradient of selection observed for each state the population goes though in
         * the next generation. This means, that if the update is asynchronous, the population will be
         * evolved for population_size time-steps and the gradient will be computed for each time-step
         * and averaged over other simulations in which the population has gone through the same aggregated
         * state.
         *
         * @note We recommend only using this method with asynchronous updates.
         *
         * @param states : a vector containing the starting states of every simulation
         * @param nb_simulations : the number of simulations to run
         * @param generation_start : the generation at which we start to calculate the average gradient of selection
         * @param generation_stop : the final generation of the simulation
         * @param networks : a vector containing shared pointers to the networks to use
         * @return the time-dependent average gradient of selection
         */
        static Matrix2D estimate_time_dependent_average_gradients_of_selection(std::vector<VectorXui> &states, int_fast64_t nb_simulations,
                                                                               int_fast64_t generation_start, int_fast64_t generation_stop,
                                                                               std::vector<AbstractNetworkStructure *> networks);

        /**
         * Estimates the time independent average gradient of selection.
         *
         * It is important here that the user takes into account that generations have a slightly different meaning if
         * the network updates are synchronous or asynchronous. In a synchronous case, in each generation, there is
         * a simultaneous update of every member of the population, thus, there a Z (population_size) steps.
         *
         * In the asynchronous case, we will adopt the definition used in Pinheiro, Pacheco and Santos 2012,
         * and assume that 1 generation = Z time-steps (Z asynchronous updates of the population). Thus, a simulation
         * with 25 generations and with 1000 individuals, will run for 25000 time-steps.
         *
         * This method will run a total of nb_simulations simulations. The final gradients are averaged over
         * simulations * nb_generations.
         *
         * @warning Don't use this method if the population has too many possible states, since it will likely take both a long time,
         * produce a bad estimation, and possible your computer will run out of memory.
         *
         * @param initial_states : a vector containing the starting states of every simulation
         * @param nb_simulations : the number of simulations to run
         * @param nb_generations : the number of generations to run
         * @param network : the network to use
         * @return The average gradient of selection for each possible population state.
         */
        static Matrix2D estimate_time_independent_average_gradients_of_selection(std::vector<VectorXui> &initial_states,
                                                                                 int_fast64_t nb_simulations,
                                                                                 int_fast64_t nb_generations,
                                                                                 AbstractNetworkStructure &network);

        /**
         * Estimates the time independent average gradient of selection.
         *
         * It is important here that the user takes into account that generations have a slightly different meaning if
         * the network updates are synchronous or asynchronous. In a synchronous case, in each generation, there is
         * a simultaneous update of every member of the population, thus, there a Z (population_size) steps.
         *
         * In the asynchronous case, we will adopt the definition used in Pinheiro, Pacheco and Santos 2012,
         * and assume that 1 generation = Z time-steps (Z asynchronous updates of the population). Thus, a simulation
         * with 25 generations and with 1000 individuals, will run for 25000 time-steps.
         *
         * This method will run a total of simulations * networks.size() simulations. The final gradients are averaged over
         * simulations * networks.size() * nb_generations.
         *
         * @warning Don't use this method if the population has too many possible states, since it will likely take both a long time,
         * produce a bad estimation, and possible your computer will run out of memory.
         *
         * @param initial_states : the starting states of every simulation
         * @param nb_simulations : the number of simulations to run
         * @param nb_generations : the number of generations to run
         * @param networks : a vector containing shared pointers to the networks to use
         * @return The average gradient of selection for each possible population state.
         */
        static Matrix2D estimate_time_independent_average_gradients_of_selection(std::vector<VectorXui> &initial_states,
                                                                                 int_fast64_t nb_simulations,
                                                                                 int_fast64_t nb_generations,
                                                                                 std::vector<AbstractNetworkStructure *> networks);
    };
}// namespace egttools::FinitePopulations::evolvers

#endif//EGTTOOLS_FINITEPOPULATIONS_EVOLVERS_NETWORKEVOLVER_HPP

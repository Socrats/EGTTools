/** Copyright (c) 2020-2023  Elias Fernandez
*
* This file is part of EGTtools.
*
* EGTtools is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* EGTtools is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with EGTtools.  If not, see <http://www.gnu.org/licenses/>
*/
#pragma once
#ifndef EGTTOOLS_FINITEPOPULATIONS_STRUCTURE_NETWORKGROUP_HPP
#define EGTTOOLS_FINITEPOPULATIONS_STRUCTURE_NETWORKGROUP_HPP

#include <egttools/Sampling.h>
#include <egttools/SeedGenerator.h>
#include <egttools/Types.h>

#include <algorithm>
#include <egttools/LruCache.hpp>
#include <egttools/finite_populations/Utils.hpp>
#include <egttools/finite_populations/structure/AbstractNetworkStructure.hpp>
#include <map>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace egttools::FinitePopulations::structure {
    template<class GameType, class CacheType = egttools::Utils::LRUCache<std::string, double>>
    class NetworkGroup final : public AbstractNetworkStructure {
    public:
        /**
         * @brief Network structure for N-player games
         *
         * It is necessary for provide a different structure for N-player games
         * since the literature often considers that these games (played in groups)
         * in a network occur not only among the focal player and its neighbours,
         * but also between the neighbours' neighbours and the focal player.
         * Thus, each player at a given generation plays k+1 games, where k
         * is the number of neighbours of the focal player. And the total fitness
         * is the accumulated payoff of those games.
         *
         * @param nb_strategies : number of strategies that can be present in the network
         * @param beta : intensity of selection
         * @param mu : mutation rate
         * @param network : a map of focal players and neighbours that define the network
         * @param game : a game object that will be used to calculate the payoff of the players
         * @param cache_size : the size of the cache used to store fitness values
         */
        NetworkGroup(int nb_strategies, double beta, double mu,
                     NodeDictionary &network, GameType &game,
                     int cache_size = 1000);

        void initialize() override;
        void initialize_state(VectorXui &state) override;
        //        void initialize_state(VectorXui &state, std::mt19937_64 &generator);
        void update_population() override;
        void update_node(int node) override;
        /**
         * Calculates the average gradient of selection given the current state of the network
         *
         * This method runs a single trial.
         *
         * @return the average gradient of selection for the current network
         */
        Vector &calculate_average_gradient_of_selection() override;
        Vector &calculate_average_gradient_of_selection_and_update_population() override;

        double calculate_fitness(int index);
        double calculate_game_payoff(int index);

        // getters
        [[nodiscard]] int population_size() override;
        [[nodiscard]] int nb_strategies() override;
        [[nodiscard]] NodeDictionary &network() override;
        [[nodiscard]] std::vector<int> population_strategies() const;
        [[nodiscard]] VectorXui &mean_population_state() override;
        [[nodiscard]] GameType &game();

        // setters
        //        void set_network(NodeDictionary network);
        //        void set_sync(bool sync);

    protected:
        int population_size_, nb_strategies_;
        double beta_, mu_;

        NodeDictionary network_;
        GameType &game_;

        CacheType cache_;

        // Population holder
        std::vector<int> population_;

        // Mean state
        VectorXui mean_population_state_;

        // Random distributions
        std::uniform_int_distribution<int> strategy_sampler_;
        std::uniform_int_distribution<int> population_sampler_;
        std::uniform_real_distribution<double> real_rand_;

        // Helper vectors
        Vector average_gradient_of_selection_;
        Vector transition_probability_;
        // T+ and T- for every strategy
        Vector transitions_plus_;
        Vector transitions_minus_;
        VectorXui neighbourhood_state_;

        std::mt19937_64 generator_{egttools::Random::SeedGenerator::getInstance().getSeed()};
    };

    template<class GameType, class CacheType>
    NetworkGroup<GameType, CacheType>::NetworkGroup(int nb_strategies,
                                                    double beta,
                                                    double mu,
                                                    NodeDictionary &network,
                                                    GameType &game,
                                                    int cache_size) : nb_strategies_(nb_strategies),
                                                                      beta_(beta),
                                                                      mu_(mu),
                                                                      network_(std::move(network)),
                                                                      game_(game),
                                                                      cache_(cache_size) {

        // The population size must be equal to the number of nodes in the network
        population_size_ = network_.size();
        population_ = std::vector<int>(population_size_);

        // Initialize the vector that will hold the mean population state
        // That is, the number of individuals adopting each strategy
        mean_population_state_ = VectorXui::Zero(nb_strategies_);

        // Initialize random generators
        strategy_sampler_ = std::uniform_int_distribution<int>(0, nb_strategies_ - 1);
        population_sampler_ = std::uniform_int_distribution<int>(0, population_size_ - 1);
        real_rand_ = std::uniform_real_distribution<double>(0.0, 1.0);

        // Initialize helper vectors
        transition_probability_ = Vector::Zero(nb_strategies_);
        // T+ and T- for every strategy
        transitions_plus_ = Vector::Zero(nb_strategies_);
        transitions_minus_ = Vector::Zero(nb_strategies_);
        average_gradient_of_selection_ = Vector::Zero(nb_strategies_);
        neighbourhood_state_ = VectorXui::Zero(nb_strategies_);
    }

    template<class GameType, class CacheType>
    void NetworkGroup<GameType, CacheType>::initialize() {
        for (int i = 0; i < population_size_; ++i) {
            auto strategy_index = strategy_sampler_(generator_);
            population_[i] = strategy_index;
            mean_population_state_(strategy_index) += 1;
        }

        assert(static_cast<int>(mean_population_state_.sum()) == population_size_);
    }

    template<class GameType, class CacheType>
    void NetworkGroup<GameType, CacheType>::initialize_state(egttools::VectorXui &state) {
        // We first fill the population with the number of strategies indicated by state in order
        mean_population_state_ = state;
        int index = 0;
        for (int s = 0; s < nb_strategies_; ++s) {
            for (size_t i = 0; i < state[s]; ++i) {
                population_[index] = s;
                index++;
            }
        }
        // Finally we shuffle
        std::shuffle(population_.begin(), population_.end(), generator_);
    }

    //    template<class GameType, class CacheType>
    //    void NetworkGroup<GameType, CacheType>::initialize_state(egttools::VectorXui &state, std::mt19937_64 &generator) {
    //        // We first fill the population with the number of strategies indicated by state in order
    //        int index = 0;
    //        for (int s = 0; s < nb_strategies_; ++s) {
    //            for (size_t i = 0; i < state[s]; ++i) {
    //                population_[index] = s;
    //                index++;
    //            }
    //        }
    //        // Finally we shuffle
    //        std::shuffle(population_.begin(), population_.end(), generator);
    //    }

    template<class GameType, class CacheType>
    Vector &NetworkGroup<GameType, CacheType>::calculate_average_gradient_of_selection() {
        // For every node, we need to calculate:
        // 1. probability of changing strategy
        // 2. probability of changing to a specific strategy
        average_gradient_of_selection_.setZero();
        // T+ and T- for every strategy
        transitions_plus_.setZero();
        transitions_minus_.setZero();

        // Iterate through every node
        for (int i = 0; i < population_size_; ++i) {
            // Calculate the probability of changing behavior of that node T
            auto fitness_focal = calculate_fitness(i);
            double transition_probability_unconditional = 0.;
            transition_probability_.setZero();

            for (size_t j = 0; j < network_[i].size(); ++j) {
                if (population_[network_[i][j]] == population_[i]) continue;
                // Get the fitness of both players
                auto fitness_neighbor = calculate_fitness(j);
                auto prob = egttools::FinitePopulations::fermi(beta_, fitness_focal, fitness_neighbor);
                transition_probability_unconditional += prob;
                transition_probability_(population_[network_[i][j]]) += prob;
            }
            transition_probability_unconditional /= network_[i].size();
            transition_probability_ /= network_[i].size();

            // Now add these transition probabilities to T+ and T-
            // T+ is the probability that there will be an increase in the strategy sk, so any other strategy must change to sk
            transitions_plus_ += transition_probability_;
            // T- is the probability that there will be a decrease in the strategy sk
            transitions_minus_(population_[i]) += transition_probability_unconditional;
        }

        average_gradient_of_selection_ = (transitions_plus_ - transitions_minus_) / population_size_;

        return average_gradient_of_selection_;
    }

    template<class GameType, class CacheType>
    Vector &NetworkGroup<GameType, CacheType>::calculate_average_gradient_of_selection_and_update_population() {
        // For every node, we need to calculate:
        // 1. probability of changing strategy
        // 2. probability of changing to a specific strategy
        average_gradient_of_selection_.setZero();
        // T+ and T- for every strategy
        transitions_plus_.setZero();
        transitions_minus_.setZero();

        // Iterate through every node
        for (int i = 0; i < population_size_; ++i) {
            // Calculate the probability of changing behavior of that node T
            auto fitness_focal = calculate_fitness(i);
            double transition_probability_unconditional = 0.;
            transition_probability_.setZero();

            for (size_t j = 0; j < network_[i].size(); ++j) {
                if (population_[network_[i][j]] == population_[i]) continue;
                // Get the fitness of both players
                auto fitness_neighbor = calculate_fitness(j);
                auto prob = egttools::FinitePopulations::fermi(beta_, fitness_focal, fitness_neighbor);
                transition_probability_unconditional += prob;
                transition_probability_(population_[network_[i][j]]) += prob;
            }
            transition_probability_unconditional /= network_[i].size();
            transition_probability_ /= network_[i].size();

            // Now add these transition probabilities to T+ and T-
            // T+ is the probability that there will be an increase in the strategy sk, so any other strategy must change to sk
            transitions_plus_ += transition_probability_;
            // T- is the probability that there will be a decrease in the strategy sk
            transitions_minus_(population_[i]) += transition_probability_unconditional;
        }

        // Now update this node in the population
        update_population();

        average_gradient_of_selection_ = (transitions_plus_ - transitions_minus_) / population_size_;

        // We need to multiply by the probability of selecting the current node for update (1/population_size)
        return average_gradient_of_selection_;
    }

    template<class GameType, class CacheType>
    void NetworkGroup<GameType, CacheType>::update_population() {
        // At the moment we will consider only an asynchronous update
        // In the future we should make this adaptable between sync and
        // async

        // NOTE: make sure to add a check for when the population reaches a
        // homogenous state. In that case, the simulation should be advanced
        // several rounds - This can be done from the evolver side

        // select randomly an individual to die
        auto focal_player = population_sampler_(generator_);

        // check if a mutation event occurs
        if (real_rand_(generator_) < mu_) {
            auto new_strategy = strategy_sampler_(generator_);
            while (new_strategy == population_[focal_player]) new_strategy = strategy_sampler_(generator_);

            mean_population_state_(population_[focal_player]) -= 1;
            mean_population_state_(new_strategy) += 1;

            population_[focal_player] = new_strategy;
        } else {// if not we continue

            // select a random neighbour
            auto dist = std::uniform_int_distribution<int>(0, network_[focal_player].size() - 1);
            auto neighbor_index = dist(generator_);
            int neighbor = network_[focal_player][neighbor_index];

            // If the strategies are the same, there is no change in the population
            if (population_[focal_player] == population_[neighbor]) return;

            // Get the fitness of both players
            auto fitness_focal = calculate_fitness(focal_player);
            auto fitness_neighbor = calculate_fitness(neighbor);

            // Check if update happens
            if (real_rand_(generator_) < egttools::FinitePopulations::fermi(beta_, fitness_focal, fitness_neighbor)) {
                // update mean counts
                mean_population_state_(population_[focal_player]) -= 1;
                mean_population_state_(population_[neighbor]) += 1;

                // update focal player strategy
                population_[focal_player] = population_[neighbor];
            }
        }
    }

    template<class GameType, class CacheType>
    void NetworkGroup<GameType, CacheType>::update_node(int node) {
        // check if a mutation event occurs
        if (real_rand_(generator_) < mu_) {
            auto new_strategy = strategy_sampler_(generator_);
            while (new_strategy == population_[node]) new_strategy = strategy_sampler_(generator_);

            mean_population_state_(population_[node]) -= 1;
            mean_population_state_(new_strategy) += 1;

            population_[node] = new_strategy;
        } else {// if not we continue

            // select a random neighbour
            auto dist = std::uniform_int_distribution<int>(0, network_[node].size() - 1);
            auto neighbor_index = dist(generator_);
            int neighbor = network_[node][neighbor_index];

            // If the strategies are the same, there is no change in the population
            if (population_[node] == population_[neighbor]) return;

            // Get the fitness of both players
            auto fitness_focal = calculate_fitness(node);
            auto fitness_neighbor = calculate_fitness(neighbor);

            // Check if update happens
            if (real_rand_(generator_) < egttools::FinitePopulations::fermi(beta_, fitness_focal, fitness_neighbor)) {
                // update mean counts
                mean_population_state_(population_[node]) -= 1;
                mean_population_state_(population_[neighbor]) += 1;

                // update focal player strategy
                population_[node] = population_[neighbor];
            }
        }
    }

    template<class GameType, class CacheType>
    double NetworkGroup<GameType, CacheType>::calculate_fitness(int index) {
        // We now need to iterate over each possible game the player will play,
        // we start with the central game which is played among the focal
        // player and her neighbourhood
        auto fitness = calculate_game_payoff(index);

        for (int &i : network_[index]) {
            fitness += calculate_game_payoff(i);
        }

        return fitness;
    }

    template<class GameType, class CacheType>
    double NetworkGroup<GameType, CacheType>::calculate_game_payoff(int index) {
        double payoff;

        // Let's get the neighborhood strategies
        // @note: this needs to be done more efficiently!
        neighbourhood_state_.setZero();
        for (int &i : network_[index]) {
            neighbourhood_state_(population_[i]) += 1;
        }

        std::stringstream result;
        result << neighbourhood_state_;

        std::string key = std::to_string(population_[index]) + result.str();

        // First we check if fitness value is in the lookup table
        if (!cache_.exists(key)) {
            payoff = game_.calculate_fitness(population_[index], neighbourhood_state_);

            // Store the fitness in the cache
            cache_.insert(key, payoff);
        } else {
            payoff = cache_.get(key);
        }

        return payoff;
    }

    template<class GameType, class CacheType>
    int NetworkGroup<GameType, CacheType>::population_size() {
        return population_size_;
    }

    template<class GameType, class CacheType>
    int NetworkGroup<GameType, CacheType>::nb_strategies() {
        return nb_strategies_;
    }

    template<class GameType, class CacheType>
    NodeDictionary &NetworkGroup<GameType, CacheType>::network() {
        return network_;
    }

    template<class GameType, class CacheType>
    std::vector<int> NetworkGroup<GameType, CacheType>::population_strategies() const {
        return population_;
    }

    template<class GameType, class CacheType>
    VectorXui &NetworkGroup<GameType, CacheType>::mean_population_state() {
        return mean_population_state_;
    }

    template<class GameType, class CacheType>
    GameType &NetworkGroup<GameType, CacheType>::game() {
        return game_;
    }


}// namespace egttools::FinitePopulations::structure

#endif//EGTTOOLS_FINITEPOPULATIONS_STRUCTURE_NETWORKGROUP_HPP

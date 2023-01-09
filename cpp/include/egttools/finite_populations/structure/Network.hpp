//
// Created by Elias Fernandez on 04/01/2023.
//
#pragma once
#ifndef EGTTOOLS_FINITEPOPULATIONS_STRUCTURE_NETWORK_HPP
#define EGTTOOLS_FINITEPOPULATIONS_STRUCTURE_NETWORK_HPP

#include <egttools/SeedGenerator.h>
#include <egttools/Types.h>

#include <egttools/LruCache.hpp>
#include <egttools/finite_populations/Utils.hpp>
#include <egttools/finite_populations/structure/AbstractStructure.hpp>
#include <map>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace egttools::FinitePopulations::structure {
    using NodeDictionary = std::map<int, std::vector<int>>;

    template<class GameType, class CacheType = egttools::Utils::LRUCache<std::string, double>>
    class Network final : public AbstractStructure {
    public:
        Network(int nb_strategies, double beta, double mu,
                NodeDictionary &network, GameType &game,
                int cache_size = 1000);

        void initialize() override;
        void update_population() override;

        double calculate_fitness(int index);

        // getters
        [[nodiscard]] int population_size();
        [[nodiscard]] int nb_strategies() override;
        [[nodiscard]] NodeDictionary &network();
        [[nodiscard]] std::vector<int> population_strategies() const;
        [[nodiscard]] VectorXui &mean_population_state() override;
        [[nodiscard]] GameType &game();

        // setters
        //        void set_network(NodeDictionary network);
        //        void set_sync(bool sync);

    private:
        int population_size_, nb_strategies_;
        double beta_, mu_;

        NodeDictionary network_;
        GameType &game_;

        int cache_size_;
        CacheType cache_;

        // Population holder
        std::vector<int> population_;

        // Mean state
        VectorXui mean_population_state_;

        // Random distributions
        std::uniform_int_distribution<int> strategy_sampler_;
        std::uniform_int_distribution<int> population_sampler_;
        std::uniform_real_distribution<double> real_rand_;

        std::mt19937_64 generator_{egttools::Random::SeedGenerator::getInstance().getSeed()};
    };

    template<class GameType, class CacheType>
    Network<GameType, CacheType>::Network(int nb_strategies,
                                          double beta,
                                          double mu,
                                          NodeDictionary &network,
                                          GameType &game,
                                          int cache_size) : nb_strategies_(nb_strategies),
                                                            beta_(beta),
                                                            mu_(mu),
                                                            network_(std::move(network)),
                                                            game_(game),
                                                            cache_size_(cache_size),
                                                            cache_(cache_size) {

        // The population size must be equal to the number of nodes in the network
        population_size_ = network_.size();

        // Initialize the vector that will hold the mean population state
        // That is, the number of individuals adopting each strategy
        mean_population_state_ = VectorXui::Zero(nb_strategies_);

        // Initialize random generators
        strategy_sampler_ = std::uniform_int_distribution<int>(0, nb_strategies_ - 1);
        population_sampler_ = std::uniform_int_distribution<int>(0, population_size_ - 1);
        real_rand_ = std::uniform_real_distribution<double>(0.0, 1.0);
    }

    template<class GameType, class CacheType>
    void Network<GameType, CacheType>::initialize() {
        for (int i = 0; i < population_size_; ++i) {
            auto strategy_index = strategy_sampler_(generator_);
            population_.push_back(strategy_index);
            mean_population_state_(strategy_index) += 1;
        }

        assert(static_cast<int>(mean_population_state_.sum()) == population_size_);
    }

    template<class GameType, class CacheType>
    void Network<GameType, CacheType>::update_population() {
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
            return;
        }// if not we continue

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

    template<class GameType, class CacheType>
    double Network<GameType, CacheType>::calculate_fitness(int index) {
        double fitness;

        // Let's get the neighborhood strategies
        // @note: this needs to be done more efficiently!
        VectorXui neighborhood_state = VectorXui::Zero(nb_strategies_);
        for (int &i : network_[index]) {
            neighborhood_state(population_[i]) += 1;
        }

        std::stringstream result;
        result << neighborhood_state;

        std::string key = std::to_string(population_[index]) + result.str();

        // First we check if fitness value is in the lookup table
        if (!cache_.exists(key)) {
            fitness = game_.calculate_fitness(population_[index], neighborhood_state);

            // Store the fitness in the cache
            cache_.insert(key, fitness);
        } else {
            fitness = cache_.get(key);
        }

        return fitness;
    }

    template<class GameType, class CacheType>
    int Network<GameType, CacheType>::population_size() {
        return population_size_;
    }

    template<class GameType, class CacheType>
    int Network<GameType, CacheType>::nb_strategies() {
        return nb_strategies_;
    }

    template<class GameType, class CacheType>
    NodeDictionary &Network<GameType, CacheType>::network() {
        return network_;
    }

    template<class GameType, class CacheType>
    std::vector<int> Network<GameType, CacheType>::population_strategies() const {
        return population_;
    }

    template<class GameType, class CacheType>
    VectorXui &Network<GameType, CacheType>::mean_population_state() {
        return mean_population_state_;
    }

    template<class GameType, class CacheType>
    GameType &Network<GameType, CacheType>::game() {
        return game_;
    }


}// namespace egttools::FinitePopulations::structure

#endif//EGTTOOLS_FINITEPOPULATIONS_STRUCTURE_NETWORK_HPP

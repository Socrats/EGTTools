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
#ifndef EGTTOOLS_FINITEPOPULATIONS_STRUCTURE_NETWORKSYNC_HPP
#define EGTTOOLS_FINITEPOPULATIONS_STRUCTURE_NETWORKSYNC_HPP

#include <egttools/SeedGenerator.h>
#include <egttools/Types.h>

#include <egttools/LruCache.hpp>
#include <egttools/finite_populations/Utils.hpp>
#include <egttools/finite_populations/structure/AbstractNetworkStructure.hpp>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace egttools::FinitePopulations::structure {
    /**
     * @brief Synchronous network structure with pairwise-comparison (Fermi) imitation updates.
     *
     * All nodes are updated simultaneously each generation using a double-buffer:
     * strategies for generation t+1 are computed from the state at generation t.
     *
     * See Network<> for the update rule and topology conventions.
     */
    template<class GameType, class CacheType = egttools::Utils::LRUCache<std::string, double>>
    class NetworkSync final : public AbstractNetworkStructure {
    public:
        NetworkSync(int nb_strategies, double beta, double mu,
                    const NodeDictionary &network, GameType &game,
                    int cache_size = 1000);

        void initialize() override;
        void initialize_state(VectorXui &state) override;
        void update_population() override;
        void update_node(int node) override;

        Vector &calculate_average_gradient_of_selection() override;
        Vector &calculate_average_gradient_of_selection_and_update_population() override;

        double calculate_fitness(int index);

        [[nodiscard]] int population_size() override;
        [[nodiscard]] int nb_strategies() override;
        [[nodiscard]] const AdjacencyList &network() override;
        [[nodiscard]] std::vector<int> population_strategies() const;
        [[nodiscard]] VectorXui &mean_population_state() override;
        [[nodiscard]] GameType &game();

    protected:
        int population_size_, nb_strategies_;
        double beta_, mu_;

        AdjacencyList network_;
        GameType &game_;

        CacheType cache_;

        std::vector<int> population_;
        std::vector<int> population_new_;
        VectorXui mean_population_state_;

        std::uniform_int_distribution<int> strategy_sampler_;
        std::uniform_int_distribution<int> population_sampler_;
        std::uniform_real_distribution<double> real_rand_;

        Vector average_gradient_of_selection_;
        Vector transition_probability_;
        Vector transitions_plus_;
        Vector transitions_minus_;
        VectorXui neighbourhood_state_;

        std::mt19937_64 generator_{egttools::Random::SeedGenerator::getInstance().getSeed()};
    };

    template<class GameType, class CacheType>
    NetworkSync<GameType, CacheType>::NetworkSync(int nb_strategies,
                                                  double beta,
                                                  double mu,
                                                  const NodeDictionary &network,
                                                  GameType &game,
                                                  int cache_size)
        : nb_strategies_(nb_strategies),
          beta_(beta),
          mu_(mu),
          network_(dict_to_adjacency_list(network)),
          game_(game),
          cache_(cache_size) {

        population_size_ = static_cast<int>(network_.size());
        population_ = std::vector<int>(population_size_);
        population_new_ = std::vector<int>(population_size_);
        mean_population_state_ = VectorXui::Zero(nb_strategies_);

        strategy_sampler_ = std::uniform_int_distribution<int>(0, nb_strategies_ - 1);
        population_sampler_ = std::uniform_int_distribution<int>(0, population_size_ - 1);
        real_rand_ = std::uniform_real_distribution<double>(0.0, 1.0);

        average_gradient_of_selection_ = Vector::Zero(nb_strategies_);
        transition_probability_ = Vector::Zero(nb_strategies_);
        transitions_plus_ = Vector::Zero(nb_strategies_);
        transitions_minus_ = Vector::Zero(nb_strategies_);
        neighbourhood_state_ = VectorXui::Zero(nb_strategies_);
    }

    template<class GameType, class CacheType>
    void NetworkSync<GameType, CacheType>::initialize() {
        mean_population_state_.setZero();
        for (int i = 0; i < population_size_; ++i) {
            auto s = strategy_sampler_(generator_);
            population_[i] = s;
            population_new_[i] = s;
            mean_population_state_(s) += 1;
        }
    }

    template<class GameType, class CacheType>
    void NetworkSync<GameType, CacheType>::initialize_state(egttools::VectorXui &state) {
        mean_population_state_ = state;
        int index = 0;
        for (int s = 0; s < nb_strategies_; ++s) {
            for (size_t i = 0; i < state[s]; ++i) {
                population_[index++] = s;
            }
        }
        std::shuffle(population_.begin(), population_.end(), generator_);
        for (int i = 0; i < population_size_; ++i) {
            population_new_[i] = population_[i];
        }
    }

    template<class GameType, class CacheType>
    Vector &NetworkSync<GameType, CacheType>::calculate_average_gradient_of_selection() {
        average_gradient_of_selection_.setZero();
        transitions_plus_.setZero();
        transitions_minus_.setZero();

        for (int i = 0; i < population_size_; ++i) {
            const auto degree = static_cast<double>(network_[i].size());
            if (degree == 0.0) continue;

            auto fitness_focal = calculate_fitness(i);
            double t_unconditional = 0.0;
            transition_probability_.setZero();

            for (int neighbor : network_[i]) {
                if (population_[neighbor] == population_[i]) continue;
                auto fitness_neighbor = calculate_fitness(neighbor);
                auto prob = egttools::FinitePopulations::fermi(beta_, fitness_focal, fitness_neighbor);
                t_unconditional += prob;
                transition_probability_(population_[neighbor]) += prob;
            }

            transitions_plus_ += transition_probability_ / degree;
            transitions_minus_(population_[i]) += t_unconditional / degree;
        }

        average_gradient_of_selection_ = (transitions_plus_ - transitions_minus_) / population_size_;
        return average_gradient_of_selection_;
    }

    template<class GameType, class CacheType>
    Vector &NetworkSync<GameType, CacheType>::calculate_average_gradient_of_selection_and_update_population() {
        average_gradient_of_selection_.setZero();
        transitions_plus_.setZero();
        transitions_minus_.setZero();

        for (int i = 0; i < population_size_; ++i) {
            const auto degree = static_cast<double>(network_[i].size());
            if (degree == 0.0) continue;

            auto fitness_focal = calculate_fitness(i);
            double t_unconditional = 0.0;
            transition_probability_.setZero();

            for (int neighbor : network_[i]) {
                if (population_[neighbor] == population_[i]) continue;
                auto fitness_neighbor = calculate_fitness(neighbor);
                auto prob = egttools::FinitePopulations::fermi(beta_, fitness_focal, fitness_neighbor);
                t_unconditional += prob;
                transition_probability_(population_[neighbor]) += prob;
            }

            transitions_plus_ += transition_probability_ / degree;
            transitions_minus_(population_[i]) += t_unconditional / degree;

            update_node(i);
        }

        for (int i = 0; i < population_size_; ++i)
            population_[i] = population_new_[i];

        average_gradient_of_selection_ = (transitions_plus_ - transitions_minus_) / population_size_;
        return average_gradient_of_selection_;
    }

    template<class GameType, class CacheType>
    void NetworkSync<GameType, CacheType>::update_population() {
        for (int i = 0; i < population_size_; ++i) {
            if (real_rand_(generator_) < mu_) {
                auto new_s = strategy_sampler_(generator_);
                while (new_s == population_[i]) new_s = strategy_sampler_(generator_);
                mean_population_state_(population_[i]) -= 1;
                mean_population_state_(new_s) += 1;
                population_new_[i] = new_s;
                continue;
            }

            const auto &neighbors = network_[i];
            if (neighbors.empty()) continue;

            auto dist = std::uniform_int_distribution<int>(0, static_cast<int>(neighbors.size()) - 1);
            int neighbor = neighbors[dist(generator_)];

            if (population_[i] == population_[neighbor]) continue;

            auto ff = calculate_fitness(i);
            auto fn = calculate_fitness(neighbor);

            if (real_rand_(generator_) < egttools::FinitePopulations::fermi(beta_, ff, fn)) {
                mean_population_state_(population_[i]) -= 1;
                mean_population_state_(population_[neighbor]) += 1;
                population_new_[i] = population_[neighbor];
            }
        }
        for (int i = 0; i < population_size_; ++i)
            population_[i] = population_new_[i];
    }

    template<class GameType, class CacheType>
    void NetworkSync<GameType, CacheType>::update_node(int node) {
        if (real_rand_(generator_) < mu_) {
            auto new_s = strategy_sampler_(generator_);
            while (new_s == population_[node]) new_s = strategy_sampler_(generator_);
            mean_population_state_(population_[node]) -= 1;
            mean_population_state_(new_s) += 1;
            population_new_[node] = new_s;
            return;
        }

        const auto &neighbors = network_[node];
        if (neighbors.empty()) return;

        auto dist = std::uniform_int_distribution<int>(0, static_cast<int>(neighbors.size()) - 1);
        int neighbor = neighbors[dist(generator_)];

        if (population_[node] == population_[neighbor]) return;

        auto ff = calculate_fitness(node);
        auto fn = calculate_fitness(neighbor);

        if (real_rand_(generator_) < egttools::FinitePopulations::fermi(beta_, ff, fn)) {
            mean_population_state_(population_[node]) -= 1;
            mean_population_state_(population_[neighbor]) += 1;
            population_new_[node] = population_[neighbor];
        }
    }

    template<class GameType, class CacheType>
    double NetworkSync<GameType, CacheType>::calculate_fitness(int index) {
        neighbourhood_state_.setZero();
        for (int nb : network_[index]) {
            neighbourhood_state_(population_[nb]) += 1;
        }

        std::ostringstream oss;
        oss << neighbourhood_state_;
        std::string key = std::to_string(population_[index]) + oss.str();

        if (!cache_.exists(key)) {
            double fitness = game_.calculate_fitness(population_[index], neighbourhood_state_);
            cache_.insert(key, fitness);
            return fitness;
        }
        return cache_.get(key);
    }

    template<class GameType, class CacheType>
    int NetworkSync<GameType, CacheType>::population_size() { return population_size_; }

    template<class GameType, class CacheType>
    int NetworkSync<GameType, CacheType>::nb_strategies() { return nb_strategies_; }

    template<class GameType, class CacheType>
    const AdjacencyList &NetworkSync<GameType, CacheType>::network() { return network_; }

    template<class GameType, class CacheType>
    std::vector<int> NetworkSync<GameType, CacheType>::population_strategies() const { return population_; }

    template<class GameType, class CacheType>
    VectorXui &NetworkSync<GameType, CacheType>::mean_population_state() { return mean_population_state_; }

    template<class GameType, class CacheType>
    GameType &NetworkSync<GameType, CacheType>::game() { return game_; }

}// namespace egttools::FinitePopulations::structure

#endif//EGTTOOLS_FINITEPOPULATIONS_STRUCTURE_NETWORKSYNC_HPP

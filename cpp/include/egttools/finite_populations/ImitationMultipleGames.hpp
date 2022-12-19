/** Copyright (c) 2019-2021  Elias Fernandez
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
#ifndef EGTTOOLS_FINITEPOPULATIONS_IMITATIONMULTIPLEGAMES_HPP
#define EGTTOOLS_FINITEPOPULATIONS_IMITATIONMULTIPLEGAMES_HPP

#include <egttools/SeedGenerator.h>
#include <egttools/Types.h>

#include <egttools/LruCache.hpp>
#include <egttools/finite_populations/Utils.hpp>
#include <egttools/finite_populations/games/AbstractGame.hpp>
#include <stdexcept>

#if defined(_OPENMP)
#include <egttools/OpenMPUtils.hpp>
#endif

namespace egttools::FinitePopulations {
    /**
        * This class caches the results according to the specified class in the template
        * parameter.
        *
        * @tparam Cache
        */
    template<class Cache = egttools::Utils::LRUCache<std::string, double>>
    class ImitationMultipleGames {
    public:
        /**
         * @brief Implements the Pairwise comparison Moran process.
         *
         * The selection dynamics implemented in this class are as follows:
         * At each generation 2 players are selected at random from the whole population.
         * Their fitness is compared according to the fermi function which returns a probability
         * that defines the likelihood that the first player will imitate the second.
         *
         * This process may include mutation.
         *
         * This class uses a cache to accelerate the computations.
         *
         * @param nb_strategies
         * @param pop_size
         * @param group_size : size of the group
         * @param game : pointer to the game class (it must be a child of AbstractGame)
         * @param cache_size : maximum number of elements in the cache
         */
        ImitationMultipleGames(size_t pop_size, std::vector<egttools::FinitePopulations::AbstractGame> &games, size_t cache_size = 1000000);

        /**
         * @brief Runs a moran process with social imitation
         *
         * Runs the moran process for a given number of generations and returns
         * all the states the simulation went through.
         *
         * @param nb_generations : maximum number of generations
         * @param beta : intensity of selection
         * @param mu: mutation probability
         * @param init_state : initial state of the population
         * @return a matrix with all the states the system went through during the simulation
         */
        MatrixXui2D run(int nb_generations, double beta, double mu, const Eigen::Ref<const VectorXui> &init_state);

        /**
         * @brief Estimates the stationary distribution of the population of strategies in the game.
         *
         * This methods is equal to estimate_stationary_distribution, but returns a Sparse Matrix instead of a
         * dense one. You should use this method one the system has a very large number of states, since
         * most of the entries of the stationary distribution will be 0, making it sparse.
         *
         * @param nb_runs : number of trials used to estimate the stationary distribution
         * @param nb_generations : number of generations per trial
         * @param transitory : transitory period not taken into account for the estimation
         * @param beta : intensity of selection
         * @param mu : mutation probability
         * @return the stationary distribution
         */
        SparseMatrix2D estimate_stationary_distribution_sparse(size_t nb_runs, size_t nb_generations, size_t transitory, double beta, double mu);

        // Getters
        [[nodiscard]] size_t nb_strategies() const;

        [[nodiscard]] size_t population_size() const;

        [[nodiscard]] size_t cache_size() const;

        [[nodiscard]] std::string game_type(size_t game_index) const;

        [[nodiscard]] const GroupPayoffs &payoffs(size_t game_index) const;

        // Setters
        void set_population_size(size_t pop_size);

        void set_cache_size(size_t cache_size);

        void change_game(size_t game_index, egttools::FinitePopulations::AbstractGame &game);

    private:
        size_t _nb_strategies, _pop_size, _cache_size, _nb_states;
        std::vector<egttools::FinitePopulations::AbstractGame> &_games;

        // Random distributions
        std::uniform_int_distribution<size_t> _pop_sampler;
        std::uniform_int_distribution<size_t> _strategy_sampler;
        std::uniform_int_distribution<size_t> _state_sampler;
        std::uniform_real_distribution<double> _real_rand;

        // Random generators
        std::mt19937_64 _mt{egttools::Random::SeedGenerator::getInstance().getSeed()};

        /**
         * @brief updates the population of strategies one step
         * @param s1 : index of strategy 1
         * @param s2 : index of strategy 2
         * @param beta : intensity of selection
         * @param birth : container for the index of the birth strategy
         * @param die : container for the index of the die strategy
         * @param strategies : vector of strategy counts
         * @param cache : reference to cache container
         * @param generator : random generator
         */
        inline bool
        _update_step(int s1, int s2, double beta, int &birth, int &die, VectorXui &strategies,
                     Cache &cache,
                     std::mt19937_64 &generator);

        inline void _update_step(int s1, int s2, double beta, double mu,
                                 int &birth, int &die, bool &homogeneous, int &idx_homo,
                                 VectorXui &strategies,
                                 Cache &cache,
                                 std::mt19937_64 &generator);

        /**
         * @brief updates the population of strategies and return the number of steps
         * @param s1 : index of strategy 1
         * @param s2 : index of strategy 2
         * @param beta : intensity of selection
         * @param mu : mutation probability
         * @param birth : container for the index of the birth strategy
         * @param die : container for the index of the die strategy
         * @param homogeneous : container indicating whether the population is homogeneous
         * @param idx_homo : container indicating the index of the homogeneous strategy
         * @param strategies : vector of strategy counts
         * @param cache : reference to cache container
         * @param geometric : geometric distribution of steps for a mutation to occur
         * @param generator : random generator
         * @return the number of steps that the update takes.
         */
        inline size_t
        _update_multi_step(int s1, int s2, double beta, double mu,
                           int &birth, int &die, bool &homogeneous, int &idx_homo,
                           VectorXui &strategies,
                           Cache &cache, std::geometric_distribution<size_t> &geometric,
                           std::mt19937_64 &generator);

        inline std::pair<size_t, size_t> _sample_players();

        inline std::pair<size_t, size_t> _sample_players(std::mt19937_64 &generator);

        /**
         * @brief samples 2 players from the population of strategies and updates references @param s1 and s2.
         *
         * @param s1 : reference container for strategy 1
         * @param s2 : reference container for strategy 2
         * @param generator
         */
        inline void _sample_players(size_t &s1, size_t &s2, std::mt19937_64 &generator);

        /**
         * @brief samples 2 players from the population of strategies and updates references @param s1 and s2.
         * @param s1 : reference container for strategy 1
         * @param s2 : reference container for strategy 2
         * @param strategies : vector of strategy counts
         * @param generator : random generator
         * @return true if the sampled strategies are equal, otherwise false
         */
        inline bool _sample_players(int &s1, int &s2, VectorXui &strategies, std::mt19937_64 &generator);

        inline double
        _calculate_fitness(const int &player_type, size_t game_index, VectorXui &strategies, Cache &cache);

        inline std::pair<bool, int> _is_homogeneous(VectorXui &strategies);
        inline void mutate_(std::mt19937_64 &generator, int &birth, int &idx_homo);
    };

    template<class Cache>
    ImitationMultipleGames<Cache>::ImitationMultipleGames(size_t pop_size,
                                                          std::vector<egttools::FinitePopulations::AbstractGame> &games,
                                                          size_t cache_size) : _pop_size(pop_size),
                                                                               _cache_size(cache_size),
                                                                               _games(games) {
        // For now, we assume that the number of strategies is constant across games
        _nb_strategies = games[0].nb_strategies();
        _pop_sampler = std::uniform_int_distribution<size_t>(0, _pop_size - 1);
        _strategy_sampler = std::uniform_int_distribution<size_t>(0, _nb_strategies - 1);
        _real_rand = std::uniform_real_distribution<double>(0.0, 1.0);
        _nb_states = egttools::starsBars(_pop_size, _nb_strategies);
        _state_sampler = std::uniform_int_distribution<size_t>(0, _nb_states - 1);
    }

    template<class Cache>
    MatrixXui2D ImitationMultipleGames<Cache>::run(int nb_generations, double beta, double mu,
                                                   const Eigen::Ref<const egttools::VectorXui> &init_state) {

        // Check that there is the length of init_state is the same as the number of strategies
        if (init_state.size() != static_cast<int>(_nb_strategies)) {
            throw std::invalid_argument(
                    "The length of the initial state array must be the number of strategies " + std::to_string(_nb_strategies));
        }
        // Check that the initial state is valid
        if (init_state.sum() != _pop_size) {
            throw std::invalid_argument(
                    "The sum of the entries of the initial state must be equal to the population size Z=" + std::to_string(_pop_size));
        }

        int die, birth, strategy_p1 = 0, strategy_p2 = 0;
        MatrixXui2D states = MatrixXui2D::Zero(nb_generations + 1, _nb_strategies);
        VectorXui strategies(_nb_strategies);
        // initialise initial state
        states.row(0).array() = init_state;
        strategies.array() = init_state;

        // Distribution number of generations for a mutation to happen
        std::geometric_distribution<int> geometric(mu);

        // Check if state is homogeneous
        auto [homogeneous, idx_homo] = _is_homogeneous(strategies);

        // Creates a cache for the fitness data
        Cache cache(_cache_size);
        int k;

        // Imitation process
        for (int j = 1; j < nb_generations + 1; ++j) {
            // Update with mutation and return how many steps should be added to the current
            // generation if the only change in the population could have been a mutation
            if (homogeneous) {
                k = geometric(_mt);
                // Update states matrix
                if (k == 0) states.row(j) = strategies;
                else if ((j + k) <= nb_generations) {
                    for (int z = j; z <= j + k; ++z)
                        states.row(z).array() = strategies;
                } else {
                    for (int z = j; z <= nb_generations; ++z)
                        states.row(z).array() = strategies;
                }

                // mutate
                birth = _strategy_sampler(_mt);
                // If population still homogeneous we wait for another mutation
                while (birth == idx_homo) birth = _strategy_sampler(_mt);
                strategies(birth) += 1;
                strategies(idx_homo) -= 1;
                homogeneous = false;

                // Update state count by k steps
                j += k + 1;
                // Update state after mutation
                if (j <= nb_generations)
                    states.row(j).array() = strategies;
            } else {
                // First we pick 2 players randomly
                _sample_players(strategy_p1, strategy_p2, strategies, _mt);

                _update_step(strategy_p1, strategy_p2, beta, mu,
                             birth, die, homogeneous, idx_homo,
                             strategies, cache, _mt);

                states.row(j).array() = strategies;
            }
        }
        return states;
    }


    template<class Cache>
    SparseMatrix2D
    ImitationMultipleGames<Cache>::estimate_stationary_distribution_sparse(size_t nb_runs, size_t nb_generations, size_t transitory, double beta,
                                                                           double mu) {
        // First we initialise the container for the stationary distribution
        auto sdist = SparseMatrix2DXui(1, _nb_states);
        //        sdist.reserve(VectorXi::Constant(_nb_states, std::min(10000, static_cast<int>(_nb_states))));
        // Distribution number of generations for a mutation to happen
        std::geometric_distribution<size_t> geometric(mu);

#pragma omp parallel for reduction(+ \
                                   : sdist) default(none) shared(geometric, nb_runs, nb_generations, transitory, beta, mu)
        for (size_t i = 0; i < nb_runs; ++i) {
            // Random generators - each thread should have its own generator
            std::mt19937_64 generator{egttools::Random::SeedGenerator::getInstance().getSeed()};

            // Then we sample a random population state
            VectorXui strategies = VectorXui::Zero(_nb_strategies);
            auto current_state = _state_sampler(generator);
            egttools::FinitePopulations::sample_simplex(current_state, _pop_size, _nb_strategies, strategies);

            int die = 0, birth = 0, strategy_p1 = 0, strategy_p2 = 0;
            // Check if state is homogeneous
            auto [homogeneous, idx_homo] = _is_homogeneous(strategies);

            // If it is we add a random mutant
            if (homogeneous) {
                mutate_(generator, birth, idx_homo);
                strategies(static_cast<int>(birth)) += 1;
                strategies(idx_homo) -= 1;
                homogeneous = false;
            }

            // Creates a cache for the fitness data
            Cache cache(_cache_size);
            size_t k, j;

            // First we run the simulations for a @param transitory number of generations
            for (j = 0; j < transitory; ++j) {
                _sample_players(strategy_p1, strategy_p2, strategies, generator);

                // Update with mutation and return how many steps should be added to the current
                // generation if the only change in the population could have been a mutation
                k = _update_multi_step(strategy_p1, strategy_p2, beta, mu,
                                       birth, die, homogeneous, idx_homo,
                                       strategies, cache,
                                       geometric, generator);

                // Update state count by k steps
                j += k;
            }

            // Update current state
            current_state = egttools::FinitePopulations::calculate_state(_pop_size, strategies);

            // Then we start counting
            for (; j < nb_generations; ++j) {
                // If the strategies are the same, there will be no change in the population
                if (homogeneous) {
                    k = geometric(generator);
                    // Update state count by k steps
                    sdist.coeffRef(0, static_cast<signed long>(current_state)) += k + 1;
                    mutate_(generator, birth, idx_homo);

                    strategies(static_cast<int>(birth)) += 1;
                    strategies(idx_homo) -= 1;

                    // Update state count by 1 step
                    current_state = egttools::FinitePopulations::calculate_state(_pop_size, strategies);
                    // and now update distribution after mutation
                    sdist.coeffRef(0, static_cast<signed long>(current_state)) += 1;
                    homogeneous = false;

                    // Update state count by k steps
                    j += k;
                } else {
                    // First we pick 2 players randomly
                    _sample_players(strategy_p1, strategy_p2, strategies, generator);

                    _update_step(strategy_p1, strategy_p2, beta, mu,
                                 birth, die, homogeneous, idx_homo,
                                 strategies, cache, generator);
                    // Update state count by k steps
                    current_state = egttools::FinitePopulations::calculate_state(_pop_size, strategies);
                    sdist.coeffRef(0, static_cast<signed long>(current_state)) += 1;
                }
            }
        }
        return sdist.cast<double>() / (nb_runs * (nb_generations - transitory));
    }


    template<class Cache>
    void ImitationMultipleGames<Cache>::mutate_(std::mt19937_64 &generator, int &birth, int &idx_homo) {
        // mutate
        birth = _strategy_sampler(generator);
        // We assume mutations imply changing strategy
        while (birth == idx_homo) birth = _strategy_sampler(generator);
    }

    template<class Cache>
    bool ImitationMultipleGames<Cache>::_update_step(int s1, int s2, double beta, int &birth, int &die,
                                                     VectorXui &strategies,
                                                     Cache &cache,
                                                     std::mt19937_64 &generator) {
        // Then we let them play to calculate their payoffs
        auto fitness_p1 = _calculate_fitness(s1, strategies, cache);
        auto fitness_p2 = _calculate_fitness(s2, strategies, cache);

        // Then we apply the moran process without mutation
        if (_real_rand(generator) < egttools::FinitePopulations::fermi(beta, fitness_p1, fitness_p2)) {
            // player 1 copies player 2
            die = s1;
            birth = s2;

            strategies(birth) += 1;
            strategies(die) -= 1;
            if (strategies(birth) == _pop_size) return true;
        }
        return false;
    }

    template<class Cache>
    void ImitationMultipleGames<Cache>::_update_step(int s1, int s2, double beta, double mu,
                                                     int &birth, int &die, bool &homogeneous, int &idx_homo,
                                                     VectorXui &strategies,
                                                     Cache &cache,
                                                     std::mt19937_64 &generator) {
        die = s1;

        if (s1 == s2) {// if the strategies are the same, the only change is with mutation
            // Check if player mutates
            if (_real_rand(generator) < mu) {
                mutate_(generator, birth, die);
                strategies(die) -= 1;
                strategies(birth) += 1;
                // Check if population is homogeneous
                if (strategies(birth) == _pop_size) {
                    homogeneous = true;
                    idx_homo = birth;
                }
            }

        } else {
            // Check if player mutates
            if (_real_rand(generator) < mu) {
                mutate_(generator, birth, die);
                strategies(birth) += 1;
                strategies(die) -= 1;

                // Check if population is homogeneous
                if (strategies(birth) == _pop_size) {
                    homogeneous = true;
                    idx_homo = birth;
                }
            } else {// If no mutation, player imitates

                // Then we let them play to calculate their payoffs
                auto fitness_p1 = _calculate_fitness(s1, strategies, cache);
                auto fitness_p2 = _calculate_fitness(s2, strategies, cache);

                // Then we apply the moran process with mutation
                if (_real_rand(generator) < egttools::FinitePopulations::fermi(beta, fitness_p1, fitness_p2)) {
                    // player 1 copies player 2
                    birth = s2;

                    strategies(birth) += 1;
                    strategies(die) -= 1;

                    // Check if population is homogeneous
                    if (strategies(birth) == _pop_size) {
                        homogeneous = true;
                        idx_homo = birth;
                    }
                }
            }
        }
    }

    template<class Cache>
    size_t
    ImitationMultipleGames<Cache>::_update_multi_step(int s1, int s2, double beta, double mu,
                                                      int &birth, int &die,
                                                      bool &homogeneous, int &idx_homo,
                                                      VectorXui &strategies,
                                                      Cache &cache,
                                                      std::geometric_distribution<size_t> &geometric,
                                                      std::mt19937_64 &generator) {

        size_t k = 0;
        die = s1, birth = s1;

        if (homogeneous) {
            k += geometric(generator);
            // mutate
            die = idx_homo;
            mutate_(generator, birth, die);
            strategies(birth) += 1;
            strategies(die) -= 1;
            homogeneous = false;
        } else if (s1 == s2) {// if the strategies are the same, the only change is with mutation
            // Check if player mutates
            if (_real_rand(generator) < mu) {
                mutate_(generator, birth, die);
                strategies(die) -= 1;
                strategies(birth) += 1;
                // Check if population is homogeneous
                if (strategies(birth) == _pop_size) {
                    homogeneous = true;
                    idx_homo = birth;
                }
            }
        } else {
            // Check if player mutates
            if (_real_rand(generator) < mu) {
                birth = _strategy_sampler(generator);
                // Assumes that a mutation is always to a different strategy
                while (birth == die) birth = _strategy_sampler(generator);
                strategies(birth) += 1;
                strategies(die) -= 1;

                // Check if population is homogeneous
                if (strategies(birth) == _pop_size) {
                    homogeneous = true;
                    idx_homo = birth;
                }
            } else {// If no mutation, player imitates

                // Then we let them play to calculate their payoffs
                auto fitness_p1 = _calculate_fitness(s1, strategies, cache);
                auto fitness_p2 = _calculate_fitness(s2, strategies, cache);

                // Then we check if player imitates
                if (_real_rand(generator) < egttools::FinitePopulations::fermi(beta, fitness_p1, fitness_p2)) {
                    // player 1 copies player 2
                    birth = s2;

                    strategies(birth) += 1;
                    strategies(die) -= 1;

                    // Check if population is homogeneous
                    if (strategies(birth) == _pop_size) {
                        homogeneous = true;
                        idx_homo = birth;
                    }
                }
            }
        }
        return k;
    }

    template<class Cache>
    std::pair<size_t, size_t> ImitationMultipleGames<Cache>::_sample_players() {
        auto player1 = _pop_sampler(_mt);
        auto player2 = _pop_sampler(_mt);
        while (player2 == player1) player2 = _pop_sampler(_mt);
        return std::make_pair(player1, player2);
    }

    template<class Cache>
    std::pair<size_t, size_t> ImitationMultipleGames<Cache>::_sample_players(std::mt19937_64 &generator) {
        auto player1 = _pop_sampler(generator);
        auto player2 = _pop_sampler(generator);
        while (player2 == player1) player2 = _pop_sampler(generator);
        return std::make_pair(player1, player2);
    }

    template<class Cache>
    void ImitationMultipleGames<Cache>::_sample_players(size_t &s1, size_t &s2, std::mt19937_64 &generator) {
        s1 = _pop_sampler(generator);
        s2 = _pop_sampler(generator);
        while (s1 == s2) s2 = _pop_sampler(generator);
    }

    template<class Cache>
    bool
    ImitationMultipleGames<Cache>::_sample_players(int &s1, int &s2, VectorXui &strategies, std::mt19937_64 &generator) {
        // sample 2 players from the pool
        auto player1 = _pop_sampler(generator);
        auto player2 = _pop_sampler(generator);
        while (player2 == player1) player2 = _pop_sampler(generator);

        size_t tmp = 0;
        s1 = 0;
        s2 = 0;
        bool unset_p1 = true, unset_p2 = true;

        // check which strategies correspond with these 2 players
        for (int i = 0; i < static_cast<int>(_nb_strategies); ++i) {
            tmp += strategies(i);
            if (tmp > player1 && unset_p1) {
                s1 = i;
                unset_p1 = false;
            }
            if (tmp > player2 && unset_p2) {
                s2 = i;
                unset_p2 = false;
            }
            if (!unset_p1 && !unset_p2) break;
        }
        return s1 == s2;
    }

    template<class Cache>
    double
    ImitationMultipleGames<Cache>::_calculate_fitness(const int &player_type, size_t game_index, VectorXui &strategies, Cache &cache) {
        double fitness;
        std::stringstream result;
        result << strategies;

        std::string key = std::to_string(player_type) + result.str();

        // First we check if fitness value is in the lookup table
        if (!cache.exists(key)) {
            strategies(player_type) -= 1;
            fitness = _games[game_index].calculate_fitness(player_type, _pop_size, strategies);
            strategies(player_type) += 1;

            // Finally we store the new fitness in the Cache. We also keep a Cache for the payoff given each group combination
            cache.insert(key, fitness);
        } else {
            fitness = cache.get(key);
        }

        return fitness;
    }

    template<class Cache>
    std::pair<bool, int> ImitationMultipleGames<Cache>::_is_homogeneous(VectorXui &strategies) {
        for (int i = 0; i < static_cast<int>(_nb_strategies); ++i) {
            if (strategies(i) == _pop_size) return std::make_pair(true, i);
        }
        return std::make_pair(false, -1);
    }

    template<class Cache>
    size_t ImitationMultipleGames<Cache>::nb_strategies() const {
        return _nb_strategies;
    }

    template<class Cache>
    size_t ImitationMultipleGames<Cache>::population_size() const {
        return _pop_size;
    }

    template<class Cache>
    size_t ImitationMultipleGames<Cache>::cache_size() const {
        return _cache_size;
    }

    template<class Cache>
    std::string ImitationMultipleGames<Cache>::game_type(size_t game_index) const {
        return _games[game_index].type();
    }

    template<class Cache>
    const GroupPayoffs &ImitationMultipleGames<Cache>::payoffs(size_t game_index) const {
        return _games[game_index].payoffs();
    }

    template<class Cache>
    void ImitationMultipleGames<Cache>::set_population_size(size_t pop_size) {
        _pop_size = pop_size;
        _nb_states = egttools::starsBars(_pop_size, _nb_strategies);
        _state_sampler = std::uniform_int_distribution<size_t>(0, _nb_states - 1);
        _pop_sampler = std::uniform_int_distribution<size_t>(0, _pop_size - 1);
    }

    template<class Cache>
    void ImitationMultipleGames<Cache>::set_cache_size(size_t cache_size) {
        _cache_size = cache_size;
    }

    template<class Cache>
    void ImitationMultipleGames<Cache>::change_game(size_t game_index, egttools::FinitePopulations::AbstractGame &game) {
        _games[game_index] = game;
    }
}// namespace egttools::FinitePopulations

#endif//EGTTOOLS_FINITEPOPULATIONS_IMITATIONMULTIPLEGAMES_HPP

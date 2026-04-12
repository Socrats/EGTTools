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
#ifndef EGTTOOLS_FINITEPOPULATIONS_PAIRWISECOMPARISONNUMERICAL_HPP
#define EGTTOOLS_FINITEPOPULATIONS_PAIRWISECOMPARISONNUMERICAL_HPP

#include <egttools/Distributions.h>
#include <egttools/SeedGenerator.h>
#include <egttools/Types.h>

#include <algorithm>
#include <egttools/finite_populations/Utils.hpp>
#include <egttools/finite_populations/games/AbstractGame.hpp>
#include <egttools/utils/ThreadSafeLRUCache.hpp>
#include <random>
#include <stdexcept>
#include <vector>

#if defined(_OPENMP)
#include <egttools/OpenMPExtensions.hpp>
#endif

namespace egttools::FinitePopulations {
    /**
        * This class caches the results according to the specified class in the template
        * parameter.
        *
        * @tparam Cache
        */
    template<class Cache = Utils::ThreadSafeLRUCache<std::string, double> >
    class PairwiseComparisonNumerical {
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
         * @param pop_size
         * @param game : pointer to the game class (it must be a child of AbstractGame)
         * @param cache_size : maximum number of elements in the cache
         */
        PairwiseComparisonNumerical(size_t pop_size, AbstractGame &game, size_t cache_size = 1000000);

        /**
         * Runs the moran process for a given number of generations or until it reaches a monomorphic state
         *
         * @param nb_generations : maximum number of generations
         * @param beta : intensity of selection
         * @param mu: mutation probability
         * @param init_state : initial state of the population
         * @return a vector with the final state of the population
         */
        VectorXui
        evolve(size_t nb_generations, double beta, double mu, const Eigen::Ref<const VectorXui> &init_state);

        /**
         * Runs the moran process for a given number of generations or until it reaches a monomorphic state
         *
         * @param nb_generations : maximum number of generations
         * @param beta : intensity of selection
         * @param strategies : reference vector with the initial state of the population
         * @param generator : random engine
         */
        void evolve(size_t nb_generations, double beta, VectorXui &strategies, std::mt19937_64 &generator);

        /**
         * Runs the moran process for a given number of generations or until it reaches a monomorphic state
         *
         * @param nb_generations : maximum number of generations
         * @param beta : intensity of selection
         * @param mu: mutation probability
         * @param init_state : initial state of the population
         * @param generator : random engine
         * @return a vector with the final state of the population
         */
        VectorXui
        evolve(size_t nb_generations, double beta, double mu, const Eigen::Ref<const VectorXui> &init_state,
               std::mt19937_64 &generator);

        /**
         * @brief Runs a moran process with social imitation without mutation.
         *
         * Runs the moran process for a given number of generations and returns
         * all the states the simulation went through.
         *
         * If initial_state is homogeneous, it will return a matrix of shape nb_generations x nb_strategies,
         * where every row contains init_state.
         *
         * @param nb_generations : maximum number of generations
         * @param beta : intensity of selection
         * @param init_state : initial state of the population
         * @return a matrix with all the states the system went through during the simulation
         */
        MatrixXui2D run(int64_t nb_generations, double beta, const Eigen::Ref<const VectorXui> &init_state);

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
        MatrixXui2D run(int64_t nb_generations, double beta, double mu, const Eigen::Ref<const VectorXui> &init_state);

        /**
         * @brief Runs a moran process with social imitation
         *
         * Runs the moran process for a given number of generations and returns
         * all the states the simulation went through.
         *
         * @param nb_generations : maximum number of generations
         * @param transient : the state of the population during the transient period will not be stored. Thus the
         *                    shape of the return matrix will be (nb_generations - transient, nb_strategies)
         * @param beta : intensity of selection
         * @param init_state : initial state of the population
         * @return a matrix with all the states the system went through during the simulation
         */
        MatrixXui2D run(int64_t nb_generations, int64_t transient, double beta,
                        const Eigen::Ref<const VectorXui> &init_state);

        /**
         * @brief Runs a moran process with social imitation
         *
         * Runs the moran process for a given number of generations and returns
         * all the states the simulation went through.
         *
         * @param nb_generations : maximum number of generations
         * @param transient : the state of the population during the transient period will not be stored. Thus the
         *                    shape of the return matrix will be (nb_generations - transient, nb_strategies)
         * @param beta : intensity of selection
         * @param mu: mutation probability
         * @param init_state : initial state of the population
         * @return a matrix with all the states the system went through during the simulation
         */
        MatrixXui2D run(int64_t nb_generations, int64_t transient, double beta, double mu,
                        const Eigen::Ref<const VectorXui> &init_state);

        /**
         * @brief Numerically estimates the gradient of selection between 2 strategies.
         *
         * Estimates T+(k) - T-(k) for each interior state k = 1, …, Z-1, where T+(k) is the
         * probability that the number of invaders increases and T-(k) is the probability that
         * it decreases. Mutation is not included; only the imitation step is considered.
         *
         * @param runs : number of independent one-step trials per interior state
         * @param invader : index of the invading strategy
         * @param resident : index of the resident strategy
         * @param beta : intensity of selection (Fermi parameter)
         * @return a vector of length pop_size + 1 with the gradient at each state (0 at boundaries)
         * @throws std::invalid_argument if invader or resident are out of range or equal
         */
        Vector estimate_gradient_of_selection(size_t runs, int invader, int resident, double beta);

        /**
         * @brief Estimates the fixation probability of the invading strategy over the resident strategy.
         *
         * This function will estimate numerically the fixation probability of an @param invader strategy
         * in a population of @param resident strategies.
         *
         * @param invader : index of the invading strategy
         * @param resident : index of the resident strategy
         * @param runs : number of independent runs (the estimation improves with the number of runs)
         * @param nb_generations : maximum number of generations per run
         * @param beta : intensity of selection
         * @return the fixation probability of the invader strategy
         */
        double estimate_fixation_probability(int invader, int resident, size_t runs, size_t nb_generations,
                                             double beta);

        /**
         * @brief Estimates the stationary distribution of the population of strategies in the game.
         *
         * The estimation of the stationary distribution is done by averaging the fraction of
         * the population of each strategy at the end of each trial over all trials.
         *
         * When @param tolerance > 0, runs are processed in batches of @param check_every and the
         * simulation stops as soon as the L∞ change in the normalised estimate between consecutive
         * batches falls below @param tolerance, without waiting for all @param nb_runs to complete.
         *
         * @param nb_runs : maximum number of trials used to estimate the stationary distribution
         * @param nb_generations : number of generations per trial
         * @param transitory : transitory period not taken into account for the estimation
         * @param beta : intensity of selection
         * @param mu : mutation probability
         * @param tolerance : convergence threshold (L∞ norm between consecutive batch estimates);
         *                    0.0 (default) disables early stopping and always runs all nb_runs
         * @param check_every : number of runs per convergence-check batch; 0 (default) uses
         *                      max(1, nb_runs / 10)
         * @return the stationary distribution
         */
        Vector estimate_stationary_distribution(size_t nb_runs, size_t nb_generations, size_t transitory, double beta,
                                                double mu, double tolerance = 0.0, size_t check_every = 0);

        /**
         * @brief Estimates the stationary distribution of the population of strategies in the game.
         *
         * This methods is equal to estimate_stationary_distribution, but returns a Sparse Matrix instead of a
         * dense one. You should use this method one the system has a very large number of states, since
         * most of the entries of the stationary distribution will be 0, making it sparse.
         *
         * When @param tolerance > 0, runs are processed in batches and stopped early when converged.
         * See estimate_stationary_distribution for full description of the tolerance/check_every params.
         *
         * @param nb_runs : maximum number of trials used to estimate the stationary distribution
         * @param nb_generations : number of generations per trial
         * @param transitory : transitory period not taken into account for the estimation
         * @param beta : intensity of selection
         * @param mu : mutation probability
         * @param tolerance : convergence threshold (L∞ norm); 0.0 disables early stopping
         * @param check_every : runs per convergence-check batch; 0 uses max(1, nb_runs / 10)
         * @return the stationary distribution
         */
        SparseMatrix2D estimate_stationary_distribution_sparse(size_t nb_runs, size_t nb_generations, size_t transitory,
                                                               double beta, double mu, double tolerance = 0.0,
                                                               size_t check_every = 0);

        /**
         * @brief Estimates the distribution of strategies in the population given the current game.
         *
         * This method directly estimates how frequent each strategy is in the population, without calculating
         * the stationary distribution as an intermediary step. You should use this method when the number
         * of states of the system is bigger than MAX_LONG_INT, since it would not be possible to index the states
         * in this case, and estimate_stationary_distribution and estimate_stationary_distribution_sparse would run into an
         * overflow error.
         *
         * When @param tolerance > 0, runs are processed in batches and stopped early when converged.
         * See estimate_stationary_distribution for full description of the tolerance/check_every params.
         *
         * @param nb_runs : maximum number of trials used to estimate the strategy distribution
         * @param nb_generations : number of generations per trial
         * @param transitory : transitory period not taken into account for the estimation
         * @param beta : intensity of selection
         * @param mu : mutation probability
         * @param tolerance : convergence threshold (L∞ norm); 0.0 disables early stopping
         * @param check_every : runs per convergence-check batch; 0 uses max(1, nb_runs / 10)
         * @return the strategy distribution
         */
        Vector estimate_strategy_distribution(size_t nb_runs, size_t nb_generations, size_t transitory, double beta,
                                              double mu, double tolerance = 0.0, size_t check_every = 0);

        /**
         * @brief Estimates the expected value of one or more indicator functions under
         *        the stationary distribution, without first computing the full distribution.
         *
         * At each post-transitory simulation step the method looks up the pre-computed
         * indicator values for the current population state and accumulates them.  The
         * per-run time-average converges to E[f_k] = Σ_s sd(s)·indicator_values(s,k) by
         * the ergodic theorem.  No explicit stationary distribution is stored.
         *
         * @p indicator_values must be a dense matrix of shape (nb_states × nb_indicators)
         * where row s contains the values of all indicators for the population state
         * corresponding to index s.  For group-level indicators f(group_config), build
         * this matrix with egttools::utils::precompute_group_to_state_indicator_matrix
         * before calling this method.
         *
         * Returns a matrix of shape (nb_runs_used × nb_indicators): one row per completed
         * run, containing that run's time-averaged indicator values.  The caller can
         * aggregate (mean, bootstrap CI, etc.) in Python without re-running the simulation.
         *
         * When @p tolerance > 0, runs are processed in batches of @p check_every
         * (default max(1, nb_runs/10)) and the simulation stops early when the L1 norm
         * of the change in the column-means between consecutive batches falls below
         * @p tolerance.  tolerance=0 (default) always runs all nb_runs.
         *
         * @param nb_runs            maximum number of independent simulation runs.
         * @param nb_generations     number of generations per run.
         * @param transitory         transitory period excluded from accumulation.
         * @param beta               intensity of selection.
         * @param mu                 mutation probability (must be > 0).
         * @param indicator_values   (nb_states × nb_indicators) precomputed matrix.
         * @param tolerance          L1 convergence threshold; 0.0 disables early stopping.
         * @param check_every        batch size for convergence checks; 0 → auto.
         * @return Matrix2D of shape (nb_runs_used × nb_indicators).
         */
        Matrix2D estimate_stationary_indicators(
            size_t nb_runs, size_t nb_generations, size_t transitory,
            double beta, double mu,
            const Eigen::Ref<const Matrix2D> &indicator_values,
            double tolerance = 0.0, size_t check_every = 0);

        // Getters
        [[nodiscard]] size_t nb_strategies() const;

        [[nodiscard]] size_t population_size() const;

        [[nodiscard]] size_t cache_size() const;

        [[nodiscard]] std::string game_type() const;

        [[nodiscard]] const GroupPayoffs &payoffs() const;

        [[nodiscard]] int64_t nb_states() const;

        // Setters
        void set_population_size(size_t pop_size);

        void set_cache_size(size_t cache_size);

        void change_game(egttools::FinitePopulations::AbstractGame &game);

    private:
        size_t _nb_strategies, _pop_size, _cache_size, _nb_states;
        egttools::FinitePopulations::AbstractGame *_game;

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

        /**
         * @brief samples 2 players from the population of strategies and updates references @param s1 and s2.
         * @param s1 : reference container for strategy 1 (index into strategies)
         * @param s2 : reference container for strategy 2 (index into strategies)
         * @param strategies : vector of strategy counts
         * @param generator : random generator
         * @return true if the sampled strategies are equal, otherwise false
         */
        inline bool _sample_players(int &s1, int &s2, VectorXui &strategies, std::mt19937_64 &generator);

        inline double
        _calculate_fitness(const int &player_type, VectorXui &strategies, Cache &cache);

        inline std::pair<bool, int> _is_homogeneous(VectorXui &strategies) const;

        inline void mutate_(std::mt19937_64 &generator, int &birth, const int &idx_homo);

        /**
         * @brief Validates common run/evolve arguments.
         * Throws std::invalid_argument if nb_generations <= 0, init_state size doesn't match
         * nb_strategies, or init_state sum doesn't equal pop_size.
         */
        void _validate_args(int64_t nb_generations, const Eigen::Ref<const VectorXui> &init_state) const;

        /**
         * @brief Validates transient period for run methods with transient.
         * Throws std::invalid_argument if transient < 0 or transient >= nb_generations.
         */
        void _validate_transient(int64_t nb_generations, int64_t transient) const;
    };

    template<class Cache>
    PairwiseComparisonNumerical<Cache>::PairwiseComparisonNumerical(size_t pop_size,
                                                                    egttools::FinitePopulations::AbstractGame &game,
                                                                    size_t cache_size) : _pop_size(pop_size),
        _cache_size(cache_size),
        _game(&game) {
        // Initialize random uniform distribution
        _nb_strategies = game.nb_strategies();
        _pop_sampler = std::uniform_int_distribution<size_t>(0, _pop_size - 1);
        _strategy_sampler = std::uniform_int_distribution<size_t>(0, _nb_strategies - 1);
        _real_rand = std::uniform_real_distribution<double>(0.0, 1.0);
        _nb_states = egttools::starsBars(_pop_size, _nb_strategies);
        _state_sampler = std::uniform_int_distribution<size_t>(0, _nb_states - 1);
    }

    template<class Cache>
    VectorXui
    PairwiseComparisonNumerical<Cache>::evolve(const size_t nb_generations, const double beta, const double mu,
                                               const Eigen::Ref<const VectorXui> &init_state) {
        _validate_args(static_cast<int64_t>(nb_generations), init_state);
        if (mu <= 0)
            throw std::invalid_argument(
                "mu must be > 0. If you want to run a simulation without mutation, "
                "please use the method signature without the mu parameter");

        VectorXui strategies(_nb_strategies);
        // Initialise strategies from init_state
        strategies.array() = init_state.eval();

        // Avg. number of rounds for a mutation to happen
        std::geometric_distribution<size_t> geometric(mu);
        auto [homogeneous, idx_homo] = _is_homogeneous(strategies);

        // Creates a cache for the fitness data
        Cache cache(_cache_size);
        // Initialize helper parameters
        int die = 0, birth = 0, strategy_p1 = 0, strategy_p2 = 0;

        // Imitation process
        for (size_t j = 0; j < nb_generations; ++j) {
            _sample_players(strategy_p1, strategy_p2, strategies, _mt);

            // Update with mutation and return how many steps should be added to the current
            // generation if the only change in the population could have been a mutation
            size_t k = _update_multi_step(strategy_p1, strategy_p2, beta, mu,
                                          birth, die, homogeneous, idx_homo,
                                          strategies, cache,
                                          geometric, _mt);

            // Saturating add: avoid size_t wrap-around when k is very large
            if (k >= nb_generations - j) break;
            j += k;
        }

        return strategies;
    }

    template<class Cache>
    void
    PairwiseComparisonNumerical<Cache>::evolve(const size_t nb_generations, const double beta, VectorXui &strategies,
                                               std::mt19937_64 &generator) {
        // This method runs a Moran process with pairwise comparison
        // using the fermi rule and no mutation
        _validate_args(static_cast<int64_t>(nb_generations), strategies);

        int die, birth, strategy_p1 = 0, strategy_p2 = 0;

        // Check if initial state is already homogeneous, in which case return
        if ((strategies.array() == _pop_size).any()) return;

        // Creates a cache for the fitness data
        Cache cache(_cache_size);

        // Now we start the imitation process
        for (size_t i = 0; i < nb_generations; ++i) {
            // First we pick 2 players randomly
            // If the strategies are the same, there will be no change in the population
            if (_sample_players(strategy_p1, strategy_p2, strategies, generator)) continue;

            if (_update_step(strategy_p1, strategy_p2, beta, birth, die, strategies, cache, generator))
                break;
        }
    }

    template<class Cache>
    auto PairwiseComparisonNumerical<Cache>::evolve(const size_t nb_generations, double beta, double mu,
                                                    const Eigen::Ref<const VectorXui> &init_state,
                                                    std::mt19937_64 &generator) -> VectorXui {
        _validate_args(static_cast<int64_t>(nb_generations), init_state);
        if (mu <= 0)
            throw std::invalid_argument(
                "mu must be > 0. If you want to run a simulation without mutation, "
                "please use the method signature without the mu parameter");

        VectorXui strategies(_nb_strategies);
        // Initialise strategies from init_state
        strategies.array() = init_state.eval();

        // Avg. number of rounds for a mutation to happen
        std::geometric_distribution<size_t> geometric(mu);
        auto [homogeneous, idx_homo] = _is_homogeneous(strategies);

        // Creates a cache for the fitness data
        Cache cache(_cache_size);
        // Initialize helper parameters

        // Imitation process
        int strategy_p1 = 0, strategy_p2 = 0, birth = 0, die = 0;
        for (size_t j = 0; j < nb_generations; ++j) {
            _sample_players(strategy_p1, strategy_p2, strategies, generator);

            // Update with mutation and return how many steps should be added to the current
            // generation if the only change in the population could have been a mutation
            size_t k = _update_multi_step(strategy_p1, strategy_p2, beta, mu,
                                          birth, die, homogeneous, idx_homo,
                                          strategies, cache,
                                          geometric, generator);

            // Saturating add: avoid size_t wrap-around when k is very large
            if (k >= nb_generations - j) break;
            j += k;
        }

        return strategies;
    }

    template<class Cache>
    MatrixXui2D PairwiseComparisonNumerical<Cache>::run(const int64_t nb_generations, const double beta,
                                                        const double mu,
                                                        const Eigen::Ref<const VectorXui> &init_state) {
        _validate_args(nb_generations, init_state);
        if (mu <= 0)
            throw std::invalid_argument(
                "mu must be > 0. If you want to run a simulation without mutation, "
                "please use the method signature without the mu parameter");

        int die, birth, strategy_p1 = 0, strategy_p2 = 0;
        MatrixXui2D states = MatrixXui2D::Zero(nb_generations + 1, _nb_strategies);
        VectorXui strategies(_nb_strategies);
        // initialise initial state
        states.row(0).array() = init_state.eval();
        strategies.array() = init_state.eval();

        // Distribution number of generations for a mutation to happen
        std::geometric_distribution<int64_t> geometric(mu);

        // Check if state is homogeneous
        auto [homogeneous, idx_homo] = _is_homogeneous(strategies);

        // Creates a cache for the fitness data
        Cache cache(_cache_size);

        // Imitation process
        for (int64_t j = 1; j < nb_generations + 1; ++j) {
            // Update with mutation and return how many steps should be added to the current
            // generation if the only change in the population could have been a mutation
            if (homogeneous) {
                int k = geometric(_mt);
                // Update states matrix
                if (k == 0) states.row(j) = strategies;
                else if ((j + k) < nb_generations + 1) {
                    for (int64_t z = j; z <= j + k; ++z)
                        states.row(z).array() = strategies;
                } else {
                    for (int64_t z = j; z < nb_generations + 1; ++z)
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
                if (j < nb_generations + 1)
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
    MatrixXui2D PairwiseComparisonNumerical<Cache>::run(const int64_t nb_generations, const int64_t transient,
                                                        const double beta,
                                                        const double mu,
                                                        const Eigen::Ref<const VectorXui> &init_state) {
        _validate_args(nb_generations, init_state);
        _validate_transient(nb_generations, transient);
        if (mu <= 0)
            throw std::invalid_argument(
                "mu must be > 0. If you want to run a simulation without mutation, "
                "please use the method signature without the mu parameter");

        int die, birth, strategy_p1 = 0, strategy_p2 = 0;
        const auto total_counting_generations = nb_generations - transient;
        MatrixXui2D states = MatrixXui2D::Zero(total_counting_generations, _nb_strategies);
        VectorXui strategies(_nb_strategies);
        // initialise initial state
        strategies.array() = init_state.eval();

        // Distribution number of generations for a mutation to happen
        std::geometric_distribution<int64_t> geometric(mu);

        // Check if state is homogeneous
        auto [homogeneous, idx_homo] = _is_homogeneous(strategies);

        // Creates a cache for the fitness data
        Cache cache(_cache_size);
        int k;

        for (int64_t j = 0; j < transient; ++j) {
            // Update with mutation and return how many steps should be added to the current
            // generation if the only change in the population could have been a mutation
            if (homogeneous) {
                k = geometric(_mt);

                // mutate
                birth = _strategy_sampler(_mt);
                // If population still homogeneous we wait for another mutation
                while (birth == idx_homo) birth = _strategy_sampler(_mt);
                strategies(birth) += 1;
                strategies(idx_homo) -= 1;
                homogeneous = false;

                // Update state count by k steps
                j += k + 1;
            } else {
                // First we pick 2 players randomly
                _sample_players(strategy_p1, strategy_p2, strategies, _mt);

                _update_step(strategy_p1, strategy_p2, beta, mu,
                             birth, die, homogeneous, idx_homo,
                             strategies, cache, _mt);
            }
        }

        // Imitation process
        for (int64_t j = 0; j < total_counting_generations; ++j) {
            // Update with mutation and return how many steps should be added to the current
            // generation if the only change in the population could have been a mutation
            if (homogeneous) {
                k = geometric(_mt);
                // Update states matrix
                if (k == 0) states.row(j) = strategies;
                else if ((j + k) < total_counting_generations) {
                    for (int64_t z = j; z <= j + k; ++z)
                        states.row(z).array() = strategies;
                } else {
                    for (int64_t z = j; z < total_counting_generations; ++z)
                        states.row(z).array() = strategies;
                }

                // Update state count by k steps
                j += k + 1;
                // Update state after mutation
                if (j <= total_counting_generations) {
                    // mutate
                    birth = _strategy_sampler(_mt);
                    // If population still homogeneous we wait for another mutation
                    while (birth == idx_homo) birth = _strategy_sampler(_mt);
                    strategies(birth) += 1;
                    strategies(idx_homo) -= 1;
                    homogeneous = false;

                    states.row(j).array() = strategies;
                }
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
    MatrixXui2D PairwiseComparisonNumerical<Cache>::run(const int64_t nb_generations, const int64_t transient,
                                                        const double beta,
                                                        const Eigen::Ref<const VectorXui> &init_state) {
        _validate_args(nb_generations, init_state);
        _validate_transient(nb_generations, transient);

        int die, birth, strategy_p1 = 0, strategy_p2 = 0;
        auto total_counting_generations = nb_generations - transient;
        MatrixXui2D states = MatrixXui2D::Zero(total_counting_generations, _nb_strategies);
        // initialise initial state
        VectorXui strategies(_nb_strategies);
        strategies.array() = init_state.eval();

        // Check if state is homogeneous

        // If homogeneous we return a matrix where the population never changes
        if (auto [homogeneous, idx_homo] = _is_homogeneous(strategies); homogeneous) {
            for (int64_t j = 0; j < total_counting_generations; ++j)
                states.row(j).array() = strategies;
            return states;
        }

        // Creates a cache for the fitness data
        Cache cache(_cache_size);

        for (int64_t j = 0; j < transient; ++j) {
            // First we pick 2 players randomly
            _sample_players(strategy_p1, strategy_p2, strategies, _mt);

            if (_update_step(strategy_p1, strategy_p2, beta,
                             birth, die, strategies, cache, _mt)) {
                for (int64_t z = 0; z < total_counting_generations; ++z)
                    states.row(z).array() = strategies;
                return states;
            }
        }

        for (int64_t j = 0; j < total_counting_generations; ++j) {
            // First we pick 2 players randomly
            _sample_players(strategy_p1, strategy_p2, strategies, _mt);

            if (_update_step(strategy_p1, strategy_p2, beta,
                             birth, die, strategies, cache, _mt)) {
                for (int64_t z = j; z < total_counting_generations; ++z)
                    states.row(z).array() = strategies;
                break;
            }

            // update state for the current generation
            states.row(j).array() = strategies;
        }
        return states;
    }

    template<class Cache>
    MatrixXui2D PairwiseComparisonNumerical<Cache>::run(const int64_t nb_generations, const double beta,
                                                        const Eigen::Ref<const VectorXui> &init_state) {
        _validate_args(nb_generations, init_state);

        int die, birth, strategy_p1 = 0, strategy_p2 = 0, current_generation = 1;
        MatrixXui2D states = MatrixXui2D::Zero(nb_generations + 1, _nb_strategies);
        VectorXui strategies(_nb_strategies);
        // initialise initial state
        states.row(0).array() = init_state.eval();
        strategies.array() = init_state.eval();

        // Check if state is homogeneous
        auto [homogeneous, idx_homo] = _is_homogeneous(strategies);

        // If homogeneous we return a matrix where the population never changes
        if (homogeneous) {
            for (int j = 1; j <= nb_generations; ++j)
                states.row(j).array() = strategies;
            return states;
        }

        // Creates a cache for the fitness data
        Cache cache(_cache_size);

        for (int64_t j = current_generation; j <= nb_generations; ++j) {
            // First we pick 2 players randomly
            _sample_players(strategy_p1, strategy_p2, strategies, _mt);

            if (_update_step(strategy_p1, strategy_p2, beta,
                             birth, die, strategies, cache, _mt)) {
                for (int64_t z = j; z <= nb_generations; ++z)
                    states.row(z).array() = strategies;
                break;
            }

            // update state for the current generation
            states.row(j).array() = strategies;
        }
        return states;
    }

    template<class Cache>
    double
    PairwiseComparisonNumerical<Cache>::estimate_fixation_probability(const int invader, const int resident,
                                                                      const size_t runs,
                                                                      const size_t nb_generations,
                                                                      const double beta) {
        if (invader >= static_cast<int>(_nb_strategies) || resident >= static_cast<int>(_nb_strategies))
            throw std::invalid_argument(
                "you must specify a valid index for invader and resident [0, " + std::to_string(_nb_strategies) +
                ")");
        if (invader == resident) throw std::invalid_argument("mutant must be different from resident");

        long int r2m = 0; // resident to mutant count
        long int r2r = 0; // resident to resident count

        // This loop can be done in parallel
#if defined(_OPENMP) && !defined(_MSC_VER)
#pragma omp parallel for reduction(+ : r2m, r2r) default(none) shared(resident, invader, runs, nb_generations, beta)
#endif
        for (size_t i = 0; i < runs; ++i) {
            // Random generators - each thread should have its own generator
            std::mt19937_64 generator(egttools::Random::SeedGenerator::getInstance().getSeed());

            // First we initialize a homogeneous population with the resident strategy
            VectorXui strategies = VectorXui::Zero(_nb_strategies);
            strategies(resident) = _pop_size - 1;
            strategies(invader) = 1;

            // Then we run the Moran Process
            evolve(nb_generations, beta, strategies, generator);

            if (strategies(invader) == 0) {
                ++r2r;
            } else if (strategies(resident) == 0) {
                ++r2m;
            }
        } // end runs loop
        if ((r2m == 0) && (r2r == 0)) return 0.0;
        else
            return static_cast<double>(r2m) / static_cast<double>(r2m + r2r);
    }

    template<class Cache>
    Vector PairwiseComparisonNumerical<Cache>::estimate_gradient_of_selection(const size_t runs, const int invader,
                                                                              const int resident,
                                                                              const double beta) {
        if (invader >= static_cast<int>(_nb_strategies) || resident >= static_cast<int>(_nb_strategies) ||
            invader < 0 || resident < 0)
            throw std::invalid_argument(
                "invader and resident must be valid strategy indices in [0, " + std::to_string(_nb_strategies) + ")");
        if (invader == resident)
            throw std::invalid_argument("invader and resident must be different strategies");

        // T+[k] = number of trials (out of runs) where the invader count increased at state k
        // T-[k] = number of trials where it decreased
        VectorXi t_plus = VectorXi::Zero(static_cast<Eigen::Index>(_pop_size + 1));
        VectorXi t_minus = VectorXi::Zero(static_cast<Eigen::Index>(_pop_size + 1));

#if defined(_OPENMP) && !defined(_MSC_VER)
#pragma omp parallel for reduction(+ : t_plus, t_minus) default(none) shared(invader, resident, runs, beta)
#endif
        for (size_t run = 0; run < runs; ++run) {
            std::mt19937_64 generator(egttools::Random::SeedGenerator::getInstance().getSeed());
            Cache cache(_cache_size);

            for (size_t k = 1; k < _pop_size; ++k) {
                VectorXui strategies = VectorXui::Zero(static_cast<Eigen::Index>(_nb_strategies));
                strategies(resident) = static_cast<size_t>(_pop_size - k);
                strategies(invader) = static_cast<size_t>(k);

                int s1 = 0, s2 = 0;
                // If both sampled players have the same strategy no imitation occurs
                if (_sample_players(s1, s2, strategies, generator)) continue;

                int birth = 0, die = 0;
                _update_step(s1, s2, beta, birth, die, strategies, cache, generator);

                // Record whether invader count moved up or down
                const size_t new_k = strategies(invader);
                if (new_k > k)
                    ++t_plus(static_cast<Eigen::Index>(k));
                else if (new_k < k)
                    ++t_minus(static_cast<Eigen::Index>(k));
            }
        }

        return (t_plus - t_minus).cast<double>() / static_cast<double>(runs);
    }

    template<class Cache>
    auto PairwiseComparisonNumerical<Cache>::estimate_stationary_distribution(
        const size_t nb_runs, const size_t nb_generations,
        const size_t transitory,
        const double beta,
        double mu,
        const double tolerance,
        const size_t check_every) -> Vector {
        if (mu <= 0) {
            throw std::invalid_argument(
                "mu must be > 0. If you want to run a simulation without mutation, "
                "please use the method signature without the mu parameter");
        }
        if (beta < 0) {
            throw std::invalid_argument(
                "beta must be >= 0!");
        }
        // Check if transient > nb_generations
        if (transitory > nb_generations) {
            throw std::invalid_argument(
                "transient must be < than nb_generations!");
        }
        if (nb_runs < 1) {
            throw std::invalid_argument(
                "nb_runs must be >= 1!");
        }
        if (nb_generations < 1) {
            throw std::invalid_argument(
                "nb_generations must be >= 1!");
        }

        const size_t counting_gens = nb_generations - transitory;

        // Helper lambda that runs a batch of `batch_size` runs and accumulates into `sdist`.
        auto run_batch = [&](VectorXui &sdist, const size_t batch_size) {
            // Distribution number of generations for a mutation to happen (shared, read-only)
            std::geometric_distribution<size_t> geometric(mu);

#if defined(_OPENMP) && !defined(_MSC_VER)
#pragma omp parallel for reduction(+ : sdist) default(none) shared(geometric, batch_size, nb_generations, transitory, beta, mu)
#endif
            for (size_t i = 0; i < batch_size; ++i) {
                // Random generators and cache are per-thread to avoid contention
                std::mt19937_64 generator{egttools::Random::SeedGenerator::getInstance().getSeed()};
                Cache cache(_cache_size);

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

                size_t k, j;

                // First we run the simulations for a @param transitory number of generations
                for (j = 0; j < transitory; ++j) {
                    _sample_players(strategy_p1, strategy_p2, strategies, generator);

                    k = _update_multi_step(strategy_p1, strategy_p2, beta, mu,
                                           birth, die, homogeneous, idx_homo,
                                           strategies, cache,
                                           geometric, generator);
                    j += k;
                }

                // Update current state
                current_state = egttools::FinitePopulations::calculate_state(_pop_size, strategies);

                // Then we start counting
                for (; j < nb_generations; ++j) {
                    if (homogeneous) {
                        k = geometric(generator);
                        sdist(static_cast<int64_t>(current_state)) += k + 1;
                        mutate_(generator, birth, idx_homo);

                        strategies(static_cast<int64_t>(birth)) += 1;
                        strategies(idx_homo) -= 1;

                        current_state = egttools::FinitePopulations::calculate_state(_pop_size, strategies);
                        ++sdist(static_cast<int64_t>(current_state));
                        homogeneous = false;

                        j += k;
                    } else {
                        _sample_players(strategy_p1, strategy_p2, strategies, generator);

                        _update_step(strategy_p1, strategy_p2, beta, mu,
                                     birth, die, homogeneous, idx_homo,
                                     strategies, cache, generator);
                        current_state = egttools::FinitePopulations::calculate_state(_pop_size, strategies);
                        ++sdist(static_cast<int64_t>(current_state));
                    }
                }
            }
        };

        // First we initialise the container for the stationary distribution
        VectorXui sdist = VectorXui::Zero(_nb_states);

        if (tolerance <= 0.0) {
            // Original behavior: run all nb_runs without convergence checks
            run_batch(sdist, nb_runs);
            return sdist.cast<double>() / (nb_runs * counting_gens);
        }

        // Tolerance-based early stopping: process runs in batches, check L∞ convergence after each
        const size_t batch_size = (check_every > 0) ? check_every : std::max<size_t>(1, nb_runs / 10);
        Vector prev_estimate = Vector::Zero(_nb_states);
        size_t runs_done = 0;

        while (runs_done < nb_runs) {
            const size_t this_batch = std::min(batch_size, nb_runs - runs_done);
            run_batch(sdist, this_batch);
            runs_done += this_batch;

            Vector current_estimate = sdist.cast<double>() / (static_cast<double>(runs_done) * counting_gens);
            // Total variation distance (L1 norm of difference, unnormalised by factor 2)
            // is the standard convergence metric for Markov chains: bounded, symmetric,
            // handles zero entries, and has the direct interpretation "total probability
            // mass that shifted between consecutive estimates".
            const double l1_change = (current_estimate - prev_estimate).lpNorm<1>();
            prev_estimate = current_estimate;

            if (l1_change < tolerance) {
                return current_estimate;
            }
        }

        return sdist.cast<double>() / (static_cast<double>(runs_done) * counting_gens);
    }

    template<class Cache>
    auto PairwiseComparisonNumerical<Cache>::estimate_stationary_distribution_sparse(const size_t nb_runs,
        const size_t nb_generations,
        const size_t transitory, const double beta,
        double mu,
        const double tolerance,
        const size_t check_every) -> SparseMatrix2D {
        if (mu <= 0) {
            throw std::invalid_argument(
                "mu must be > 0. If you want to run a simulation without mutation, "
                "please use the method signature without the mu parameter");
        }
        if (beta < 0) {
            throw std::invalid_argument(
                "beta must be >= 0!");
        }
        // Check if transient > nb_generations
        if (transitory > nb_generations) {
            throw std::invalid_argument(
                "transient must be < than nb_generations!");
        }
        if (nb_runs < 1) {
            throw std::invalid_argument(
                "nb_runs must be >= 1!");
        }
        if (nb_generations < 1) {
            throw std::invalid_argument(
                "nb_generations must be >= 1!");
        }

        const size_t counting_gens = nb_generations - transitory;

        // Helper lambda that runs a batch of `batch_size` runs and accumulates into `sdist`.
        auto run_batch = [&](SparseMatrix2DXui &sdist, const size_t batch_size) {
            std::geometric_distribution<size_t> geometric(mu);

#if defined(_OPENMP) && !defined(_MSC_VER)
#pragma omp parallel for reduction(+ : sdist) default(none) shared(geometric, batch_size, nb_generations, transitory, beta, mu)
#endif
            for (size_t i = 0; i < batch_size; ++i) {
                // Random generators and cache are per-thread to avoid contention
                std::mt19937_64 generator{egttools::Random::SeedGenerator::getInstance().getSeed()};
                Cache cache(_cache_size);

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

                size_t k, j;

                // Transient phase
                for (j = 0; j < transitory; ++j) {
                    _sample_players(strategy_p1, strategy_p2, strategies, generator);

                    k = _update_multi_step(strategy_p1, strategy_p2, beta, mu,
                                           birth, die, homogeneous, idx_homo,
                                           strategies, cache,
                                           geometric, generator);
                    j += k;
                }

                // Update current state
                current_state = egttools::FinitePopulations::calculate_state(_pop_size, strategies);

                // Counting phase
                for (; j < nb_generations; ++j) {
                    if (homogeneous) {
                        k = geometric(generator);
                        sdist.coeffRef(0, static_cast<signed long>(current_state)) += k + 1;
                        mutate_(generator, birth, idx_homo);

                        strategies(static_cast<int>(birth)) += 1;
                        strategies(idx_homo) -= 1;

                        current_state = egttools::FinitePopulations::calculate_state(_pop_size, strategies);
                        sdist.coeffRef(0, static_cast<signed long>(current_state)) += 1;
                        homogeneous = false;

                        j += k;
                    } else {
                        _sample_players(strategy_p1, strategy_p2, strategies, generator);

                        _update_step(strategy_p1, strategy_p2, beta, mu,
                                     birth, die, homogeneous, idx_homo,
                                     strategies, cache, generator);
                        current_state = egttools::FinitePopulations::calculate_state(_pop_size, strategies);
                        sdist.coeffRef(0, static_cast<signed long>(current_state)) += 1;
                    }
                }
            }
        };

        // First we initialise the container for the stationary distribution
        auto sdist = SparseMatrix2DXui(1, _nb_states);

        if (tolerance <= 0.0) {
            // Original behavior: run all nb_runs without convergence checks
            run_batch(sdist, nb_runs);
            return sdist.cast<double>() / (nb_runs * counting_gens);
        }

        // Tolerance-based early stopping
        const size_t batch_size = (check_every > 0) ? check_every : std::max<size_t>(1, nb_runs / 10);
        Vector prev_estimate = Vector::Zero(_nb_states);
        size_t runs_done = 0;

        while (runs_done < nb_runs) {
            const size_t this_batch = std::min(batch_size, nb_runs - runs_done);
            run_batch(sdist, this_batch);
            runs_done += this_batch;

            // Convert sparse to dense for convergence check (dense L1 comparison)
            Vector current_estimate = Vector(sdist.cast<double>() / (static_cast<double>(runs_done) * counting_gens));
            const double l1_change = (current_estimate - prev_estimate).lpNorm<1>();
            prev_estimate = current_estimate;

            if (l1_change < tolerance) {
                return sdist.cast<double>() / (static_cast<double>(runs_done) * counting_gens);
            }
        }

        return sdist.cast<double>() / (static_cast<double>(runs_done) * counting_gens);
    }

    template<class Cache>
    auto PairwiseComparisonNumerical<Cache>::estimate_strategy_distribution(
        const size_t nb_runs, const size_t nb_generations,
        const size_t transitory, const double beta,
        double mu,
        const double tolerance,
        const size_t check_every) -> Vector {
        // Here we are going to estimate the strategy distribution directly, without the stationary distribution.
        // To do that, we need to keep count of the average frequency of each strategy in the population during the simulation.

        if (mu <= 0) {
            throw std::invalid_argument(
                "mu must be > 0. If you want to run a simulation without mutation, "
                "please use the method signature without the mu parameter");
        }
        if (beta < 0) {
            throw std::invalid_argument(
                "beta must be >= 0!");
        }
        // Check if transient > nb_generations
        if (transitory > nb_generations) {
            throw std::invalid_argument(
                "transient must be < than nb_generations!");
        }
        if (nb_runs < 1) {
            throw std::invalid_argument(
                "nb_runs must be >= 1!");
        }
        if (nb_generations < 1) {
            throw std::invalid_argument(
                "nb_generations must be >= 1!");
        }

        const size_t counting_gens = nb_generations - transitory;

        // Helper lambda that runs a batch of `batch_size` runs and accumulates into `strategy_dist`.
        auto run_batch = [&](VectorXui &strategy_dist, const size_t batch_size) {
            std::geometric_distribution<size_t> geometric(mu);

#if defined(_OPENMP) && !defined(_MSC_VER)
#pragma omp parallel for reduction(+ : strategy_dist) default(none) shared(geometric, batch_size, nb_generations, transitory, beta, mu)
#endif
            for (size_t i = 0; i < batch_size; ++i) {
                // Random generators — each thread gets its own generator
                std::mt19937_64 generator{egttools::Random::SeedGenerator::getInstance().getSeed()};

                // Sample a random population state
                VectorXui strategies = VectorXui::Zero(_nb_strategies);
                auto current_state = _state_sampler(generator);
                egttools::FinitePopulations::sample_simplex(current_state, _pop_size, _nb_strategies, strategies);

                int die = 0, birth = 0, strategy_p1 = 0, strategy_p2 = 0;
                auto [homogeneous, idx_homo] = _is_homogeneous(strategies);

                if (homogeneous) {
                    mutate_(generator, birth, idx_homo);
                    strategies(static_cast<int>(birth)) += 1;
                    strategies(idx_homo) -= 1;
                    homogeneous = false;
                }

                Cache cache(_cache_size);
                size_t k, j;

                // Transient phase
                for (j = 0; j < transitory; ++j) {
                    _sample_players(strategy_p1, strategy_p2, strategies, generator);

                    k = _update_multi_step(strategy_p1, strategy_p2, beta, mu,
                                           birth, die, homogeneous, idx_homo,
                                           strategies, cache,
                                           geometric, generator);
                    j += k;
                }

                // Counting phase
                for (; j < nb_generations; ++j) {
                    if (homogeneous) {
                        k = geometric(generator);
                        strategy_dist += strategies * k;
                        mutate_(generator, birth, idx_homo);

                        strategies(static_cast<int>(birth)) += 1;
                        strategies(idx_homo) -= 1;

                        homogeneous = false;
                        j += k;
                    } else {
                        _sample_players(strategy_p1, strategy_p2, strategies, generator);

                        _update_step(strategy_p1, strategy_p2, beta, mu,
                                     birth, die, homogeneous, idx_homo,
                                     strategies, cache, generator);
                        strategy_dist += strategies;
                    }
                }
            }
        };

        VectorXui strategy_dist = VectorXui::Zero(_nb_strategies);

        if (tolerance <= 0.0) {
            // Original behavior: run all nb_runs without convergence checks
            run_batch(strategy_dist, nb_runs);
            return strategy_dist.cast<double>() / (static_cast<double>(_pop_size) * nb_runs * counting_gens);
        }

        // Tolerance-based early stopping
        const size_t batch_size = (check_every > 0) ? check_every : std::max<size_t>(1, nb_runs / 10);
        Vector prev_estimate = Vector::Zero(_nb_strategies);
        size_t runs_done = 0;

        while (runs_done < nb_runs) {
            const size_t this_batch = std::min(batch_size, nb_runs - runs_done);
            run_batch(strategy_dist, this_batch);
            runs_done += this_batch;

            Vector current_estimate = strategy_dist.cast<double>() /
                                      (static_cast<double>(_pop_size) * runs_done * counting_gens);
            const double l1_change = (current_estimate - prev_estimate).lpNorm<1>();
            prev_estimate = current_estimate;

            if (l1_change < tolerance) {
                return current_estimate;
            }
        }

        return strategy_dist.cast<double>() / (static_cast<double>(_pop_size) * runs_done * counting_gens);
    }

    template<class Cache>
    auto PairwiseComparisonNumerical<Cache>::estimate_stationary_indicators(
        const size_t nb_runs, const size_t nb_generations,
        const size_t transitory, const double beta, double mu,
        const Eigen::Ref<const Matrix2D> &indicator_values,
        const double tolerance,
        const size_t check_every) -> Matrix2D {
        if (mu <= 0) {
            throw std::invalid_argument(
                "mu must be > 0. If you want to run a simulation without mutation, "
                "please use the method signature without the mu parameter");
        }
        if (beta < 0) throw std::invalid_argument("beta must be >= 0!");
        if (transitory > nb_generations)
            throw std::invalid_argument("transitory must be < nb_generations!");
        if (nb_runs < 1) throw std::invalid_argument("nb_runs must be >= 1!");
        if (nb_generations < 1) throw std::invalid_argument("nb_generations must be >= 1!");

        const int64_t nb_indicators = indicator_values.cols();
        if (static_cast<size_t>(indicator_values.rows()) != _nb_states)
            throw std::invalid_argument(
                "indicator_values must have nb_states rows (one per population state).");

        // Pre-allocate output: one row per run (up to nb_runs).
        // Rows beyond runs_done are unused when early stopping triggers.
        Matrix2D per_run_results = Matrix2D::Zero(static_cast<int64_t>(nb_runs), nb_indicators);

        // Lambda that fills rows [start, start+batch_size) of per_run_results.
        auto run_batch = [&](const size_t start, const size_t batch_size) {
            std::geometric_distribution<size_t> geometric(mu);

#if defined(_OPENMP) && !defined(_MSC_VER)
#pragma omp parallel for default(none) \
    shared(per_run_results, start, batch_size, nb_generations, transitory, beta, mu, \
           geometric, indicator_values, nb_indicators)
#endif
            for (size_t i = 0; i < batch_size; ++i) {
                std::mt19937_64 generator{egttools::Random::SeedGenerator::getInstance().getSeed()};
                Cache cache(_cache_size);

                VectorXui strategies = VectorXui::Zero(_nb_strategies);
                auto current_state = _state_sampler(generator);
                egttools::FinitePopulations::sample_simplex(current_state, _pop_size, _nb_strategies, strategies);

                int die = 0, birth = 0, strategy_p1 = 0, strategy_p2 = 0;
                auto [homogeneous, idx_homo] = _is_homogeneous(strategies);
                if (homogeneous) {
                    mutate_(generator, birth, idx_homo);
                    strategies(static_cast<int>(birth)) += 1;
                    strategies(idx_homo) -= 1;
                    homogeneous = false;
                }

                // Accumulator for this run: row vector matching indicator_values columns.
                // Using a row vector avoids transposing on every access.
                Eigen::RowVectorXd run_sum = Eigen::RowVectorXd::Zero(nb_indicators);
                size_t run_count = 0;
                size_t k, j;

                // Transitory phase (no accumulation).
                for (j = 0; j < transitory; ++j) {
                    _sample_players(strategy_p1, strategy_p2, strategies, generator);
                    k = _update_multi_step(strategy_p1, strategy_p2, beta, mu,
                                           birth, die, homogeneous, idx_homo,
                                           strategies, cache, geometric, generator);
                    j += k;
                }

                current_state = egttools::FinitePopulations::calculate_state(_pop_size, strategies);

                // Counting phase.
                for (; j < nb_generations; ++j) {
                    if (homogeneous) {
                        k = geometric(generator);
                        // k+1 steps spent in current_state before mutation.
                        run_sum += indicator_values.row(static_cast<int64_t>(current_state)) * static_cast<double>(k + 1);
                        run_count += k + 1;
                        mutate_(generator, birth, idx_homo);
                        strategies(static_cast<int>(birth)) += 1;
                        strategies(idx_homo) -= 1;
                        current_state = egttools::FinitePopulations::calculate_state(_pop_size, strategies);
                        // 1 step in the post-mutation state.
                        run_sum += indicator_values.row(static_cast<int64_t>(current_state));
                        ++run_count;
                        homogeneous = false;
                        j += k;
                    } else {
                        _sample_players(strategy_p1, strategy_p2, strategies, generator);
                        _update_step(strategy_p1, strategy_p2, beta, mu,
                                     birth, die, homogeneous, idx_homo,
                                     strategies, cache, generator);
                        current_state = egttools::FinitePopulations::calculate_state(_pop_size, strategies);
                        run_sum += indicator_values.row(static_cast<int64_t>(current_state));
                        ++run_count;
                    }
                }

                // Store time-average for this run.
                const auto row_idx = static_cast<int64_t>(start + i);
                if (run_count > 0)
                    per_run_results.row(row_idx) = run_sum / static_cast<double>(run_count);
                else
                    per_run_results.row(row_idx) = run_sum;  // edge case: 0 counting steps
            }
        };

        if (tolerance <= 0.0) {
            run_batch(0, nb_runs);
            return per_run_results;
        }

        // Tolerance-based early stopping: process in batches, check L1 on column means.
        const size_t batch_size = (check_every > 0) ? check_every : std::max<size_t>(1, nb_runs / 10);
        Eigen::RowVectorXd prev_mean = Eigen::RowVectorXd::Zero(nb_indicators);
        size_t runs_done = 0;

        while (runs_done < nb_runs) {
            const size_t this_batch = std::min(batch_size, nb_runs - runs_done);
            run_batch(runs_done, this_batch);
            runs_done += this_batch;

            // Column-wise mean over completed rows.
            Eigen::RowVectorXd current_mean =
                per_run_results.topRows(static_cast<int64_t>(runs_done)).colwise().mean();
            const double l1 = (current_mean - prev_mean).lpNorm<1>();
            prev_mean = current_mean;
            if (l1 < tolerance) break;
        }

        return per_run_results.topRows(static_cast<int64_t>(runs_done)).eval();
    }

    template<class Cache>
    void PairwiseComparisonNumerical<Cache>::mutate_(std::mt19937_64 &generator, int &birth, const int &idx_homo) {
        // mutate
        birth = _strategy_sampler(generator);
        // We assume mutations imply changing strategy
        while (birth == idx_homo) birth = _strategy_sampler(generator);
    }

    template<class Cache>
    bool PairwiseComparisonNumerical<Cache>::_update_step(const int s1, const int s2, double beta, int &birth, int &die,
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
    void PairwiseComparisonNumerical<Cache>::_update_step(const int s1, const int s2, double beta, const double mu,
                                                          int &birth, int &die, bool &homogeneous, int &idx_homo,
                                                          VectorXui &strategies,
                                                          Cache &cache,
                                                          std::mt19937_64 &generator) {
        die = s1;

        if (s1 == s2) {
            // if the strategies are the same, the only change is with mutation
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
            } else {
                // If no mutation, player imitates

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
    PairwiseComparisonNumerical<Cache>::_update_multi_step(const int s1, const int s2, double beta, const double mu,
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
        } else if (s1 == s2) {
            // if the strategies are the same, the only change is with mutation
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
            } else {
                // If no mutation, player imitates

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
    bool
    PairwiseComparisonNumerical<Cache>::_sample_players(int &s1, int &s2, VectorXui &strategies,
                                                        std::mt19937_64 &generator) {
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
    PairwiseComparisonNumerical<
        Cache>::_calculate_fitness(const int &player_type, VectorXui &strategies, Cache &cache) {
        double fitness;
        std::stringstream result;
        result << strategies;

        std::string key = std::to_string(player_type) + result.str();

        // First we check if fitness value is in the lookup table
        if (auto value = cache.get(key); value) {
            fitness = *value;
        } else {
            strategies(player_type) -= 1;
            fitness = _game->calculate_fitness(player_type, _pop_size, strategies);
            strategies(player_type) += 1;

            // Finally we store the new fitness in the Cache. We also keep a Cache for the payoff given each group combination
            cache.put(key, fitness);
        }

        return fitness;
    }

    template<class Cache>
    std::pair<bool, int> PairwiseComparisonNumerical<Cache>::_is_homogeneous(VectorXui &strategies) const {
        for (int i = 0; i < static_cast<int>(_nb_strategies); ++i) {
            if (strategies(i) == _pop_size) return std::make_pair(true, i);
        }
        return std::make_pair(false, -1);
    }

    template<class Cache>
    size_t PairwiseComparisonNumerical<Cache>::nb_strategies() const {
        return _nb_strategies;
    }

    template<class Cache>
    size_t PairwiseComparisonNumerical<Cache>::population_size() const {
        return _pop_size;
    }

    template<class Cache>
    size_t PairwiseComparisonNumerical<Cache>::cache_size() const {
        return _cache_size;
    }

    template<class Cache>
    int64_t PairwiseComparisonNumerical<Cache>::nb_states() const {
        return _nb_states;
    }

    template<class Cache>
    std::string PairwiseComparisonNumerical<Cache>::game_type() const {
        return _game->type();
    }

    template<class Cache>
    const GroupPayoffs &PairwiseComparisonNumerical<Cache>::payoffs() const {
        return _game->payoffs();
    }

    template<class Cache>
    void PairwiseComparisonNumerical<Cache>::set_population_size(const size_t pop_size) {
        _pop_size = pop_size;
        _nb_states = egttools::starsBars(_pop_size, _nb_strategies);
        _state_sampler = std::uniform_int_distribution<size_t>(0, _nb_states - 1);
        _pop_sampler = std::uniform_int_distribution<size_t>(0, _pop_size - 1);
    }

    template<class Cache>
    void PairwiseComparisonNumerical<Cache>::set_cache_size(const size_t cache_size) {
        _cache_size = cache_size;
    }

    template<class Cache>
    void PairwiseComparisonNumerical<Cache>::change_game(egttools::FinitePopulations::AbstractGame &game) {
        _game = &game;
    }

    template<class Cache>
    void PairwiseComparisonNumerical<Cache>::_validate_args(const int64_t nb_generations,
                                                            const Eigen::Ref<const VectorXui> &init_state) const {
        if (nb_generations <= 0)
            throw std::invalid_argument("nb_generations must be > 0");
        if (init_state.size() != static_cast<int64_t>(_nb_strategies))
            throw std::invalid_argument(
                "init_state length must equal nb_strategies (" + std::to_string(_nb_strategies) + ")");
        if (static_cast<size_t>(init_state.sum()) != _pop_size)
            throw std::invalid_argument(
                "init_state entries must sum to pop_size Z=" + std::to_string(_pop_size));
    }

    template<class Cache>
    void PairwiseComparisonNumerical<Cache>::_validate_transient(const int64_t nb_generations,
                                                                 const int64_t transient) const {
        if (transient < 0)
            throw std::invalid_argument("transient must be >= 0");
        if (transient >= nb_generations)
            throw std::invalid_argument("transient must be < nb_generations");
    }

} // namespace egttools::FinitePopulations

#endif//EGTTOOLS_FINITEPOPULATIONS_PAIRWISECOMPARISONNUMERICAL_HPP

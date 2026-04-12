/** Copyright (c) 2019-2026  Elias Fernandez
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
#ifndef EGTTOOLS_FINITEPOPULATIONS_GAMES_ABSTRACTNPLAYERSTATEGAME_HPP
#define EGTTOOLS_FINITEPOPULATIONS_GAMES_ABSTRACTNPLAYERSTATEGAME_HPP

#include <egttools/finite_populations/games/AbstractNPlayerGame.hpp>

namespace egttools::FinitePopulations {

    /**
     * @brief Abstract base class for N-player games with state-dependent payoffs.
     *
     * This class is designed for games where payoffs cannot be precomputed at
     * initialization because they depend on the current population state (e.g.,
     * games with variable risk functions that change with population composition).
     *
     * Concrete Python subclasses must implement `get_payoffs_for_player`, which
     * is called once per `calculate_fitness` invocation. The C++ base class then
     * runs the hypergeometric sampling loop entirely in C++, dramatically reducing
     * the number of Python call-throughs compared to a pure Python implementation
     * (from O(nb_group_configurations) calls to exactly 1 per fitness evaluation).
     *
     * @note The `expected_payoffs_` matrix inherited from `AbstractNPlayerGame`
     *       is not used for fitness computation in this class. Subclasses may
     *       optionally populate it via `calculate_payoffs()` for inspection, but
     *       it is not required.
     *
     * Usage
     * -----
     * Python subclasses should inherit from `egttools.games.AbstractNPlayerStateGame`
     * and implement:
     *
     * - `get_payoffs_for_player(player_type, state_index, state) -> np.ndarray`
     *   Returns a 1-D array of length `nb_group_configurations` with the payoff
     *   for `player_type` under each possible group configuration, given that the
     *   current full population state (including the focal player) has index
     *   `state_index`.
     *
     * - All other abstract methods inherited from `AbstractNPlayerGame` / `AbstractGame`.
     */
    class AbstractNPlayerStateGame : public AbstractNPlayerGame {
    public:
        AbstractNPlayerStateGame(int nb_strategies, int group_size);

        /**
         * @brief Returns the payoff row for `player_type` across all group configurations.
         *
         * This is the single virtual method that Python subclasses must implement.
         * It is called once per `calculate_fitness` invocation, with:
         *
         * @param player_type   Index of the focal player's strategy.
         * @param state_index   Linear index of the full population state (including
         *                      the focal player), as returned by `calculate_state`.
         * @param state         Population state vector *excluding* the focal player
         *                      (same as the `strategies` argument to `calculate_fitness`).
         * @return              A vector of length `nb_group_configurations_` containing
         *                      the payoff for `player_type` in each possible group
         *                      composition drawn from `state`.
         */
        virtual egttools::Vector get_payoffs_for_player(
            int player_type,
            int64_t state_index,
            const Eigen::Ref<const VectorXui> &state) = 0;

        /**
         * @brief Computes the fitness of `player_type` in a population with state `strategies`.
         *
         * This implementation calls `get_payoffs_for_player` once to obtain the full
         * payoff row for the focal player, then evaluates the hypergeometric expectation
         * entirely in C++.
         *
         * @param player_type  Index of the focal player's strategy.
         * @param pop_size     Total population size (excluding the focal player).
         * @param strategies   Strategy counts in the population, excluding the focal player.
         * @return             Expected fitness of `player_type`.
         */
        double calculate_fitness(const int &player_type, const size_t &pop_size,
                                 const Eigen::Ref<const VectorXui> &strategies) override;

        [[nodiscard]] std::string toString() const override;
        [[nodiscard]] std::string type() const override;
    };

}// namespace egttools::FinitePopulations

#endif//EGTTOOLS_FINITEPOPULATIONS_GAMES_ABSTRACTNPLAYERSTATEGAME_HPP

/** Copyright (c) 2019-2020  Elias Fernandez
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
#ifndef EGTTOOLS_FINITEPOPULATIONS_GAMES_ABSTRACTNPLAYERGAME_HPP
#define EGTTOOLS_FINITEPOPULATIONS_GAMES_ABSTRACTNPLAYERGAME_HPP

#include <egttools/Distributions.h>
#include <egttools/Types.h>
#include <egttools/Utils.h>

#include <cassert>
#include <egttools/finite_populations/Utils.hpp>
#include <egttools/finite_populations/games/AbstractGame.hpp>
#include <fstream>
#include <ios>
#include <ostream>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(_OPENMP)
#include <egttools/OpenMPUtils.hpp>
#endif

namespace egttools::FinitePopulations {
    using PayoffVector = std::vector<double>;

    /**
     * @brief This class defines the interface of a game to be used in an evolutionary process.
     */
    class AbstractNPlayerGame : public AbstractGame {
    public:
        AbstractNPlayerGame(int nb_strategies, int group_size);


        /**
         * @brief updates the vector of payoffs with the payoffs of each player after playing the game.
         *
         * This method will run the game using the players and player types defined in @param group_composition,
         * and will update the vector @param game_payoffs with the resulting payoff of each player.
         *
         * @param group_composition number of players of each strategy in the group
         * @param game_payoffs container for the payoffs of each player
         */
        void play(const egttools::FinitePopulations::StrategyCounts &group_composition,
                  PayoffVector &game_payoffs) override = 0;

        /**
         * @brief Estimates the payoff matrix for each strategy.
         *
         * @return a payoff matrix
         */
        const GroupPayoffs &calculate_payoffs() override = 0;

        /**
         * @brief Estimates the fitness for a @param player_type in the population with state @param strategies.
         *
         * This function assumes that the player with strategy @param player_type is not included in
         * the vector of strategy counts @param strategies.
         *
         * @param player_type : index of the strategy used by the player
         * @param pop_size : size of the population
         * @param strategies : current state of the population
         * @return a fitness value
         */
        double
        calculate_fitness(const int &player_type, const size_t &pop_size, const Eigen::Ref<const VectorXui> &strategies) override;

        [[nodiscard]] size_t nb_strategies() const override;

        [[nodiscard]] virtual size_t group_size() const;

        [[nodiscard]] virtual int64_t nb_group_configurations() const;

        /**
         * @return Returns a small description of the game.
         */
        [[nodiscard]] std::string toString() const override;

        /**
         *
         * @return The type of game
         */
        [[nodiscard]] std::string type() const override;

        /**
         *
         * @return payoff matrix of the game
         */
        [[nodiscard]] const GroupPayoffs &payoffs() const override;

        /**
         * @brief returns the payoff of a strategy given a group composition
         *
         * If the group composition does not include the strategy, the payoff should be zero
         *
         * @param strategy : index of the strategy
         * @param group_composition : vector with the group composition
         * @return the payoff value
         */
        [[nodiscard]] double payoff(int strategy, const egttools::FinitePopulations::StrategyCounts &group_composition) const override;

        virtual void update_payoff(int strategy_index, int group_configuration_index, double value);

        /**
         * @brief stores the payoff matrix in a txt file
         *
         * @param file_name : name of the file in which the data will be stored
         */
        void save_payoffs(std::string file_name) const override;

    protected:
        int nb_strategies_, group_size_;
        int64_t nb_group_configurations_;

        GroupPayoffs expected_payoffs_;
    };
}// namespace egttools::FinitePopulations

#endif//EGTTOOLS_FINITEPOPULATIONS_GAMES_ABSTRACTNPLAYERGAME_HPP

/** Copyright (c) 2020-2021  Elias Fernandez
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
#ifndef EGTTOOLS_FINITEPOPULATIONS_GAMES_NORMALFORMGAME_HPP
#define EGTTOOLS_FINITEPOPULATIONS_GAMES_NORMALFORMGAME_HPP

#include <egttools/Distributions.h>
#include <egttools/Types.h>

#include <cassert>
#include <egttools/finite_populations/Utils.hpp>
#include <egttools/finite_populations/behaviors/AbstractNFGStrategy.hpp>
#include <egttools/finite_populations/games/AbstractGame.hpp>
#include <fstream>
#include <stdexcept>
#include <vector>

#if defined(_OPENMP)
#include <egttools/OpenMPUtils.hpp>
#endif

namespace egttools::FinitePopulations {
    using PayoffVector = std::vector<double>;
    using AbstractNFGStrategy = egttools::FinitePopulations::behaviors::AbstractNFGStrategy;
    using NFGStrategyVector = std::vector<AbstractNFGStrategy *>;

    class NormalFormGame final : public egttools::FinitePopulations::AbstractGame {
    public:
        /**
        * @brief This class implements a normal form game.
        *
        * The payoffs of the game are defined by a payoff matrix.
        * It is always a 2-player game, but may contain more than 2 possible actions.
        *
        * If @param nb_rounds > 1, than the game is iterated (has more than 1 round).
        *
        * In case the number of rounds is > 1, this class will estimate
        * The expected payoff for each strategy and update it's own internal
        * payoff matrix.
        *
        * The drawback of this method is that is that if the number of strategies is high,
        * it might take a long time to precalculate the payoffs. In cases where the payoffs of each
        * strategy are analytically defined, it might be best input the payoff matrix with the expected
        * payoffs for all strategies and indicate that the game takes only 1 round.
        * In this latter case, the cooperation vector has no meaning
        *
        * @param nb_rounds : number of rounds of the game.
        * @param payoff_matrix : Eigen matrix containing the payoffs.
        */
        NormalFormGame(int64_t nb_rounds, const Eigen::Ref<const Matrix2D> &payoff_matrix);

        NormalFormGame(int64_t nb_rounds, const Eigen::Ref<const Matrix2D> &payoff_matrix, const NFGStrategyVector &strategies);

        //        NormalFormGame(size_t nb_rounds, const Eigen::Ref<const Matrix2D> &payoff_matrix, const std::vector<std::string> strategies);

        void play(const egttools::FinitePopulations::StrategyCounts &group_composition,
                  PayoffVector &game_payoffs) override;

        /**
         * @brief Gets an action from the strategy defined by player type.
         *
         * This method will call one of the behaviors specified in CrdBehaviors.hpp indexed by
         * @param player_type with the parameters @param prev_donation, threshold, current_round.
         *
         * @param player_type : type of strategy (as an unsigned integer).
         * @param       prev_action : previous donation of the group.
         * @param current_round : current round of the game
         * @return action of the strategy
         */
        //  static inline size_t get_action(const size_t &player_type, const size_t &prev_action, const size_t &current_round);

        /**
        * @brief updates private payoff matrix and returns it
        *
        * @return payoff matrix of the game
        */
        const GroupPayoffs &calculate_payoffs() override;

        double
        calculate_fitness(const int &player_type, const size_t &pop_size,
                          const Eigen::Ref<const VectorXui> &strategies) override;

        /**
        * @brief Calculates the expected level of cooperation given a population state
        * @param pop_size : size of the population
        *       @param population_state : state of the population (number of players of each strategy)
        * @return the level of cooperation of that population state
        */
        double calculate_cooperation_level(size_t pop_size, const Eigen::Ref<const VectorXui> &population_state);

        // getters
        [[nodiscard]] size_t nb_strategies() const override;
        [[nodiscard]] size_t nb_rounds() const;
        [[nodiscard]] size_t nb_states() const;
        [[nodiscard]] std::string toString() const override;
        [[nodiscard]] std::string type() const override;
        [[nodiscard]] const GroupPayoffs &payoffs() const override;
        [[nodiscard]] const Matrix2D &expected_payoffs() const;
        [[nodiscard]] double payoff(int strategy, const egttools::FinitePopulations::StrategyCounts &group_composition) const override;
        [[nodiscard]] const NFGStrategyVector &strategies() const;
        void save_payoffs(std::string file_name) const override;

        // setters

    protected:
        int64_t nb_rounds_, nb_strategies_, nb_states_;
        Matrix2D payoffs_, expected_payoffs_, coop_level_;
        NFGStrategyVector strategies_;

        /**
        * @brief updates the expected_payoffs_ and coop_level_ matrices for the strategies indicates
        * @param s1 : strategy 1
        * @param s2 : strategy 2
        */
        void _update_cooperation_and_payoffs(int s1, int s2);
    };

}// namespace egttools::FinitePopulations

#endif//EGTTOOLS_FINITEPOPULATIONS_GAMES_NORMALFORMGAME_HPP

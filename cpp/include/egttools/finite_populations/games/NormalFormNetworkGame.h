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
#ifndef EGTTOOLS_FINITEPOPULATIONS_GAMES_NORMALFORMNETWORKGAME_HPP
#define EGTTOOLS_FINITEPOPULATIONS_GAMES_NORMALFORMNETWORKGAME_HPP

#include <egttools/Distributions.h>
#include <egttools/Types.h>

#include <cassert>
#include <egttools/finite_populations/Utils.hpp>
#include <egttools/finite_populations/behaviors/AbstractNFGStrategy.hpp>
#include <egttools/finite_populations/behaviors/NFGStrategies.hpp>
#include <egttools/finite_populations/games/AbstractSpatialGame.hpp>
#include <memory>
#include <stdexcept>
#include <vector>

namespace egttools::FinitePopulations::games {
    using AbstractNFGStrategy_ptr = egttools::FinitePopulations::behaviors::AbstractNFGStrategy *;
    using NFGStrategyVector = std::vector<AbstractNFGStrategy_ptr>;

    class NormalFormNetworkGame final : public egttools::FinitePopulations::games::AbstractSpatialGame {
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
        * @param nb_rounds : number of rounds of the game.
        * @param payoff_matrix : Eigen matrix containing the payoffs.
        */
        NormalFormNetworkGame(int nb_rounds, const Eigen::Ref<const Matrix2D> &payoff_matrix);

        NormalFormNetworkGame(int nb_rounds, const Eigen::Ref<const Matrix2D> &payoff_matrix, NFGStrategyVector &strategies);

        void calculate_payoffs();

        double calculate_fitness(int strategy_index, VectorXui &state) override;

        /**
        * @brief Calculates the average cooperation of a player with its neighborhood
        * @param index_strategy_focal : index of the strategy of the focal individual
        * @param neighborhood_state : counts of each strategy in the neighborhood
        * @return the level of cooperation of a strategy given the neighborhood
        */
        double calculate_cooperation_level_neighborhood(int index_strategy_focal, const Eigen::Ref<const VectorXui> &neighborhood_state);

        // getters
        [[nodiscard]] int nb_strategies() const override;
        [[nodiscard]] size_t nb_rounds() const;
        [[nodiscard]] std::string toString() const override;
        [[nodiscard]] std::string type() const override;
        [[nodiscard]] const NFGStrategyVector &strategies() const;
        [[nodiscard]] const Matrix2D &expected_payoffs() const;

        // setters
        //        void set_strategies(NFGStrategyVector &strategies);

    protected:
        int nb_rounds_, nb_strategies_;
        Matrix2D payoff_matrix_, expected_payoffs_, coop_level_;

        NFGStrategyVector strategies_;

        /**
        * @brief updates the expected_payoffs_ and coop_level_ matrices for the strategies indicates
        * @param s1 : strategy 1
        * @param s2 : strategy 2
        */
        void update_cooperation_and_payoffs_(int s1, int s2);

        //        inline double calculate_payoff_not_stochastic(std::tuple<int, int> &key);
        //        inline double calculate_payoff_stochastic(std::tuple<int, int> &key);
    };

    //    template<class Cache>
    //    double egttools::FinitePopulations::games::NormalFormNetworkGame<Cache>::play(int strategy_index_focal, int strategy_index_opponent) {
    //        auto key = std::tuple<int, int>(strategy_index_focal, strategy_index_opponent);
    //        double payoff;
    //
    //        if (cache_.exists(key)) {
    //            payoff = cache_.get(key);
    //        } else {
    //            // Check if any of the strategies is stochastic, in which case repeat the loop 10000 times (for good statistics)
    //            bool is_stochastic = strategies_[strategy_index_focal]->is_stochastic() || strategies_[strategy_index_opponent]->is_stochastic();
    //
    //            if (is_stochastic) {
    //                payoff = calculate_payoff_stochastic(key);
    //            } else {
    //                payoff = calculate_payoff_not_stochastic(key);
    //            }
    //        }
    //
    //        return payoff;
    //    }
    //
    //    template<class Cache>
    //    inline double egttools::FinitePopulations::games::NormalFormNetworkGame<Cache>::calculate_payoff_not_stochastic(std::tuple<int, int> &key) {
    //        int action1, action1_prev = 0, action2 = 0;
    //        double payoff1 = 0, payoff2 = 0;
    //
    //        auto index1 = std::get<0>(key);
    //        auto index2 = std::get<1>(key);
    //
    //        for (int i = 0; i < nb_rounds_; ++i) {
    //            // For now since we will consider only C and D as strategies, we will pre-calculate all actions since they
    //            // correspond with the strategies
    //            action1 = strategies_[index1]->get_action(i, action2);
    //            action2 = strategies_[index2]->get_action(i, action1_prev);
    //            action1_prev = action1;
    //            payoff1 += payoff_matrix_(action1, action2);
    //            payoff2 += payoff_matrix_(action2, action1);
    //        }
    //        payoff1 /= static_cast<double>(nb_rounds_);
    //        payoff2 /= static_cast<double>(nb_rounds_);
    //
    //        cache_.insert(key, payoff1);
    //        cache_.insert(std::tuple<int, int>(index2, index1), payoff2);
    //
    //        return payoff1;
    //    }
    //
    //    template<class Cache>
    //    inline double egttools::FinitePopulations::games::NormalFormNetworkGame<Cache>::calculate_payoff_stochastic(std::tuple<int, int> &key) {
    //        int action1, action1_prev = 0, action2 = 0;
    //
    //        int divisor = nb_rounds_ * 10000;
    //        double payoff1 = 0, payoff2 = 0;
    //
    //        auto index1 = std::get<0>(key);
    //        auto index2 = std::get<1>(key);
    //
    //        // Iterate 10000 times and average payoffs
    //        for (int j = 0; j < 10000; ++j) {
    //            for (int i = 0; i < nb_rounds_; ++i) {
    //                // For now since we will consider only C and D as strategies, we will pre-calculate all actions since they
    //                // correspond with the strategies
    //                action1 = strategies_[index1]->get_action(i, action2);
    //                action2 = strategies_[index2]->get_action(i, action1_prev);
    //                action1_prev = action1;
    //                payoff1 += payoff_matrix_(action1, action2);
    //                payoff2 += payoff_matrix_(action2, action1);
    //            }
    //        }
    //
    //        payoff1 /= static_cast<double>(divisor);
    //        payoff2 /= static_cast<double>(divisor);
    //
    //        cache_.insert(key, payoff1);
    //        cache_.insert(std::tuple<int, int>(index2, index1), payoff2);
    //
    //        return payoff1;
    //    }

}// namespace egttools::FinitePopulations::games

#endif//EGTTOOLS_FINITEPOPULATIONS_GAMES_NORMALFORMNETWORKGAME_HPP

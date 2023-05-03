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
#ifndef EGTTOOLS_FINITEPOPULATIONS_GAMES_NETWORKONESHOTCRD_HPP
#define EGTTOOLS_FINITEPOPULATIONS_GAMES_NETWORKONESHOTCRD_HPP

#include <egttools/Distributions.h>
#include <egttools/Types.h>

#include <cassert>
#include <egttools/LruCache.hpp>
#include <egttools/finite_populations/Utils.hpp>
#include <egttools/finite_populations/games/AbstractSpatialGame.hpp>
#include <memory>
#include <stdexcept>
#include <vector>

namespace egttools::FinitePopulations::games {
    class OneShotCRDNetworkGame final : public egttools::FinitePopulations::games::AbstractSpatialGame {
    public:
        /**
         * @brief This class implements a One-Shot Collective Risk Dilemma on networks.
         *
         *  This N-player game was first introduced in "Santos, F. C., & Pacheco, J. M. (2011).
         *  Risk of collective failure provides an escape from the tragedy of the commons.
         *  Proceedings of the National Academy of Sciences of the United States of America, 108(26), 10421–10425.".*
         *
         *  The game consists of a group of size ``group_size`` (N) which can be composed of
         *  Cooperators (Cs) who will contribute a fraction ``cost`` (c) of their
         *  ``endowment`` (b) to the public good. And of Defectors (Ds) who contribute 0.
         *
         *  If the total contribution of the group is equal or surpasses the collective target Mcb,
         *  with M being the ``min_nb_cooperators``, then all participants will receive as payoff
         *  their remaining endowment. Which is, Cs receive b - cb and Ds receive b. Otherwise, all
         *  participants receive 0 endowment with a probability equal to ``risk`` (r), and will
         *  keep their endowment with probability 1-r. This means that each group must have at least
         *  M Cs for the collective target to be achieved.
         *
         * @param endowment : The initial endowment (b) received by all participants
         * @param cost : The fraction of the endowment that Cooperators contribute to the public good.
         *               This value must be in the interval [0, 1]
         * @param risk : The risk that all members of the group will lose their remaining endowment if the
         *               collective target is not achieved.
         * @param group_size : The size of the group (N)
         * @param min_nb_cooperators : The minimum number of cooperators (M) required to reach the
         *                             collective target. In other words, the collective target is
         *                             reached if the collective effort is at least Mcb. This value
         *                             must be in the discrete interval [[0, N]].
         */
        OneShotCRDNetworkGame(double endowment, double cost, double risk, int group_size, int min_nb_cooperators);

        double calculate_fitness(int strategy_index, VectorXui &state) override;

        /**
        * @brief Calculates the average cooperation of a player with its neighborhood
        * @param index_strategy_focal : index of the strategy of the focal individual
        * @param neighborhood_state : counts of each strategy in the neighborhood
        * @return the level of cooperation of a strategy given the neighborhood
        */
        double calculate_cooperation_level_neighborhood(int index_strategy_focal, const Eigen::Ref<const VectorXui> &neighborhood_state);

        /**
         * @brief Calculates the payoff of a focal player in its neighborhood
         * @param index_strategy_focal
         * @param neighborhood_state
         * @return
         */
        double calculate_expected_payoff_neighborhood(int index_strategy_focal, const Eigen::Ref<const VectorXui> &neighborhood_state);

        // getters
        [[nodiscard]] int nb_strategies() const override;
        [[nodiscard]] double endowment() const;
        [[nodiscard]] double risk() const;
        [[nodiscard]] int group_size() const;
        [[nodiscard]] int min_nb_cooperators() const;
        [[nodiscard]] std::string toString() const override;
        [[nodiscard]] std::string type() const override;

    protected:
        double endowment_, cost_, risk_;
        int group_size_, min_nb_cooperators_;
    };
}// namespace egttools::FinitePopulations::games

#endif//EGTTOOLS_FINITEPOPULATIONS_GAMES_NETWORKONESHOTCRD_HPP

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
#include <egttools/finite_populations/games/NormalFormNetworkGame.h>

#include <utility>


egttools::FinitePopulations::games::NormalFormNetworkGame::NormalFormNetworkGame(int nb_rounds,
                                                                                 const Eigen::Ref<const Matrix2D> &payoff_matrix) : nb_rounds_(nb_rounds),
                                                                                                                                    payoff_matrix_(payoff_matrix) {
    nb_strategies_ = 2;
    strategies_ = NFGStrategyVector();

    expected_payoffs_ = Matrix2D::Zero(nb_strategies_, nb_strategies_);
    coop_level_ = Matrix2D::Zero(nb_strategies_, nb_strategies_);

    // Initialize payoff matrix
    expected_payoffs_.array() = payoff_matrix_.array();
    coop_level_(0, 0) = 0;
    coop_level_(0, 1) = 0;
    coop_level_(1, 0) = 1;
    coop_level_(1, 1) = 1;
}


egttools::FinitePopulations::games::NormalFormNetworkGame::NormalFormNetworkGame(int nb_rounds,
                                                                                 const Eigen::Ref<const Matrix2D> &payoff_matrix,
                                                                                 NFGStrategyVector &strategies)
    : nb_rounds_(nb_rounds),
      payoff_matrix_(payoff_matrix),
      strategies_(strategies) {

    nb_strategies_ = static_cast<int>(strategies_.size());

    expected_payoffs_ = Matrix2D::Zero(nb_strategies_, nb_strategies_);
    coop_level_ = Matrix2D::Zero(nb_strategies_, nb_strategies_);

    // Initialize payoff matrix
    egttools::FinitePopulations::games::NormalFormNetworkGame::calculate_payoffs();
}

void egttools::FinitePopulations::games::NormalFormNetworkGame::calculate_payoffs() {
    int s1, s2;

    // For every possible pair combination, run the game and store the payoff of each strategy
    for (int i = 0; i < nb_strategies_; ++i) {
        s1 = i;
        for (int j = i; j < nb_strategies_; ++j) {
            // Update group composition from current
            s2 = j;

            // play game and update game_payoffs
            update_cooperation_and_payoffs_(s1, s2);
        }
    }
}

double egttools::FinitePopulations::games::NormalFormNetworkGame::calculate_fitness(int strategy_index, VectorXui &state) {
    double payoff = 0;

    for (int i = 0; i < nb_strategies_; ++i) {
        payoff += static_cast<int>(state(i)) * expected_payoffs_(strategy_index, i);
    }

    return payoff;
}

double egttools::FinitePopulations::games::NormalFormNetworkGame::calculate_cooperation_level_neighborhood(int index_strategy_focal,
                                                                                                           const Eigen::Ref<const VectorXui> &neighborhood_state) {
    // The idea here is to count how often a strategy cooperates
    double coop_level = 0.0;

    for (int i = 0; i < nb_strategies_; ++i) {
        coop_level += static_cast<int>(neighborhood_state(i)) * coop_level_(index_strategy_focal, i);
    }


    return coop_level / static_cast<double>(neighborhood_state.sum());
}

void egttools::FinitePopulations::games::NormalFormNetworkGame::update_cooperation_and_payoffs_(int s1, int s2) {
    // Initialize payoffs
    size_t coop1 = 0, coop2 = 0;
    int action1, action1_prev = 0, action2 = 0;
    int div;

    // Check if any of the strategies is stochastic, in which case repeat the loop 10000 times (for good statistics)
    bool is_stochastic = strategies_[s1]->is_stochastic() || strategies_[s2]->is_stochastic();

    // This should be repeated many times if the strategies are stochastic
    // First we play the game
    if (is_stochastic) {
        div = static_cast<int>(nb_rounds_) * 10000;
        for (int j = 0; j < 10000; ++j) {
            for (int i = 0; i < nb_rounds_; ++i) {
                action1 = strategies_[s1]->get_action(i, action2);
                action2 = strategies_[s2]->get_action(i, action1_prev);
                action1_prev = action1;
                expected_payoffs_(s1, s2) += payoff_matrix_(action1, action2);
                coop1 += action1;
                if (s1 != s2) {
                    expected_payoffs_(s2, s1) += payoff_matrix_(action2, action1);
                    coop2 += action2;
                }
            }
        }
    } else {
        div = static_cast<int>(nb_rounds_);
        for (int i = 0; i < nb_rounds_; ++i) {
            action1 = strategies_[s1]->get_action(i, action2);
            action2 = strategies_[s2]->get_action(i, action1_prev);
            action1_prev = action1;
            expected_payoffs_(s1, s2) += payoff_matrix_(action1, action2);
            coop1 += action1;
            if (s1 != s2) {
                expected_payoffs_(s2, s1) += payoff_matrix_(action2, action1);
                coop2 += action2;
            }
        }
    }

    if (s1 == s2) {
        expected_payoffs_(s1, s1) /= div;
        coop_level_(s1, s1) /= div;
    } else {
        expected_payoffs_(s1, s2) /= div;
        expected_payoffs_(s2, s1) /= div;
        coop_level_(s1, s2) = static_cast<double>(coop1) / div;
        coop_level_(s2, s1) = static_cast<double>(coop2) / div;
    }
}

int egttools::FinitePopulations::games::NormalFormNetworkGame::nb_strategies() const {
    return nb_strategies_;
}

size_t egttools::FinitePopulations::games::NormalFormNetworkGame::nb_rounds() const {
    return nb_rounds_;
}

std::string egttools::FinitePopulations::games::NormalFormNetworkGame::toString() const {
    return "Normal form game on Networks";
}

std::string egttools::FinitePopulations::games::NormalFormNetworkGame::type() const {
    return "egttools.games.NormalFormNetworkGame";
}

const egttools::FinitePopulations::games::NFGStrategyVector &egttools::FinitePopulations::games::NormalFormNetworkGame::strategies() const {
    return strategies_;
}

const egttools::Matrix2D &egttools::FinitePopulations::games::NormalFormNetworkGame::expected_payoffs() const {
    return expected_payoffs_;
}

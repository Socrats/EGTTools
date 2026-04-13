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

#include <egttools/finite_populations/games/AbstractNPlayerStateGame.hpp>
#include <egttools/finite_populations/Utils.hpp>
#include <egttools/utils/CalculateExpectedIndicators.h>

egttools::FinitePopulations::AbstractNPlayerStateGame::AbstractNPlayerStateGame(
    int nb_strategies, int group_size)
    : AbstractNPlayerGame(nb_strategies, group_size) {}

double egttools::FinitePopulations::AbstractNPlayerStateGame::calculate_fitness(
    const int &player_type,
    const size_t &pop_size,
    const Eigen::Ref<const VectorXui> &strategies) {

    // Build the full population state including the focal player.
    VectorXui full_state = strategies;
    full_state(player_type) += 1;

    // Map the full population state to its linear index in the population
    // simplex.  The first argument to calculate_state is the total sum of
    // the state vector — here that is pop_size (not group_size_).  Using
    // group_size_ here would collapse almost all states to index 0 for any
    // population larger than the group, silently corrupting payoff lookups.
    const auto state_index = static_cast<int64_t>(
        egttools::FinitePopulations::calculate_state(
            pop_size, full_state));

    // Ask the (Python) subclass for the payoff row — exactly 1 Python call.
    const egttools::Vector payoffs_row = get_payoffs_for_player(player_type, state_index, strategies);

    // Run the hypergeometric sampling loop entirely in C++.
    return egttools::utils::calculate_hypergeometric_fitness(
        player_type, pop_size,
        static_cast<size_t>(group_size_),
        static_cast<size_t>(nb_strategies_),
        strategies, payoffs_row);
}

std::string egttools::FinitePopulations::AbstractNPlayerStateGame::toString() const {
    return "AbstractNPlayerStateGame";
}

std::string egttools::FinitePopulations::AbstractNPlayerStateGame::type() const {
    return "egttools.games.AbstractNPlayerStateGame";
}

/** Copyright (c) 2019-2023  Elias Fernandez
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
AbstractSpatialGame  * You should have received a copy of the GNU General Public License
  * along with EGTtools.  If not, see <http://www.gnu.org/licenses/>
*/
#pragma once
#ifndef EGTTOOLS_FINITEPOPULATIONS_GAMES_ABSTRACTSPATIALGAME_HPP
#define EGTTOOLS_FINITEPOPULATIONS_GAMES_ABSTRACTSPATIALGAME_HPP

#include <egttools/Types.h>

namespace egttools::FinitePopulations::games {
    using PayoffVector = std::vector<double>;

    /**
     * @brief This class defines the interface of a game to be used in an evolutionary process.
     */
    class AbstractSpatialGame {
    public:
        virtual ~AbstractSpatialGame() = default;

        virtual double calculate_fitness(int strategy_index, VectorXui &state) = 0;


        [[nodiscard]] virtual int nb_strategies() const = 0;

        /**
         * @return Returns a small description of the game.
         */
        [[nodiscard]] virtual std::string toString() const = 0;

        /**
         *
         * @return The type of game
         */
        [[nodiscard]] virtual std::string type() const = 0;
    };
}// namespace egttools::FinitePopulations::games

#endif//EGTTOOLS_FINITEPOPULATIONS_GAMES_ABSTRACTSPATIALGAME_HPP

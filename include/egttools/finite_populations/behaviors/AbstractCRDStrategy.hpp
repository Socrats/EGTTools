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

#ifndef EGTTOOLS_FINITEPOPULATIONS_BEHAVIORS_ABSTRACTCRDSTRATEGY_HPP
#define EGTTOOLS_FINITEPOPULATIONS_BEHAVIORS_ABSTRACTCRDSTRATEGY_HPP

namespace egttools::FinitePopulations::behaviors {

    /**
     * @brief defines the interface for strategies that can be used with AbstractGame (and child classes)
     */
    class AbstractCRDStrategy {
    public:
        virtual ~AbstractCRDStrategy() = default;
        /**
         * Function that will return the decision of the strategy
         * given the current @param time_step (or round) and the
         * previous contributions of the group @param group_contributions_prev.
         *
         * The strategies may take more information into account,
         * by maintaining state (which may be the previous actions
         * of several rounds and their own previous actions)
         *
         * @param time_step : current round
         * @param group_contributions_prev : previous contributions of the group
         * @return the action decided by the strategy
         */
        virtual int get_action(size_t time_step, int group_contributions_prev) = 0;
        /**
         *
         * @return a string that indicates the strategy type
         * (e.g. StrategyType::Cooperator)
         */
        virtual std::string type() = 0;
    };
}// namespace egttools::FinitePopulations::behaviors

#endif//EGTTOOLS_FINITEPOPULATIONS_BEHAVIORS_ABSTRACTCRDSTRATEGY_HPP

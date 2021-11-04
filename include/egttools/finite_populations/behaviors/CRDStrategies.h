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

#ifndef EGTTOOLS_FINITEPOPULATIONS_CRDSTRATEGIES_H
#define EGTTOOLS_FINITEPOPULATIONS_CRDSTRATEGIES_H

#include <egttools/SeedGenerator.h>
#include <egttools/Types.h>

#include <egttools/finite_populations/behaviors/AbstractCRDStrategy.hpp>
#include <iostream>
#include <random>
#include <sstream>


namespace egttools::FinitePopulations::behaviors::CRD {

    class CRDMemoryOnePlayer final : public AbstractCRDStrategy {
    public:
        /**
         * @brief This strategy contributes in function of the contributions of the rest
         * of the group in the previous round.
         *
         * This strategy contributes @param initial_action in the first round of the game,
         * afterwards compares the sum of contributions of the other members of the group
         * in the previous round (a_{-i}(t-1)) to a @param personal_threshold. If the
         * a_{-i}(t-1)) > personal_threshold the agent contributions @param action_above,
         * if a_{-i}(t-1)) = personal_threshold it contributes @param action_equal
         * or if a_{-i}(t-1)) < personal_threshold it contributes @param action_below.
         *
         * @param personal_threshold : threshold value compared to the contributions of
         *                             the other members of the group.
         * @param initial_action : contribution of the agent in the first round.
         * @param action_above : contribution if a_{-i}(t-1)) > personal_threshold
         * @param action_equal : contribution if a_{-i}(t-1)) = personal_threshold
         * @param action_below : contribution if a_{-i}(t-1)) < personal_threshold
         */
        CRDMemoryOnePlayer(int personal_threshold, int initial_action, int action_above, int action_equal, int action_below);

        int get_action(size_t time_step, int group_contributions_prev) final;
        std::string type() final;
        [[nodiscard]] std::string toString() const;

        int personal_threshold_;
        int initial_action_;
        int action_above_;
        int action_equal_;
        int action_below_;
    };


}// namespace egttools::FinitePopulations::behaviors::CRD


#endif//EGTTOOLS_FINITEPOPULATIONS_CRDSTRATEGIES_H

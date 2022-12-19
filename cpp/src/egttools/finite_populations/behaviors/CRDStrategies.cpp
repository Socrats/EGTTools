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

#include <egttools/finite_populations/behaviors/CRDStrategies.h>


egttools::FinitePopulations::behaviors::CRD::CRDMemoryOnePlayer::CRDMemoryOnePlayer(int personal_threshold,
                                                                                    int initial_action,
                                                                                    int action_above,
                                                                                    int action_equal,
                                                                                    int action_below) : personal_threshold_(personal_threshold),
                                                                                                        initial_action_(initial_action),
                                                                                                        action_above_(action_above),
                                                                                                        action_equal_(action_equal),
                                                                                                        action_below_(action_below) {
}

int egttools::FinitePopulations::behaviors::CRD::CRDMemoryOnePlayer::get_action(size_t time_step, int group_contributions_prev) {
    if (time_step == 0) return initial_action_;
    else {
        if (group_contributions_prev > personal_threshold_) return action_above_;
        else if (group_contributions_prev == personal_threshold_)
            return action_equal_;
        else
            return action_below_;
    }
}
std::string egttools::FinitePopulations::behaviors::CRD::CRDMemoryOnePlayer::type() {
    return toString();
}
std::string egttools::FinitePopulations::behaviors::CRD::CRDMemoryOnePlayer::toString() const {
    std::stringstream ss;
    ss << "CRDStrategies::CRDPlayer(" << personal_threshold_ << "," << initial_action_ << ","
            << action_above_ << "," << action_equal_ << "," << action_below_ << ")";
    return  ss.str();
}

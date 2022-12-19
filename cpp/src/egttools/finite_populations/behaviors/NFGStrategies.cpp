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

#include <egttools/finite_populations/behaviors/NFGStrategies.hpp>

int egttools::FinitePopulations::behaviors::twoActions::Cooperator::get_action(size_t time_step, int action_prev) {
    UNUSED(time_step);
    UNUSED(action_prev);
    return COOPERATE;
}
std::string egttools::FinitePopulations::behaviors::twoActions::Cooperator::type() {
    return "NFGStrategies::AllC";
}
bool egttools::FinitePopulations::behaviors::twoActions::Cooperator::is_stochastic() {
    return false;
}
int egttools::FinitePopulations::behaviors::twoActions::Defector::get_action(size_t time_step, int action_prev) {
    UNUSED(time_step);
    UNUSED(action_prev);
    return DEFECT;
}
std::string egttools::FinitePopulations::behaviors::twoActions::Defector::type() {
    return "NFGStrategies::AllD";
}
bool egttools::FinitePopulations::behaviors::twoActions::Defector::is_stochastic() {
    return false;
}
egttools::FinitePopulations::behaviors::twoActions::RandomPlayer::RandomPlayer() {
    rand_int_ = std::uniform_int_distribution<int>(0, 1);
}
int egttools::FinitePopulations::behaviors::twoActions::RandomPlayer::get_action(size_t time_step, int action_prev) {
    UNUSED(time_step);
    UNUSED(action_prev);
    return rand_int_(*egttools::Random::thread_local_generator());
}
std::string egttools::FinitePopulations::behaviors::twoActions::RandomPlayer::type() {
    return "NFGStrategies::Random";
}
bool egttools::FinitePopulations::behaviors::twoActions::RandomPlayer::is_stochastic() {
    return true;
}
int egttools::FinitePopulations::behaviors::twoActions::TitForTat::get_action(size_t time_step, int action_prev) {
    if (time_step == 0) {
        return COOPERATE;
    } else {
        return action_prev;
    }
}
std::string egttools::FinitePopulations::behaviors::twoActions::TitForTat::type() {
    return "NFGStrategies::TFT";
}
bool egttools::FinitePopulations::behaviors::twoActions::TitForTat::is_stochastic() {
    return false;
}
int egttools::FinitePopulations::behaviors::twoActions::SuspiciousTFT::get_action(size_t time_step, int action_prev) {
    if (time_step == 0) {
        return DEFECT;
    } else {
        return action_prev;
    }
}
std::string egttools::FinitePopulations::behaviors::twoActions::SuspiciousTFT::type() {
    return "NFGStrategies::SuspiciousTFT";
}
bool egttools::FinitePopulations::behaviors::twoActions::SuspiciousTFT::is_stochastic() {
    return false;
}
egttools::FinitePopulations::behaviors::twoActions::GenerousTFT::GenerousTFT(double reward, double punishment,
                                                                             double temptation, double sucker) {
    p_ = std::min(1 - ((temptation - reward) / (reward - sucker)),
                  (reward - punishment) / (temptation - punishment));
    rand_double_ = std::uniform_real_distribution<double>(0, 1);
}

int egttools::FinitePopulations::behaviors::twoActions::GenerousTFT::get_action(size_t time_step, int action_prev) {
    if ((time_step == 0) || (action_prev == COOPERATE)) {
        return COOPERATE;
    } else {
        return rand_double_(*egttools::Random::thread_local_generator()) < p_ ? COOPERATE : DEFECT;
    }
}
std::string egttools::FinitePopulations::behaviors::twoActions::GenerousTFT::type() {
    return "NFGStrategies::GenerousTFT";
}
bool egttools::FinitePopulations::behaviors::twoActions::GenerousTFT::is_stochastic() {
    return true;
}
int egttools::FinitePopulations::behaviors::twoActions::GradualTFT::get_action(size_t time_step, int action_prev) {
    // Initialize variables at round 0
    if (time_step == 0) {
        defection_string_ = 0;
        cooperation_string_ = 0;
        return COOPERATE;
    } else {
        if (cooperation_string_ > 0) {
            --cooperation_string_;
            if (action_prev == DEFECT) ++defection_string_;
            return COOPERATE;
        } else if (action_prev == DEFECT) {
            ++defection_string_;
            return DEFECT;
        } else {
            if (defection_string_ > 0) {
                if (--defection_string_ == 0) cooperation_string_ = 2;
                return DEFECT;
            } else {
                return COOPERATE;
            }
        }
    }
}
std::string egttools::FinitePopulations::behaviors::twoActions::GradualTFT::type() {
    return "NFGStrategies::GradualTFT";
}
bool egttools::FinitePopulations::behaviors::twoActions::GradualTFT::is_stochastic() {
    return false;
}
egttools::FinitePopulations::behaviors::twoActions::ImperfectTFT::ImperfectTFT(double error_probability) : error_probability_(error_probability) {
    rand_double_ = std::uniform_real_distribution<double>(0, 1);
}
int egttools::FinitePopulations::behaviors::twoActions::ImperfectTFT::get_action(size_t time_step, int action_prev) {
    if (time_step == 0) {
        return COOPERATE;
    } else {
        if (rand_double_(*egttools::Random::thread_local_generator()) < error_probability_) {
            return (action_prev + 1) % static_cast<int>(egttools::FinitePopulations::behaviors::twoActions::nb_actions);
        }
        return action_prev;
    }
}
std::string egttools::FinitePopulations::behaviors::twoActions::ImperfectTFT::type() {
    return "NFGStrategies::ImperfectTFT";
}
bool egttools::FinitePopulations::behaviors::twoActions::ImperfectTFT::is_stochastic() {
    return true;
}
int egttools::FinitePopulations::behaviors::twoActions::TFTT::get_action(size_t time_step, int action_prev) {
    int action = COOPERATE;
    if (time_step == 0) {
        action_memory_ = 1;
        return COOPERATE;
    } else if ((action_prev == DEFECT) && (action_memory_ == DEFECT)) {
        action = DEFECT;
    }
    action_memory_ = action_prev;
    return action;
}
std::string egttools::FinitePopulations::behaviors::twoActions::TFTT::type() {
    return "NFGStrategies::TFTT";
}
bool egttools::FinitePopulations::behaviors::twoActions::TFTT::is_stochastic() {
    return false;
}
int egttools::FinitePopulations::behaviors::twoActions::TTFT::get_action(size_t time_step, int action_prev) {
    if (time_step == 0) {
        defection_counter_ = 0;
        return COOPERATE;
    } else if (action_prev == DEFECT) {
        ++defection_counter_;
        return DEFECT;
    } else if (defection_counter_ > 0) {
        --defection_counter_;
        return DEFECT;
    }
    return COOPERATE;
}
std::string egttools::FinitePopulations::behaviors::twoActions::TTFT::type() {
    return "NFGStrategies::TTFT";
}
bool egttools::FinitePopulations::behaviors::twoActions::TTFT::is_stochastic() {
    return false;
}
int egttools::FinitePopulations::behaviors::twoActions::GRIM::get_action(size_t time_step, int action_prev) {
    if (time_step == 0) {
        action_ = COOPERATE;
        return COOPERATE;
    } else if (action_prev == DEFECT) {
        action_ = DEFECT;
    }
    return action_;
}
std::string egttools::FinitePopulations::behaviors::twoActions::GRIM::type() {
    return "NFGStrategies::GRIM";
}
bool egttools::FinitePopulations::behaviors::twoActions::GRIM::is_stochastic() {
    return false;
}
int egttools::FinitePopulations::behaviors::twoActions::Pavlov::get_action(size_t time_step, int action_prev) {
    if ((time_step == 0) || (action_prev == action_memory_)) {
        action_memory_ = COOPERATE;
    } else {
        action_memory_ = DEFECT;
    }
    return action_memory_;
}
std::string egttools::FinitePopulations::behaviors::twoActions::Pavlov::type() {
    return "NFGStrategies::Pavlov";
}
bool egttools::FinitePopulations::behaviors::twoActions::Pavlov::is_stochastic() {
    return false;
}
egttools::FinitePopulations::behaviors::twoActions::ActionInertia::ActionInertia(double epsilon, double p) : epsilon_(epsilon),
                                                                                                             p_(p) {
    rand_double_ = std::uniform_real_distribution<double>(0, 1);
}
int egttools::FinitePopulations::behaviors::twoActions::ActionInertia::get_action(size_t time_step, int action_prev) {
    UNUSED(action_prev);
    if (time_step == 0) {
        action_ = rand_double_(*egttools::Random::thread_local_generator()) < p_ ? COOPERATE : DEFECT;
    } else {
        if (rand_double_(*egttools::Random::thread_local_generator()) < epsilon_) {
            // Change action
            action_ = (action_ + 1) % 2;
        }
    }

    return action_;
}
std::string egttools::FinitePopulations::behaviors::twoActions::ActionInertia::type() {
    return "NFGStrategies::ActionInertia";
}
bool egttools::FinitePopulations::behaviors::twoActions::ActionInertia::is_stochastic() {
    return true;
}
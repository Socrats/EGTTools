//
// Created by Elias Fernandez on 25/02/2021.
//

#include <egttools/Utils.h>

#include <algorithm>
#include <egttools/finite_populations/behaviors/NFGStrategies.hpp>

size_t egttools::FinitePopulations::behaviors::twoActions::Cooperator::get_action(size_t time_step, size_t action_prev) {
    UNUSED(time_step);
    UNUSED(action_prev);
    return COOPERATE;
}
std::string egttools::FinitePopulations::behaviors::twoActions::Cooperator::type() {
    return "NFGStrategies::AllC";
}
size_t egttools::FinitePopulations::behaviors::twoActions::Defector::get_action(size_t time_step, size_t action_prev) {
    UNUSED(time_step);
    UNUSED(action_prev);
    return DEFECT;
}
std::string egttools::FinitePopulations::behaviors::twoActions::Defector::type() {
    return "NFGStrategies::AllD";
}
egttools::FinitePopulations::behaviors::twoActions::RandomPlayer::RandomPlayer() {
    rand_int_ = std::uniform_int_distribution<size_t>(0, 1);
}
size_t egttools::FinitePopulations::behaviors::twoActions::RandomPlayer::get_action(size_t time_step, size_t action_prev) {
    UNUSED(time_step);
    UNUSED(action_prev);
    return rand_int_(*egttools::Random::thread_local_generator());
}
std::string egttools::FinitePopulations::behaviors::twoActions::RandomPlayer::type() {
    return "NFGStrategies::Random";
}
size_t egttools::FinitePopulations::behaviors::twoActions::TitForTat::get_action(size_t time_step, size_t action_prev) {
    if (time_step == 0) {
        return COOPERATE;
    } else {
        return action_prev;
    }
}
std::string egttools::FinitePopulations::behaviors::twoActions::TitForTat::type() {
    return "NFGStrategies::TFT";
}
size_t egttools::FinitePopulations::behaviors::twoActions::SuspiciousTFT::get_action(size_t time_step, size_t action_prev) {
    if (time_step == 0) {
        return DEFECT;
    } else {
        return action_prev;
    }
}
std::string egttools::FinitePopulations::behaviors::twoActions::SuspiciousTFT::type() {
    return "NFGStrategies::SuspiciousTFT";
}

egttools::FinitePopulations::behaviors::twoActions::GenerousTFT::GenerousTFT(double reward, double punishment,
                                                                             double temptation, double sucker) {
    p_ = std::min(1 - ((temptation - reward) / (reward - sucker)),
                  (reward - punishment) / (temptation - punishment));
    rand_double_ = std::uniform_real_distribution<double>(0, 1);
}

size_t egttools::FinitePopulations::behaviors::twoActions::GenerousTFT::get_action(size_t time_step, size_t action_prev) {
    if ((time_step == 0) || (action_prev == COOPERATE)) {
        return COOPERATE;
    } else {
        return rand_double_(*egttools::Random::thread_local_generator()) < p_ ? COOPERATE : DEFECT;
    }
}
std::string egttools::FinitePopulations::behaviors::twoActions::GenerousTFT::type() {
    return "NFGStrategies::GenerousTFT";
}
size_t egttools::FinitePopulations::behaviors::twoActions::GradualTFT::get_action(size_t time_step, size_t action_prev) {
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
egttools::FinitePopulations::behaviors::twoActions::ImperfectTFT::ImperfectTFT(double error_probability) : error_probability_(error_probability) {
    rand_double_ = std::uniform_real_distribution<double>(0, 1);
}
size_t egttools::FinitePopulations::behaviors::twoActions::ImperfectTFT::get_action(size_t time_step, size_t action_prev) {
    if (time_step == 0) {
        return COOPERATE;
    } else {
        if (rand_double_(*egttools::Random::thread_local_generator()) < error_probability_) {
            return action_prev + 1 % nb_actions;
        }
        return action_prev;
    }
}
std::string egttools::FinitePopulations::behaviors::twoActions::ImperfectTFT::type() {
    return "NFGStrategies::ImperfectTFT";
}
size_t egttools::FinitePopulations::behaviors::twoActions::TFTT::get_action(size_t time_step, size_t action_prev) {
    size_t action = COOPERATE;
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
size_t egttools::FinitePopulations::behaviors::twoActions::TTFT::get_action(size_t time_step, size_t action_prev) {
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
size_t egttools::FinitePopulations::behaviors::twoActions::GRIM::get_action(size_t time_step, size_t action_prev) {
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
size_t egttools::FinitePopulations::behaviors::twoActions::Pavlov::get_action(size_t time_step, size_t action_prev) {
    if (time_step == 0) {
        action_memory_ = 1;
        return COOPERATE;
    } else if (action_prev == action_memory_) {
        action_memory_ = COOPERATE;
    } else {
        action_memory_ = DEFECT;
    }
    return action_memory_;
}
std::string egttools::FinitePopulations::behaviors::twoActions::Pavlov::type() {
    return "NFGStrategies::Pavlov";
}
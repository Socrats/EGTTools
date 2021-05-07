//
// Created by Elias Fernandez on 2019-05-15.
//

#include <Dyrwin/SED/behaviors/CrdBehaviors.hpp>

size_t EGTTools::SED::CRD::cooperator(size_t prev_donation, size_t threshold, size_t current_round) {
    UNUSED(prev_donation);
    UNUSED(threshold);
    UNUSED(current_round);
    return 2;
}

size_t EGTTools::SED::CRD::defector(size_t prev_donation, size_t threshold, size_t current_round) {
    UNUSED(prev_donation);
    UNUSED(threshold);
    UNUSED(current_round);
    return 0;
}

size_t EGTTools::SED::CRD::altruist(size_t prev_donation, size_t threshold, size_t current_round) {
    UNUSED(prev_donation);
    UNUSED(threshold);
    UNUSED(current_round);
    return 4;
}

size_t EGTTools::SED::CRD::reciprocal(size_t prev_donation, size_t threshold, size_t current_round) {
    if (current_round == 0)
        return 2;
    else if (prev_donation < threshold)
        return 0;
    else
        return 4;
}

size_t EGTTools::SED::CRD::compensator(size_t prev_donation, size_t threshold, size_t current_round) {
    if (current_round == 0)
        return 2;
    else if (prev_donation > threshold)
        return 0;
    else
        return 4;
}

size_t EGTTools::SED::CRD::compensator2(size_t prev_donation, size_t threshold, size_t current_round) {
  if (current_round == 0)
    return 2;
  else if (prev_donation < threshold)
    return 4;
  else
    return 0;
}

size_t EGTTools::SED::CRD::conditional_cooperator(size_t prev_donation, size_t threshold, size_t current_round) {
    UNUSED(current_round);
    if (prev_donation < threshold) return 0;
    else if (prev_donation == threshold) return 2;
    else return 4;
}

size_t EGTTools::SED::CRD::conditional_defector(size_t prev_donation, size_t threshold, size_t current_round) {
    UNUSED(current_round);
    if (prev_donation < threshold) return 4;
    else if (prev_donation == threshold) return 2;
    else return 0;
}

size_t EGTTools::SED::CRD::early(size_t prev_donation, size_t threshold, size_t current_round) {
    UNUSED(prev_donation);
    UNUSED(threshold);
    if (current_round < 5) return 4;
    else return 0;
}

size_t EGTTools::SED::CRD::late(size_t prev_donation, size_t threshold, size_t current_round) {
    UNUSED(prev_donation);
    UNUSED(threshold);
    if (current_round < 5) return 0;
    else return 4;
}

EGTTools::SED::CRD::CrdBehavior::CrdBehavior() {
    // Random generators
    std::mt19937_64 mt{EGTTools::Random::SeedGenerator::getInstance().getSeed()};
    std::uniform_int_distribution<size_t> dist(0, nb_strategies - 1);

    type = dist(mt);
    payoff = 0;
    switch (type) {
        case 0:
            act = &cooperator;
            break;
        case 1:
            act = &defector;
            break;
        case 2:
            act = &altruist;
            break;
        case 3:
            act = &reciprocal;
            break;
        case 4:
            act = &compensator;
            break;
        case 5:
            act = &conditional_cooperator;
            break;
        case 6:
            act = &conditional_defector;
            break;
        default:
            act = &cooperator;
            this->type = 0;
            break;
    }
}

EGTTools::SED::CRD::CrdBehavior::CrdBehavior(size_t type) {
    this->type = type;
    payoff = 0;
    switch (type) {
        case 0:
            act = &cooperator;
            break;
        case 1:
            act = &defector;
            break;
        case 2:
            act = &altruist;
            break;
        case 3:
            act = &reciprocal;
            break;
        case 4:
            act = &compensator;
            break;
        case 5:
            act = &conditional_cooperator;
            break;
        case 6:
            act = &conditional_defector;
            break;
        default:
            act = &cooperator;
            this->type = 0;
            break;
    }
}

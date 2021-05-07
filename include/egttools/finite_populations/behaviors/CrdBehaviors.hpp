//
// Created by Elias Fernandez on 2019-05-15.
//

#ifndef DYRWIN_SED_BEHAVIOR_CRDBEHAVIORS_HPP
#define DYRWIN_SED_BEHAVIOR_CRDBEHAVIORS_HPP

#include <cstdlib>
#include <random>
#include <Dyrwin/SeedGenerator.h>
#include <Dyrwin/Utils.h>

/**
 * @brief This header file contains the definition of the behaviors encountered in the
 * CRD experiments Elias & Jelena & Francisco C. Santos, et. al.
 */

namespace EGTTools::SED::CRD {
constexpr size_t nb_strategies = 5;

/**
 * @brief This player always invests 2
 *
 * @param prev_donation
 * @param threshold
 * @return action of the player
 */
size_t cooperator(size_t prev_donation, size_t threshold, size_t current_round);

/**
 * @brief This player always invests 0
 * @param prev_donation
 * @param threshold
 * @return
 */
size_t defector(size_t prev_donation, size_t threshold, size_t current_round);

/**
 * @brief This player always invests 4
 * @param prev_donation
 * @param threshold
 * @return
 */
size_t altruist(size_t prev_donation, size_t threshold, size_t current_round);

/**
 * @brief Cooperates depending on threshold
 *
 * This player starts giving 2 and then gives 4 if
 * threshold >= 10 otherwise gives 0
 *
 * @param prev_donation
 * @param threshold
 * @return
 */
size_t reciprocal(size_t prev_donation, size_t threshold, size_t current_round);

/**
 * @brief Conditional compensating behavior
 *
 * This player starts giving 2 and then gives 0 if
 * threshold >= 10 otherwise gives 4.
 *
 * @param prev_donation
 * @param threshold
 * @return
 */
size_t compensator(size_t prev_donation, size_t threshold, size_t current_round);

/**
 * @brief Conditional compensating behavior
 *
 * This player contributes 2 in the first round
 * and 4 only if a_{-i}(t-1) < threshold
 *
 * @param prev_donation
 * @param threshold
 * @param current_round
 * @return
 */
size_t compensator2(size_t prev_donation, size_t threshold, size_t current_round);

size_t conditional_cooperator(size_t prev_donation, size_t threshold, size_t current_round);

size_t conditional_defector(size_t prev_donation, size_t threshold, size_t current_round);

size_t early(size_t prev_donation, size_t threshold, size_t current_round);

size_t late(size_t prev_donation, size_t threshold, size_t current_round);

double play_game(size_t focal_player, std::vector<size_t> group_composition);

struct CrdBehavior {
  CrdBehavior();

  explicit CrdBehavior(size_t type);

  size_t type;
  double payoff;

  size_t (*act)(size_t, size_t, size_t);
};

enum class CRDBehaviors {
  cooperator = 0, defector, altruist, reciprocal, compensator
};
}

#endif //DYRWIN_CRDBEHAVIORS_HPP

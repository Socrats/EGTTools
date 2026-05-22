/** Copyright (c) 2024  Elias Fernandez
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
#pragma once
#ifndef EGTTOOLS_FINITEPOPULATIONS_UPDATERULES_TIMEDEPENDENTPC_HPP
#define EGTTOOLS_FINITEPOPULATIONS_UPDATERULES_TIMEDEPENDENTPC_HPP

#include <egttools/Types.h>
#include <egttools/finite_populations/Utils.hpp>
#include <egttools/finite_populations/update_rules/PairwiseComparison.hpp>
#include <algorithm>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace egttools::FinitePopulations::update_rules {

    /**
     * @brief Pairwise-comparison with time-dependent beta and/or mu schedules.
     *
     * Implements the "stable leaders" / time-dependent exploration scenario from
     * Pinheiro, Pacheco & Santos (2021). The selection intensity beta(t) and
     * mutation rate mu(t) follow piecewise-constant C++ schedules defined as
     * breakpoint lists, avoiding any Python callback overhead per time-step.
     *
     * Schedule format: vector of (t_start, value) pairs sorted by t_start.
     * The value is held constant from t_start until the next breakpoint.
     *
     * Example — beta that doubles at generation 500:
     *   TimeDependentPC pc({{0, 1.0}, {500, 2.0}}, {});
     *
     * When the schedule is empty, the default beta/mu passed to step() is used.
     * mu schedule is optional; pass {} to keep mu constant.
     *
     * The gradient is numerically exact for the beta value active at the
     * requested time t.
     */
    class TimeDependentPC {
    public:
        using Schedule = std::vector<std::pair<int64_t, double>>;

        explicit TimeDependentPC(Schedule beta_schedule = {},
                                 Schedule mu_schedule = {})
            : beta_schedule_(std::move(beta_schedule)),
              mu_schedule_(std::move(mu_schedule)) {
            auto check = [](const Schedule &s) {
                for (size_t i = 1; i < s.size(); ++i) {
                    if (s[i].first <= s[i - 1].first)
                        throw std::invalid_argument("TimeDependentPC: schedule breakpoints must be strictly increasing");
                }
            };
            check(beta_schedule_);
            check(mu_schedule_);
        }

        [[nodiscard]] static std::string name() { return "TimeDependentPC"; }

        [[nodiscard]] double get_beta(int64_t t, double default_beta) const {
            return lookup(beta_schedule_, t, default_beta);
        }

        [[nodiscard]] double get_mu(int64_t t, double default_mu) const {
            return lookup(mu_schedule_, t, default_mu);
        }

        template<class GameType, class CacheType>
        void step(std::vector<int> &population,
                  VectorXui &mean_state,
                  const AdjacencyList &network,
                  GameType &game,
                  CacheType &cache,
                  VectorXui &nbuf,
                  int nb_strategies,
                  double default_beta,
                  double default_mu,
                  std::mt19937_64 &gen,
                  int64_t t = 0) const {
            double beta = get_beta(t, default_beta);
            double mu = get_mu(t, default_mu);
            PairwiseComparison::step(population, mean_state, network, game, cache,
                                     nbuf, nb_strategies, beta, mu, gen, t);
        }

        template<class GameType, class CacheType>
        Vector compute_exact_gradient(const std::vector<int> &population,
                                      const AdjacencyList &network,
                                      GameType &game,
                                      CacheType &cache,
                                      VectorXui &nbuf,
                                      int nb_strategies,
                                      double beta,
                                      int64_t t = 0) const {
            beta = get_beta(t, beta);
            return PairwiseComparison::compute_exact_gradient(population, network, game,
                                                              cache, nbuf, nb_strategies, beta);
        }

    private:
        Schedule beta_schedule_;
        Schedule mu_schedule_;

        static double lookup(const Schedule &schedule, int64_t t, double default_val) {
            if (schedule.empty()) return default_val;
            // Find last breakpoint <= t (schedule is sorted ascending)
            double val = schedule.front().second;
            for (const auto &[t_start, v] : schedule) {
                if (t_start > t) break;
                val = v;
            }
            return val;
        }
    };

}// namespace egttools::FinitePopulations::update_rules

#endif//EGTTOOLS_FINITEPOPULATIONS_UPDATERULES_TIMEDEPENDENTPC_HPP

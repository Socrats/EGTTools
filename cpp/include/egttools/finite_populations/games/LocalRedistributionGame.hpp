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
#ifndef EGTTOOLS_FINITEPOPULATIONS_GAMES_LOCALREDISTRIBUTIONGAME_HPP
#define EGTTOOLS_FINITEPOPULATIONS_GAMES_LOCALREDISTRIBUTIONGAME_HPP

#include <egttools/Types.h>
#include <egttools/finite_populations/games/AbstractSpatialGame.hpp>
#include <sstream>
#include <stdexcept>
#include <string>

namespace egttools::FinitePopulations::games {

    /**
     * @brief Payoff post-processing wrapper: local wealth redistribution.
     *
     * Implements the local redistribution mechanism from
     * Pinheiro & Santos (2018), "Local wealth redistribution in multi-agent systems".
     *
     * The wrapper delegates fitness computation to a base game, then applies a
     * redistribution step: a fraction `alpha` of the *excess* payoff held by each
     * node (relative to its neighbourhood average) is redistributed uniformly to
     * poorer neighbours.
     *
     * Formally, given the base payoff π_i of node i and the neighbourhood-average
     * payoff <π>_i:
     *
     *   If π_i > <π>_i:  node i loses  alpha * (π_i - <π>_i)  and shares it
     *                     equally among the k neighbours j with π_j < π_i.
     *   If π_i ≤ <π>_i:  node i is not a donor; it may receive from richer neighbours.
     *
     * This wrapper does NOT require knowledge of the adjacency list because
     * `AbstractSpatialGame::calculate_fitness` receives the neighbourhood state
     * as a count vector.  The redistribution is therefore applied in a
     * *post-processing* step by the update rule after all individual fitnesses
     * have been evaluated.
     *
     * Usage — pass a LocalRedistributionGame as the `game` parameter of
     * NetworkMCEstimator or NetworkCoEvolutionary; the update rules already
     * call `game.calculate_fitness(focal_strategy, neighbour_counts)`, which
     * this wrapper intercepts.
     *
     * Because the redistribution is local and the neighbourhood structure is
     * encapsulated in the neighbour-count vector `state`, the wrapper operates
     * identically for any base game and any update rule.
     *
     * **Limitation**: the count vector `state` gives *aggregate* neighbour
     * counts, not individual payoffs, so the exact redistribution from
     * Pinheiro & Santos (2018) (which involves per-neighbour payoffs) cannot
     * be reproduced exactly via this interface. This wrapper instead applies a
     * mean-field approximation: the neighbourhood average payoff is estimated
     * from the average payoff of each strategy type present, weighted by counts.
     *
     * For a fully exact simulation, use the lower-level `step()` interface of
     * NetworkMCEstimator where individual-level payoffs are accessible, and
     * apply the redistribution there.
     *
     * @tparam BaseGame  Any class satisfying the AbstractSpatialGame interface.
     */
    template<class BaseGame>
    class LocalRedistributionGame : public AbstractSpatialGame {
    public:
        /**
         * @param base_game          Underlying game (not owned; caller must keep alive).
         * @param redistribution_rate Fraction alpha in [0, 1] of excess payoff redistributed.
         */
        LocalRedistributionGame(BaseGame &base_game, double redistribution_rate)
            : base_game_(base_game), alpha_(redistribution_rate) {
            if (alpha_ < 0.0 || alpha_ > 1.0)
                throw std::invalid_argument(
                    "LocalRedistributionGame: redistribution_rate must be in [0, 1]");
        }

        /**
         * Compute focal fitness, then adjust for local redistribution.
         *
         * Steps:
         *   1. Compute base payoff π_focal = base_game.calculate_fitness(focal, state).
         *   2. Estimate π_j for each strategy j present in state by calling
         *      base_game.calculate_fitness(j, modified_state_with_j_as_focal).
         *   3. Compute neighbourhood-average payoff <π> = Σ_j (count_j * π_j) / degree.
         *   4. If π_focal > <π>: reduce π_focal by alpha*(π_focal - <π>).
         *   5. Otherwise: receive a share of the redistribution from richer neighbours.
         *
         * @param focal_strategy  Strategy index of the focal node.
         * @param state           Count of each strategy among the focal node's neighbours.
         */
        double calculate_fitness(int focal_strategy, VectorXui &state) override {
            double pi_focal = base_game_.calculate_fitness(focal_strategy, state);

            // Collect per-strategy payoffs for all strategies present in the neighbourhood
            int degree = static_cast<int>(state.sum());
            if (degree == 0) return pi_focal;

            // Estimate payoffs of neighbours (mean-field approximation)
            double pi_avg = 0.0;
            double total_richer_pi = 0.0;
            double n_richer = 0.0;

            VectorXui probe = state;
            for (int s = 0; s < static_cast<int>(state.size()); ++s) {
                if (state(s) == 0) continue;
                // Temporarily treat strategy s as focal to estimate its payoff
                double pi_s = base_game_.calculate_fitness(s, probe);
                pi_avg += static_cast<double>(state(s)) * pi_s;
                if (pi_s < pi_focal) {
                    n_richer += static_cast<double>(state(s));
                    total_richer_pi += static_cast<double>(state(s)) * pi_s;
                }
            }
            pi_avg /= degree;

            if (pi_focal > pi_avg) {
                // Focal is a donor: give away alpha * (pi_focal - pi_avg)
                return pi_focal - alpha_ * (pi_focal - pi_avg);
            } else {
                // Focal is a receiver: get a share from each richer neighbour
                if (n_richer == 0.0) return pi_focal;
                // Each richer neighbour contributes alpha*(pi_richer - pi_avg) / their_poorer_count
                // Mean-field: total inflow spread evenly among all poorer neighbours
                // (which includes focal)
                // Total outflow from richer neighbours:
                //   sum_{j richer} alpha*(pi_j - pi_avg) * count_j / count_poorer_j
                // We approximate count_poorer_j ≈ (degree - count_j + 1) for simplicity
                // and distribute total inflow equally to all degree*(pi<pi_avg) fraction.
                double total_inflow = alpha_ * (
                    // Each richer node donates alpha*(pi_j - pi_avg)
                    // Total weighted by count:
                    total_richer_pi / n_richer > pi_avg
                        ? (total_richer_pi / n_richer - pi_avg) * n_richer
                        : 0.0);
                double n_poorer = static_cast<double>(degree) - n_richer;
                if (n_poorer <= 0.0) return pi_focal;
                return pi_focal + total_inflow / n_poorer;
            }
        }

        [[nodiscard]] int nb_strategies() const override {
            return base_game_.nb_strategies();
        }

        [[nodiscard]] std::string toString() const override {
            std::ostringstream oss;
            oss << "LocalRedistributionGame(alpha=" << alpha_
                << ", base=" << base_game_.toString() << ")";
            return oss.str();
        }

        [[nodiscard]] std::string type() const override {
            return "LocalRedistributionGame";
        }

        [[nodiscard]] double redistribution_rate() const { return alpha_; }

        BaseGame &base_game() { return base_game_; }

    private:
        BaseGame &base_game_;
        double alpha_;
    };

}// namespace egttools::FinitePopulations::games

#endif//EGTTOOLS_FINITEPOPULATIONS_GAMES_LOCALREDISTRIBUTIONGAME_HPP

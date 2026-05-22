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
#ifndef EGTTOOLS_FINITEPOPULATIONS_UPDATERULES_ABSTRACTUPDATERULE_HPP
#define EGTTOOLS_FINITEPOPULATIONS_UPDATERULES_ABSTRACTUPDATERULE_HPP

#include <egttools/Types.h>
#include <egttools/finite_populations/games/AbstractSpatialGame.hpp>
#include <egttools/finite_populations/structure/AbstractNetworkStructure.hpp>
#include <random>
#include <string>
#include <vector>

namespace egttools::FinitePopulations::update_rules {

    using AdjacencyList = egttools::FinitePopulations::structure::AdjacencyList;
    using AbstractSpatialGame = egttools::FinitePopulations::games::AbstractSpatialGame;

    /**
     * @brief Interface for evolutionary update rules on networks.
     *
     * An update rule encapsulates a single time-step of the evolutionary process:
     * how strategies are copied, born, or die, given the current population state
     * and network topology.
     *
     * Three biologically motivated rules are provided:
     *
     *   - PairwiseComparison (PC): focal node is selected uniformly; it compares
     *     fitness with a random neighbour and copies the neighbour's strategy with
     *     the Fermi probability p = 1/(1+exp(beta*(f_focal - f_neighbour))).
     *
     *   - BirthDeath (BD): a node is selected to reproduce proportional to fitness;
     *     a random neighbour of that node is replaced by the offspring.
     *
     *   - DeathBirth (DB): a node is selected to die uniformly; a neighbour is
     *     selected to reproduce proportional to fitness and replaces the dead node.
     *
     * BD and DB are known to produce qualitatively different selection dynamics on
     * heterogeneous networks compared to PC.
     *
     * All concrete update rules are **policy structs** (stateless or near-stateless),
     * intended to be used as template parameters for NetworkMCEstimator and
     * NetworkCoEvolutionary. This avoids virtual dispatch overhead in the inner loop.
     *
     * Concrete rules must expose:
     *
     *   // Execute one time-step of the update rule.
     *   template<class GameType, class CacheType>
     *   static void step(
     *       std::vector<int>& population,
     *       VectorXui& mean_state,
     *       const AdjacencyList& network,
     *       GameType& game,
     *       CacheType& cache,
     *       VectorXui& neighbourhood_state_buf,  // scratch buffer, size nb_strategies
     *       int nb_strategies,
     *       double beta, double mu,
     *       std::mt19937_64& gen,
     *       int64_t t = 0);
     *
     *   // Numerically exact gradient of selection for the current population state.
     *   // See each concrete class for the rule-specific formula.
     *   template<class GameType, class CacheType>
     *   static Vector compute_exact_gradient(
     *       const std::vector<int>& population,
     *       const AdjacencyList& network,
     *       GameType& game,
     *       CacheType& cache,
     *       VectorXui& neighbourhood_state_buf,
     *       int nb_strategies,
     *       double beta);
     *
     *   [[nodiscard]] static std::string name();
     */
    struct AbstractUpdateRuleConcept {
        // Documents the required interface — not a runtime base class.
        // Update rules are used as template (policy) parameters, not polymorphically.
    };

}// namespace egttools::FinitePopulations::update_rules

#endif//EGTTOOLS_FINITEPOPULATIONS_UPDATERULES_ABSTRACTUPDATERULE_HPP

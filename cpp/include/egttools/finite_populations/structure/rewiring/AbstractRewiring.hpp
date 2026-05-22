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
#ifndef EGTTOOLS_FINITEPOPULATIONS_STRUCTURE_REWIRING_ABSTRACTREWIRING_HPP
#define EGTTOOLS_FINITEPOPULATIONS_STRUCTURE_REWIRING_ABSTRACTREWIRING_HPP

/**
 * @file AbstractRewiring.hpp
 * @brief Documents the rewiring policy concept.
 *
 * Rewiring rules are stateless or stateful policy structs/classes that implement
 * the following interface.  They are used as a template parameter in
 * NetworkCoEvolutionary to select a rewiring strategy at compile time with
 * zero runtime overhead.
 *
 * Required interface:
 *
 *   // Rewire one or more edges incident to focal_node.
 *   // network is modified in-place; population is read-only.
 *   static void rewire(int focal_node,
 *                      AdjacencyList &network,
 *                      const std::vector<int> &population,
 *                      std::mt19937_64 &gen);
 *
 *   // Human-readable name of the rule.
 *   static std::string name();
 *
 * Concrete implementations: RandomRewiring, HomophilicRewiring.
 */

#include <egttools/finite_populations/structure/AbstractNetworkStructure.hpp>
#include <random>
#include <string>
#include <vector>

namespace egttools::FinitePopulations::structure::rewiring {

    using AdjacencyList = egttools::FinitePopulations::structure::AdjacencyList;

}// namespace egttools::FinitePopulations::structure::rewiring

#endif//EGTTOOLS_FINITEPOPULATIONS_STRUCTURE_REWIRING_ABSTRACTREWIRING_HPP

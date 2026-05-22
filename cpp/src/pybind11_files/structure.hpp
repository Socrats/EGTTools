//
// Created by Elias Fernandez on 08/01/2023.
//
#pragma once
#ifndef EGTTOOLS_PYBIND11FILES_STRUCTURE_HPP
#define EGTTOOLS_PYBIND11FILES_STRUCTURE_HPP

#include <egttools/finite_populations/NetworkCoEvolutionary.hpp>
#include <egttools/finite_populations/NetworkMCEstimator.hpp>
#include <egttools/finite_populations/games/AbstractSpatialGame.hpp>
#include <egttools/finite_populations/structure/AbstractNetworkStructure.hpp>
#include <egttools/finite_populations/structure/AbstractStructure.hpp>
#include <egttools/finite_populations/structure/Network.hpp>
#include <egttools/finite_populations/structure/NetworkGroup.hpp>
#include <egttools/finite_populations/structure/NetworkGroupSync.hpp>
#include <egttools/finite_populations/structure/NetworkSync.hpp>
#include <egttools/finite_populations/structure/rewiring/HomophilicRewiring.hpp>
#include <egttools/finite_populations/structure/rewiring/RandomRewiring.hpp>
#include <egttools/finite_populations/update_rules/BirthDeath.hpp>
#include <egttools/finite_populations/update_rules/DeathBirth.hpp>
#include <egttools/finite_populations/update_rules/PairwiseComparison.hpp>
#include <egttools/finite_populations/update_rules/TimeDependentPC.hpp>
#include <memory>

#include "egttools_common.hpp"
#include "python_stubs.hpp"

using NetworkStructure = egttools::FinitePopulations::structure::Network<egttools::FinitePopulations::games::AbstractSpatialGame>;
using NetworkGroupStructure = egttools::FinitePopulations::structure::NetworkGroup<egttools::FinitePopulations::games::AbstractSpatialGame>;
using NetworkStructureSync = egttools::FinitePopulations::structure::NetworkSync<egttools::FinitePopulations::games::AbstractSpatialGame>;
using NetworkGroupStructureSync = egttools::FinitePopulations::structure::NetworkGroupSync<egttools::FinitePopulations::games::AbstractSpatialGame>;

using NetworkMCEstimatorPC = egttools::FinitePopulations::NetworkMCEstimator<
        egttools::FinitePopulations::update_rules::PairwiseComparison>;
using NetworkMCEstimatorBD = egttools::FinitePopulations::NetworkMCEstimator<
        egttools::FinitePopulations::update_rules::BirthDeath>;
using NetworkMCEstimatorDB = egttools::FinitePopulations::NetworkMCEstimator<
        egttools::FinitePopulations::update_rules::DeathBirth>;
using NetworkMCEstimatorTDPC = egttools::FinitePopulations::NetworkMCEstimator<
        egttools::FinitePopulations::update_rules::TimeDependentPC>;

using NetworkCoEvoPCRandom = egttools::FinitePopulations::NetworkCoEvolutionary<
        egttools::FinitePopulations::update_rules::PairwiseComparison,
        egttools::FinitePopulations::structure::rewiring::RandomRewiring>;
using NetworkCoEvoPCHomophilic = egttools::FinitePopulations::NetworkCoEvolutionary<
        egttools::FinitePopulations::update_rules::PairwiseComparison,
        egttools::FinitePopulations::structure::rewiring::HomophilicRewiring>;

#endif//EGTTOOLS_PYBIND11FILES_STRUCTURE_HPP

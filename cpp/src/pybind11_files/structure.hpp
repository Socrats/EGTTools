//
// Created by Elias Fernandez on 08/01/2023.
//
#pragma once
#ifndef EGTTOOLS_PYBIND11FILES_STRUCTURE_HPP
#define EGTTOOLS_PYBIND11FILES_STRUCTURE_HPP

#include <egttools/finite_populations/games/AbstractSpatialGame.hpp>
#include <egttools/finite_populations/structure/AbstractNetworkStructure.hpp>
#include <egttools/finite_populations/structure/AbstractStructure.hpp>
#include <egttools/finite_populations/structure/Network.hpp>
#include <egttools/finite_populations/structure/NetworkGroup.hpp>
#include <egttools/finite_populations/structure/NetworkGroupSync.hpp>
#include <egttools/finite_populations/structure/NetworkSync.hpp>
#include <memory>

#include "egttools_common.hpp"
#include "python_stubs.hpp"

using NetworkStructure = egttools::FinitePopulations::structure::Network<egttools::FinitePopulations::games::AbstractSpatialGame>;
using NetworkGroupStructure = egttools::FinitePopulations::structure::NetworkGroup<egttools::FinitePopulations::games::AbstractSpatialGame>;
using NetworkStructureSync = egttools::FinitePopulations::structure::NetworkSync<egttools::FinitePopulations::games::AbstractSpatialGame>;
using NetworkGroupStructureSync = egttools::FinitePopulations::structure::NetworkGroupSync<egttools::FinitePopulations::games::AbstractSpatialGame>;

#endif//EGTTOOLS_PYBIND11FILES_STRUCTURE_HPP

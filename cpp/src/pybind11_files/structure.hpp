//
// Created by Elias Fernandez on 08/01/2023.
//
#pragma once
#ifndef EGTTOOLS_PYBIND11FILES_STRUCTURE_HPP
#define EGTTOOLS_PYBIND11FILES_STRUCTURE_HPP

#include <egttools/finite_populations/games/AbstractSpatialGame.hpp>
#include <egttools/finite_populations/structure/AbstractStructure.hpp>
#include <egttools/finite_populations/structure/Network.hpp>
#include <memory>

#include "egttools_common.hpp"
#include "python_stubs.hpp"

using NetworkStructure = egttools::FinitePopulations::structure::Network<egttools::FinitePopulations::games::AbstractSpatialGame>;

#endif//EGTTOOLS_PYBIND11FILES_STRUCTURE_HPP

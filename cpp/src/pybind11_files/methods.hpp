//
// Created by Elias Fernandez on 14/11/2022.
//
#pragma once
#ifndef EGTTOOLS_PYBIND11FILES_METHODS_HPP
#define EGTTOOLS_PYBIND11FILES_METHODS_HPP

#include <egttools/SeedGenerator.h>
#include <egttools/utils/CalculateExpectedIndicators.h>

#include <egttools/finite_populations/PairwiseComparisonNumerical.hpp>
#include <egttools/finite_populations/analytical/PairwiseComparison.hpp>
#include <egttools/finite_populations/evolvers/GeneralPopulationEvolver.hpp>
#include <egttools/finite_populations/evolvers/NetworkEvolver.hpp>
#include <egttools/finite_populations/structure/AbstractNetworkStructure.hpp>
#include <egttools/finite_populations/structure/AbstractStructure.hpp>
#include <egttools/infinite_populations/ReplicatorDynamics.hpp>
#include <egttools/utils/ThreadSafeLRUCache.hpp>
#include <memory>
#include <optional>
#include <string>

#include "egttools_common.hpp"

#if (HAS_BOOST)
#include <boost/multiprecision/cpp_int.hpp>

#include "boost_cpp_int_cast_to_pybind11.hpp"
#endif

#endif//EGTTOOLS_PYBIND11FILES_METHODS_HPP

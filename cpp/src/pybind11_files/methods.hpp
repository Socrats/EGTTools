//
// Created by Elias Fernandez on 14/11/2022.
//
#pragma once
#ifndef EGTTOOLS_METHODS_HPP
#define EGTTOOLS_METHODS_HPP

#include <egttools/SeedGenerator.h>
#include <egttools/utils/CalculateExpectedIndicators.h>

#include <egttools/LruCache.hpp>
#include <egttools/finite_populations/PairwiseMoran.hpp>
#include <egttools/finite_populations/analytical/PairwiseComparison.hpp>

#include "egttools_common.hpp"

#if (HAS_BOOST)
#include <boost/multiprecision/cpp_int.hpp>
namespace mp = boost::multiprecision;
#endif

#endif//EGTTOOLS_METHODS_HPP

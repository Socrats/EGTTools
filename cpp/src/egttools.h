//
// Created by Elias Fernandez on 05/05/2021.
//
#pragma once
#ifndef EGTTOOLS_EGTTOOLS_H
#define EGTTOOLS_EGTTOOLS_H

#include <egttools/Distributions.h>
#include <egttools/SeedGenerator.h>
#include <egttools/Types.h>
#include <egttools/finite_populations/behaviors/CRDStrategies.h>
#include <egttools/finite_populations/games/NormalFormGame.h>
#include <egttools/utils/CalculateExpectedIndicators.h>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <egttools/Data.hpp>
#include <egttools/LruCache.hpp>
//#include <egttools/finite_populations/ImitationMultipleGames.hpp>
#include <egttools/finite_populations/PairwiseMoran.hpp>
#include <egttools/finite_populations/analytical/PairwiseComparison.hpp>
#include <egttools/finite_populations/behaviors/AbstractCRDStrategy.hpp>
#include <egttools/finite_populations/behaviors/AbstractNFGStrategy.hpp>
#include <egttools/finite_populations/behaviors/NFGStrategies.hpp>
#include <egttools/finite_populations/games/AbstractGame.hpp>
#include <egttools/finite_populations/games/AbstractNPlayerGame.hpp>
#include <egttools/finite_populations/games/CRDGame.hpp>
#include <egttools/finite_populations/games/CRDGameTU.hpp>
#include <egttools/finite_populations/games/Matrix2PlayerGameHolder.hpp>
#include <egttools/finite_populations/games/MatrixNPlayerGameHolder.hpp>
#include <egttools/finite_populations/games/NPlayerStagHunt.hpp>
#include <egttools/finite_populations/games/OneShotCRD.hpp>
#include <egttools/utils/TimingUncertainty.hpp>
#include <memory>
#include <stdexcept>
#include <utility>

#include "python_stubs.hpp"
#include "version.h"

#endif//EGTTOOLS_EGTTOOLS_H

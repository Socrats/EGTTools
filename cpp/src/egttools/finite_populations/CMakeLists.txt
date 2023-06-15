# Copyright (c) 2019-2020  Elias Fernandez
#
# This file is part of EGTtools.
#
# EGTtools is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# EGTtools is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with EGTtools.  If not, see <http://www.gnu.org/licenses/>

target_sources(numerical_ PRIVATE
        Utils.cpp
        games/AbstractNPlayerGame.cpp
        games/NormalFormGame.cpp
        games/CRDGame.cpp
        games/CRDGameTU.cpp
        games/OneShotCRD.cpp
        games/NPlayerStagHunt.cpp
        games/Matrix2PlayerGameHolder.cpp
        games/MatrixNPlayerGameHolder.cpp
        games/NormalFormNetworkGame.cpp
        games/OneShotCRDNetworkGame.cpp
        behaviors/NFGStrategies.cpp
        behaviors/CRDStrategies.cpp
        analytical/PairwiseComparison.cpp
        evolvers/GeneralPopulationEvolver.cpp
        evolvers/NetworkEvolver.cpp
        )
# Copyright (c) 2019-2021  Elias Fernandez
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

add_custom_target(pytest
        COMMAND ${CMAKE_COMMAND} -E env PYTHONPATH=$<TARGET_FILE_DIR:numerical_> ${Python_EXECUTABLE} -m pytest
        DEPENDS numerical
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
add_custom_target(tests DEPENDS pytest)

add_library(egttoolsNumerical
        ../cpp/src/egttools/SeedGenerator.cpp
        ../cpp/src/egttools/Distributions.cpp
        ../cpp/src/egttools/Data.cpp
        ../cpp/src/egttools/utils/CalculateExpectedIndicators.cpp
        ../cpp/src/egttools/finite_populations/Utils.cpp
        ../cpp/src/egttools/finite_populations/analytical/PairwiseComparison.cpp
        ../cpp/src/egttools/finite_populations/games/Matrix2PlayerGameHolder.cpp
        ../cpp/src/egttools/finite_populations/games/MatrixNPlayerGameHolder.cpp
        ../cpp/src/egttools/finite_populations/games/NormalFormGame.cpp
        ../cpp/src/egttools/finite_populations/games/CRDGame.cpp
        ../cpp/src/egttools/finite_populations/games/CRDGameTU.cpp
        ../cpp/src/egttools/finite_populations/games/OneShotCRD.cpp
        ../cpp/src/egttools/finite_populations/games/NormalFormNetworkGame.cpp
        ../cpp/src/egttools/finite_populations/games/OneShotCRDNetworkGame.cpp
        ../cpp/src/egttools/finite_populations/behaviors/NFGStrategies.cpp
        ../cpp/src/egttools/finite_populations/behaviors/CRDStrategies.cpp
        ../cpp/src/egttools/finite_populations/evolvers/GeneralPopulationEvolver.cpp
        ../cpp/src/egttools/finite_populations/evolvers/NetworkEvolver.cpp
        )

set_target_properties(egttoolsNumerical PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED TRUE
        CXX_EXTENSIONS FALSE
        #        INTERPROCEDURAL_OPTIMIZATION ${LTO_SUPPORTED}
        #        POSITION_INDEPENDENT_CODE TRUE
        #         Fixing "ld: warning: direct access in function '...' from file '...' to global weak symbol '...' from file '...' means the weak symbol cannot
        #         be overridden at runtime. This was likely caused by different translation units being compiled with different visibility settings."
        #        C_VISIBILITY_PRESET hidden
        #        CXX_VISIBILITY_PRESET hidden
        )

set(Boost_DEBUG ON)
find_package(Boost REQUIRED COMPONENTS multiprecision)

if (Boost_FOUND)
    get_target_property(Boost_INCLUDE_DIRS Boost::multiprecision INTERFACE_INCLUDE_DIRECTORIES)
    message(STATUS "Boost Include Directories: ${Boost_INCLUDE_DIRS}")
    target_include_directories(egttoolsNumerical PRIVATE ${Boost_INCLUDE_DIRS})
    target_link_libraries(egttoolsNumerical PRIVATE Boost::multiprecision)
    add_definitions("-DHAS_BOOST")
else()
    message(FATAL_ERROR "Boost not found")
endif()

add_executable(testPMrun cpp/test_PairwiseMoran_run.cpp)
#set_target_properties(testPM PROPERTIES INTERPROCEDURAL_OPTIMIZATION ${LTO_SUPPORTED})
if (Boost_FOUND)
    target_include_directories(testPMrun PRIVATE ${Boost_INCLUDE_DIRS})
    target_link_libraries(testPMrun egttoolsNumerical Boost::multiprecision)
else()
    target_link_libraries(testPMrun egttoolsNumerical)
endif ()


add_executable(testPMBstationary_distribution_sparse cpp/test_PairwiseMoran_stationary_distribution_sparse.cpp)
#set_target_properties(testPM PROPERTIES INTERPROCEDURAL_OPTIMIZATION ${LTO_SUPPORTED})
if (Boost_FOUND)
    target_include_directories(testPMBstationary_distribution_sparse PRIVATE ${Boost_INCLUDE_DIRS})
    target_link_libraries(testPMBstationary_distribution_sparse egttoolsNumerical Boost::multiprecision)
else()
    target_link_libraries(testPMBstationary_distribution_sparse egttoolsNumerical)
endif ()
#if (OPENMP_FOUND)
#    target_link_libraries(testPMBstationary_distribution_sparse OpenMP::OpenMP_CXX)
#endif ()

add_executable(testPMBstationary_distribution_dense cpp/test_PairwiseMoran_stationary_distribution_dense.cpp)
#set_target_properties(testPM PROPERTIES INTERPROCEDURAL_OPTIMIZATION ${LTO_SUPPORTED})
if (Boost_FOUND)
    target_include_directories(testPMBstationary_distribution_dense PRIVATE ${Boost_INCLUDE_DIRS})
    target_link_libraries(testPMBstationary_distribution_dense egttoolsNumerical Boost::multiprecision)
else()
    target_link_libraries(testPMBstationary_distribution_dense egttoolsNumerical)
endif ()


add_executable(testEigenSparseMatrix cpp/test_eigen_sparse_matrix.cpp)

add_executable(testOrderedSamplingWithoutReplacement cpp/test_ordered_sampling_without_replacement.cpp)
if (Boost_FOUND)
    target_include_directories(testOrderedSamplingWithoutReplacement PRIVATE ${Boost_INCLUDE_DIRS})
    target_link_libraries(testOrderedSamplingWithoutReplacement egttoolsNumerical Boost::multiprecision)
else()
    target_link_libraries(testOrderedSamplingWithoutReplacement egttoolsNumerical)
endif ()

add_executable(testCRDGame cpp/test_crd_game.cpp)
if (Boost_FOUND)
    target_include_directories(testCRDGame PRIVATE ${Boost_INCLUDE_DIRS})
    target_link_libraries(testCRDGame egttoolsNumerical Boost::multiprecision)
else()
    target_link_libraries(testCRDGame egttoolsNumerical)
endif ()


add_executable(testCRDTUGame cpp/test_crd_tu_game.cpp)
if (Boost_FOUND)
    target_include_directories(testCRDTUGame PRIVATE ${Boost_INCLUDE_DIRS})
    target_link_libraries(testCRDTUGame egttoolsNumerical Boost::multiprecision)
else()
    target_link_libraries(testCRDTUGame egttoolsNumerical)
endif ()

add_executable(testOneShotCRDGame cpp/test_oneshotcrd_game.cpp)
if (Boost_FOUND)
    target_include_directories(testOneShotCRDGame PRIVATE ${Boost_INCLUDE_DIRS})
    target_link_libraries(testOneShotCRDGame egttoolsNumerical Boost::multiprecision)
else()
    target_link_libraries(testOneShotCRDGame egttoolsNumerical)
endif ()


add_executable(testTimingUncertainty cpp/test_timing_uncertainty.cpp)
target_link_libraries(testTimingUncertainty egttoolsNumerical)

if (Boost_FOUND)
    add_executable(testBinomial cpp/test_binomial_coefficient.cpp)
    target_include_directories(testBinomial PRIVATE ${Boost_INCLUDE_DIRS})
    target_link_libraries(testBinomial egttoolsNumerical Boost::multiprecision)
endif ()

if (Boost_FOUND)
    add_executable(testBinomialTime cpp/test_binomial_coefficient_with_time.cpp)
    target_include_directories(testBinomialTime PRIVATE ${Boost_INCLUDE_DIRS})
    target_link_libraries(testBinomialTime egttoolsNumerical Boost::multiprecision)
endif ()

add_executable(testGameHolders cpp/test_game_holders.cpp)
target_link_libraries(testGameHolders egttoolsNumerical)
if (Boost_FOUND)
    target_include_directories(testGameHolders PRIVATE ${Boost_INCLUDE_DIRS})
    target_link_libraries(testGameHolders egttoolsNumerical Boost::multiprecision)
else()
    target_link_libraries(testGameHolders egttoolsNumerical)
endif ()

add_executable(testGeneralEvolverNetwork cpp/test_general_evolver_network.cpp)
if (Boost_FOUND)
    target_include_directories(testGeneralEvolverNetwork PRIVATE ${Boost_INCLUDE_DIRS})
    target_link_libraries(testGeneralEvolverNetwork egttoolsNumerical Boost::multiprecision)
else()
    target_link_libraries(testGeneralEvolverNetwork egttoolsNumerical)
endif ()

add_executable(testNetworkSync cpp/test_network_sync.cpp)
if (Boost_FOUND)
    target_include_directories(testNetworkSync PRIVATE ${Boost_INCLUDE_DIRS})
    target_link_libraries(testNetworkSync egttoolsNumerical Boost::multiprecision)
else()
    target_link_libraries(testNetworkSync egttoolsNumerical)
endif ()

add_executable(testOneShotNetworkGame cpp/test_one_shot_network_game.cpp)
if (Boost_FOUND)
    target_include_directories(testOneShotNetworkGame PRIVATE ${Boost_INCLUDE_DIRS})
    target_link_libraries(testOneShotNetworkGame egttoolsNumerical Boost::multiprecision)
else()
    target_link_libraries(testOneShotNetworkGame egttoolsNumerical)
endif ()

add_executable(testNetworkEvolver cpp/test_network_evolver.cpp)
if (Boost_FOUND)
    target_include_directories(testNetworkEvolver PRIVATE ${Boost_INCLUDE_DIRS})
    target_link_libraries(testNetworkEvolver egttoolsNumerical Boost::multiprecision)
else()
    target_link_libraries(testNetworkEvolver egttoolsNumerical)
endif ()

add_executable(testFixations cpp/test_calculate_fixation_probabilities.cpp)
if (Boost_FOUND)
    target_include_directories(testFixations PRIVATE ${Boost_INCLUDE_DIRS})
    target_link_libraries(testFixations egttoolsNumerical Boost::multiprecision)
else()
    target_link_libraries(testFixations egttoolsNumerical)
endif ()

add_executable(testTSLRUCache cpp/test_thread_safe_lru_cache.cpp)
if (Boost_FOUND)
    target_include_directories(testTSLRUCache PRIVATE ${Boost_INCLUDE_DIRS})
    target_link_libraries(testTSLRUCache egttoolsNumerical Boost::multiprecision)
else()
    target_link_libraries(testTSLRUCache egttoolsNumerical)
endif ()

add_executable(testBLASLAPACK cpp/test_blas_lapack.cpp)
if (Boost_FOUND)
    target_include_directories(testBLASLAPACK PRIVATE ${Boost_INCLUDE_DIRS})
    target_link_libraries(testBLASLAPACK egttoolsNumerical Boost::multiprecision "-framework Accelerate")
else()
    target_link_libraries(testBLASLAPACK egttoolsNumerical "-framework Accelerate")
endif ()
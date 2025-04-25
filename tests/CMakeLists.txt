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

# ------------------------------------------------------------------------------
# Enable CTest and test integration
# ------------------------------------------------------------------------------

enable_testing()

# ------------------------------------------------------------------------------
# Helper function to register a test binary
# ------------------------------------------------------------------------------

function(add_egttools_test target_name source_file)
    add_executable(${target_name} ${source_file})
    target_link_libraries(${target_name} PRIVATE egttoolsNumerical)

    if (Boost_FOUND)
        target_include_directories(${target_name} PRIVATE ${Boost_INCLUDE_DIRS})
        target_link_libraries(${target_name} PRIVATE Boost::multiprecision)
    endif ()

    if (OpenMP_CXX_FOUND)
        target_link_libraries(${target_name} PRIVATE OpenMP::OpenMP_CXX)
    endif ()

    if (APPLE)
        target_link_libraries(${target_name} PRIVATE "-framework Accelerate")
    endif ()

    # ✅ Register test with CTest
    add_test(NAME ${target_name} COMMAND ${target_name})
endfunction()

# ------------------------------------------------------------------------------
# Build internal numerical test library
# ------------------------------------------------------------------------------

add_library(egttoolsNumerical STATIC
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
)

# ------------------------------------------------------------------------------
# External dependencies
# ------------------------------------------------------------------------------

find_package(Boost REQUIRED COMPONENTS multiprecision)

if (Boost_FOUND)
    get_target_property(Boost_INCLUDE_DIRS Boost::multiprecision INTERFACE_INCLUDE_DIRECTORIES)
    message(STATUS "Boost Include Directories: ${Boost_INCLUDE_DIRS}")
    target_include_directories(egttoolsNumerical PRIVATE ${Boost_INCLUDE_DIRS})
    target_link_libraries(egttoolsNumerical PRIVATE Boost::multiprecision)
    add_definitions(-DHAS_BOOST)
endif ()

# ------------------------------------------------------------------------------
# Register C++ test executables
# ------------------------------------------------------------------------------

add_egttools_test(testPMrun cpp/test_PairwiseMoran_run.cpp)
add_egttools_test(testPMBstationary_distribution_sparse cpp/test_PairwiseMoran_stationary_distribution_sparse.cpp)
add_egttools_test(testPMBstationary_distribution_dense cpp/test_PairwiseMoran_stationary_distribution_dense.cpp)
add_egttools_test(testEigenSparseMatrix cpp/test_eigen_sparse_matrix.cpp)
add_egttools_test(testOrderedSamplingWithoutReplacement cpp/test_ordered_sampling_without_replacement.cpp)
add_egttools_test(testCRDGame cpp/test_crd_game.cpp)
add_egttools_test(testCRDTUGame cpp/test_crd_tu_game.cpp)
add_egttools_test(testOneShotCRDGame cpp/test_oneshotcrd_game.cpp)
add_egttools_test(testTimingUncertainty cpp/test_timing_uncertainty.cpp)
add_egttools_test(testBinomial cpp/test_binomial_coefficient.cpp)
add_egttools_test(testBinomialTime cpp/test_binomial_coefficient_with_time.cpp)
add_egttools_test(testGameHolders cpp/test_game_holders.cpp)
add_egttools_test(testGeneralEvolverNetwork cpp/test_general_evolver_network.cpp)
add_egttools_test(testNetworkSync cpp/test_network_sync.cpp)
add_egttools_test(testOneShotNetworkGame cpp/test_one_shot_network_game.cpp)
add_egttools_test(testNetworkEvolver cpp/test_network_evolver.cpp)
add_egttools_test(testFixations cpp/test_calculate_fixation_probabilities.cpp)
add_egttools_test(testTSLRUCache cpp/test_thread_safe_lru_cache.cpp)

if (APPLE)
    add_egttools_test(testBLASLAPACK cpp/test_blas_lapack.cpp)
    add_egttools_test(testCalculateSDSparse cpp/test_calculate_stationary_distribution_sparse.cpp)

    if (OpenMP_CXX_FOUND)
        set_target_properties(testCalculateSDSparse PROPERTIES
                BUILD_RPATH "${LIBOMP_DIR}/lib"
                INSTALL_RPATH "${LIBOMP_DIR}/lib"
        )
        target_include_directories(testCalculateSDSparse PRIVATE ${LIBOMP_DIR}/include)
        target_link_libraries(testCalculateSDSparse PRIVATE ${LIBOMP_DIR}/lib/libomp.dylib)
    endif ()
endif ()

# ------------------------------------------------------------------------------
# Pytest integration via CTest
# ------------------------------------------------------------------------------

add_custom_target(pytest
        COMMAND ${CMAKE_COMMAND} -E env
        PYTHONPATH=$<TARGET_FILE_DIR:numerical_>
        ${Python_EXECUTABLE} -m pytest -v
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        DEPENDS numerical)

add_test(NAME PythonTests
        COMMAND ${CMAKE_COMMAND} -E env
        PYTHONPATH=$<TARGET_FILE_DIR:numerical_>
        ${Python_EXECUTABLE} -m pytest -v
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

set_tests_properties(PythonTests PROPERTIES LABELS "python")

# Convenience alias
add_custom_target(tests DEPENDS pytest)
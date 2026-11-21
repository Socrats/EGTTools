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

enable_testing()

# ------------------------------------------------------------------------------
# Assertions stay enabled in the tests
# ------------------------------------------------------------------------------
# The C++ tests check results with assert(), which NDEBUG (set by Release and
# RelWithDebInfo builds) compiles out. Undefine it for the test programs, so the
# tests, and the asserts in the library headers they include, are checked in every
# build type. egttools_core itself is the library the Python extension uses, built
# with the normal flags. The option comes after the build-type flags on the command
# line, so it takes precedence. testAssertionsEnabled fails if this stops working.

set(EGTTOOLS_TEST_KEEP_ASSERTS $<IF:$<CXX_COMPILER_ID:MSVC>,/UNDEBUG,-UNDEBUG>)

# ------------------------------------------------------------------------------
# Helper function to register a test binary
# ------------------------------------------------------------------------------

function(add_egttools_test target_name source_file)
    add_executable(${target_name} ${source_file})
    target_link_libraries(${target_name} PRIVATE egttools_core)
    target_compile_options(${target_name} PRIVATE ${EGTTOOLS_TEST_KEEP_ASSERTS})

    add_test(NAME ${target_name} COMMAND ${target_name})
    set_property(GLOBAL APPEND PROPERTY EGTTOOLS_CPP_TEST_TARGETS ${target_name})
endfunction()

# ------------------------------------------------------------------------------
# Register C++ test executables
# ------------------------------------------------------------------------------

add_egttools_test(testAssertionsEnabled cpp/test_assertions_enabled.cpp)
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
add_egttools_test(testBinomialCoeffOverflow cpp/test_binomial_coeff_overflow.cpp)
add_egttools_test(testGameHolders cpp/test_game_holders.cpp)
add_egttools_test(testGeneralEvolverNetwork cpp/test_general_evolver_network.cpp)
add_egttools_test(testNetworkSync cpp/test_network_sync.cpp)
add_egttools_test(testOneShotNetworkGame cpp/test_one_shot_network_game.cpp)
add_egttools_test(testNetworkEvolver cpp/test_network_evolver.cpp)
add_egttools_test(testFixations cpp/test_calculate_fixation_probabilities.cpp)
add_egttools_test(testTSLRUCache cpp/test_thread_safe_lru_cache.cpp)
add_egttools_test(testPairwiseComparisonAnalytical cpp/test_pairwise_comparison_analytical.cpp)
add_egttools_test(testMLSTraulsen cpp/test_mls_traulsen.cpp)
add_egttools_test(testMLSGarcia cpp/test_mls_garcia.cpp)

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
# Sanitizer canary (only in builds with EGTTOOLS_SANITIZE)
# ------------------------------------------------------------------------------
# One test per enabled sanitizer: the program commits that sanitizer's error on
# purpose, and the test passes only if the sanitizer's report is in the output
# (CTest ignores the exit code when PASS_REGULAR_EXPRESSION is set). A build that
# is not instrumented prints nothing and fails.

if (EGTTOOLS_SANITIZE)
    add_executable(testSanitizerCanary cpp/test_sanitizer_canary.cpp)
    set_property(GLOBAL APPEND PROPERTY EGTTOOLS_CPP_TEST_TARGETS testSanitizerCanary)
    find_package(Threads REQUIRED)
    target_link_libraries(testSanitizerCanary PRIVATE Threads::Threads)

    set(_canary_pattern_address "AddressSanitizer: heap-buffer-overflow")
    set(_canary_pattern_undefined "runtime error: signed integer overflow")
    set(_canary_pattern_thread "ThreadSanitizer: data race")
    foreach (_sanitizer IN ITEMS address undefined thread)
        if (_sanitizer IN_LIST EGTTOOLS_SANITIZE)
            add_test(NAME testSanitizerCanary_${_sanitizer} COMMAND testSanitizerCanary ${_sanitizer})
            # On macOS the sanitizers end the process with abort() by default, which CTest
            # counts as a failure whatever the output; make them exit with an error code.
            set_tests_properties(testSanitizerCanary_${_sanitizer} PROPERTIES
                    PASS_REGULAR_EXPRESSION "${_canary_pattern_${_sanitizer}}"
                    ENVIRONMENT "ASAN_OPTIONS=abort_on_error=0;UBSAN_OPTIONS=abort_on_error=0:halt_on_error=1;TSAN_OPTIONS=abort_on_error=0")
        endif ()
    endforeach ()
endif ()

# ------------------------------------------------------------------------------
# egttools_cpp_tests: builds every C++ test program registered on this platform
# ------------------------------------------------------------------------------
# Lets CI (and developers) build only the C++ tests, without the Python extension:
#   cmake --build <build-dir> --target egttools_cpp_tests

get_property(_egttools_cpp_test_targets GLOBAL PROPERTY EGTTOOLS_CPP_TEST_TARGETS)
add_custom_target(egttools_cpp_tests DEPENDS ${_egttools_cpp_test_targets})

# The Python tests are run with pytest directly (see CONTRIBUTING.md), not through CMake/CTest.
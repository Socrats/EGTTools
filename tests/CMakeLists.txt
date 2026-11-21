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
# Catch2 v3
# ------------------------------------------------------------------------------
# With the vcpkg toolchain, Catch2 comes from the manifest feature "tests", which the root
# CMakeLists.txt enables when BUILD_NUMERICAL_TESTS=ON. Other builds must provide it
# (e.g. conda-forge's catch2).

find_package(Catch2 3 CONFIG QUIET)
if (NOT Catch2_FOUND)
    message(FATAL_ERROR
            "BUILD_NUMERICAL_TESTS=ON needs Catch2 v3, which was not found. With vcpkg it is "
            "installed through the manifest feature \"tests\"; otherwise install Catch2 v3 and add "
            "its prefix to CMAKE_PREFIX_PATH, or configure with -DBUILD_NUMERICAL_TESTS=OFF.")
endif ()
include(Catch)

# ------------------------------------------------------------------------------
# egttools_tests: one Catch2 program with every C++ test
# ------------------------------------------------------------------------------
# Each TEST_CASE is registered as its own CTest test (catch_discover_tests). Discovery runs
# the program when ctest starts (PRE_TEST), not after linking, so building the tests never
# runs them (sanitizer builds, cross-compilation).
#
# Library asserts stay enabled: NDEBUG (set by Release and RelWithDebInfo builds) compiles
# out assert(). The test program undefines it, so the asserts in the library headers it
# includes (Group, LruCache, Utils.hpp) are checked too. egttools_core itself is the library
# the Python extension uses, built with the normal flags. The option comes after the
# build-type flags on the command line, so it takes precedence. The "build configuration"
# test fails if this stops working.

add_executable(egttools_tests
        cpp/test_build_configuration.cpp
        cpp/test_distributions.cpp
        cpp/test_sampling.cpp
        cpp/test_lru_caches.cpp
        cpp/test_timing_uncertainty.cpp
        cpp/test_games.cpp
        cpp/test_crd_games.cpp
        cpp/test_pairwise_comparison.cpp
        cpp/test_pairwise_comparison_numerical.cpp
        cpp/test_replicator_dynamics.cpp
        cpp/test_networks.cpp
        cpp/test_mls.cpp
)
if (APPLE)
    # Checks that Accelerate's BLAS is linked and gives the right result.
    target_sources(egttools_tests PRIVATE cpp/test_blas_accelerate.cpp)
endif ()

target_link_libraries(egttools_tests PRIVATE egttools_core Catch2::Catch2WithMain)
target_compile_options(egttools_tests PRIVATE $<IF:$<CXX_COMPILER_ID:MSVC>,/UNDEBUG,-UNDEBUG>)
set_property(GLOBAL APPEND PROPERTY EGTTOOLS_CPP_TEST_TARGETS egttools_tests)

# Test case names must not contain ';': Catch2's discovery script (CatchAddTests.cmake) splits
# the test list on it, and every test is then registered as NOTFOUND.
catch_discover_tests(egttools_tests DISCOVERY_MODE PRE_TEST)

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
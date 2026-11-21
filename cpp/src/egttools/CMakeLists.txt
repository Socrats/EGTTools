# Copyright (c) 2019-2026  Elias Fernandez
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
# egttools_core: the C++ library
# ------------------------------------------------------------------------------
# Built once and linked by the Python extension (numerical_) and the C++ tests,
# so both use the same object code. Everything a user of the library needs
# (headers, C++17, dependencies, configuration macros) is a PUBLIC usage
# requirement. The macros must be PUBLIC: EIGEN_USE_BLAS and HAS_ARPACK change
# the code compiled from the headers, so every file that includes them must see
# the same values.

add_library(egttools_core STATIC
        SeedGenerator.cpp
        Distributions.cpp
        Data.cpp
)
add_library(EGTTools::core ALIAS egttools_core)

add_subdirectory(utils)
add_subdirectory(finite_populations)
add_subdirectory(infinite_populations)

target_include_directories(egttools_core PUBLIC
        "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/cpp/include>"
)

target_compile_features(egttools_core PUBLIC cxx_std_17)

set_target_properties(egttools_core PROPERTIES
        CXX_EXTENSIONS FALSE
        INTERPROCEDURAL_OPTIMIZATION ${LTO_SUPPORTED}
        # Linked into a Python extension module (a shared library).
        POSITION_INDEPENDENT_CODE TRUE
        # Same visibility as numerical_; mixed settings make the macOS linker warn about weak
        # symbols that cannot be overridden at runtime.
        C_VISIBILITY_PRESET hidden
        CXX_VISIBILITY_PRESET hidden
)

if (UNIX)
    target_compile_options(egttools_core PRIVATE -Wno-unknown-pragmas)
endif ()

# pybind11_add_module compiles numerical_ with pybind11's LTO flags (full LTO with Clang/GCC),
# which override the ThinLTO that INTERPROCEDURAL_OPTIMIZATION selects. Use the same flags for the
# library, so its code is optimised as it was when compiled into the module. Flags only; the library
# does not use pybind11.
if (LTO_SUPPORTED AND TARGET pybind11::lto)
    target_link_libraries(egttools_core PRIVATE $<BUILD_INTERFACE:pybind11::lto>)
endif ()

target_link_libraries(egttools_core PUBLIC Boost::multiprecision Eigen3::Eigen)
target_compile_definitions(egttools_core PUBLIC HAS_BOOST)

if (OpenMP_CXX_FOUND)
    target_link_libraries(egttools_core PUBLIC OpenMP::OpenMP_CXX)
else ()
    message(WARNING "[OpenMP] Not found or disabled for target: egttools_core")
endif ()

egttools_enable_blas_lapack(egttools_core PUBLIC)

if (DEFINED EGTTOOLS_ARPACK_TARGET)
    target_link_libraries(egttools_core PUBLIC ${EGTTOOLS_ARPACK_TARGET})
    target_compile_definitions(egttools_core PUBLIC HAS_ARPACK=1)
    message(STATUS "[EGTtools] egttools_core: linking ${EGTTOOLS_ARPACK_TARGET} (HAS_ARPACK=1)")
endif ()

# Copyright (c) 2019-2025  Elias Fernandez
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

cmake_minimum_required(VERSION 3.30)

# ------------------------------------------------------------------------------
# Parse version from version.h
# ------------------------------------------------------------------------------

file(READ "${CMAKE_SOURCE_DIR}/cpp/src/version.h" VERSION_HEADER_CONTENT)

string(REGEX MATCH "#define EGTTOOLS_VERSION_MAJOR[ \t]+([0-9]+)" _ ${VERSION_HEADER_CONTENT})
set(EGTTOOLS_VERSION_MAJOR ${CMAKE_MATCH_1})

string(REGEX MATCH "#define EGTTOOLS_VERSION_MINOR[ \t]+([0-9]+)" _ ${VERSION_HEADER_CONTENT})
set(EGTTOOLS_VERSION_MINOR ${CMAKE_MATCH_1})

string(REGEX MATCH "#define EGTTOOLS_VERSION_PATCH[ \t]+([0-9]+)" _ ${VERSION_HEADER_CONTENT})
set(EGTTOOLS_VERSION_PATCH ${CMAKE_MATCH_1})

string(REGEX MATCH "#define EGTTOOLS_VERSION_TWEAK[ \t]+([0-9]+)" _ ${VERSION_HEADER_CONTENT})
set(EGTTOOLS_VERSION_TWEAK ${CMAKE_MATCH_1})

set(EGTTOOLS_VERSION
        "${EGTTOOLS_VERSION_MAJOR}.${EGTTOOLS_VERSION_MINOR}.${EGTTOOLS_VERSION_PATCH}.${EGTTOOLS_VERSION_TWEAK}"
)

# ------------------------------------------------------------------------------
# Project definition
# ------------------------------------------------------------------------------

project(egttools VERSION ${EGTTOOLS_VERSION} LANGUAGES CXX)

# ------------------------------------------------------------------------------
# macOS deployment target enforcement
# ------------------------------------------------------------------------------

if (APPLE)
    if (NOT DEFINED CMAKE_OSX_DEPLOYMENT_TARGET OR CMAKE_OSX_DEPLOYMENT_TARGET STREQUAL "")
        set(CMAKE_OSX_DEPLOYMENT_TARGET "11.0" CACHE STRING "Minimum macOS deployment target" FORCE)
        message(WARNING
                "
[EGTTools] CMAKE_OSX_DEPLOYMENT_TARGET was not set.
[EGTTools] Defaulting to 11.0.
[EGTTools] If you need to target an older macOS version, manually set MACOSX_DEPLOYMENT_TARGET before configuring.
"
        )
    else ()
        message(STATUS "[EGTTools] CMAKE_OSX_DEPLOYMENT_TARGET=${CMAKE_OSX_DEPLOYMENT_TARGET}")
    endif ()
endif ()

# ------------------------------------------------------------------------------
# Set cmake options
# ------------------------------------------------------------------------------
option(USE_OPENMP "Enable OpenMP multithreading" ON)
# Optional tests, disabled by default
option(BUILD_NUMERICAL_TESTS "Build C++ tests in numerical-tests/" ON)
# SKIP_VCPKG configuration
option(SKIP_VCPKG "Use SKIP_VCPKG configuration to skip vcpkg toolchain" OFF)
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
# Toolchain setup: SKIP_VCPKG vs vcpkg
# ------------------------------------------------------------------------------

if (NOT SKIP_VCPKG AND NOT DEFINED CMAKE_TOOLCHAIN_FILE)
    set(CMAKE_TOOLCHAIN_FILE "${CMAKE_SOURCE_DIR}/vcpkg/scripts/buildsystems/vcpkg.cmake"
            CACHE STRING "Toolchain file for vcpkg")
    message(STATUS "Using vcpkg toolchain: ${CMAKE_TOOLCHAIN_FILE}")
elseif (DEFINED CMAKE_TOOLCHAIN_FILE)
    message(STATUS "Toolchain file already set: ${CMAKE_TOOLCHAIN_FILE}")
else ()
    message(STATUS "Using SKIP_VCPKG/system configuration — not setting vcpkg toolchain")
endif ()

if (SKIP_VCPKG AND DEFINED CMAKE_TOOLCHAIN_FILE)
    message(FATAL_ERROR "SKIP_VCPKG=ON but CMAKE_TOOLCHAIN_FILE is defined — remove it to avoid using vcpkg")
endif ()

# ------------------------------------------------------------------------------
# Compiler options and platform-specific settings
# ------------------------------------------------------------------------------

# Set module path
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake/Modules")

# CMake policy for CMake ≥ 3.27 (e.g. Boost wrapper warning)
if (POLICY CMP0167)
    cmake_policy(SET CMP0167 NEW)
endif ()

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build: Debug, Release, RelWithDebInfo, MinSizeRel" FORCE)
endif ()

# Enable IPO/LTO
include(CheckIPOSupported)
check_ipo_supported(RESULT LTO_SUPPORTED OUTPUT LTO_ERROR)
if (LTO_SUPPORTED)
    message(STATUS "IPO / LTO enabled")
else ()
    message(STATUS "IPO / LTO not supported: <${LTO_ERROR}>")
endif ()

# -------------------------------------------
# MSVC Specific Configuration
# -------------------------------------------
# Compiler-specific build options
if (MSVC)
    add_compile_options(/permissive- /Zc:inline /utf-8)
    add_compile_definitions(_CRT_SECURE_NO_WARNINGS UNICODE NOMINMAX)

    if (MSVC_VERSION GREATER_EQUAL 1920)
        add_compile_options(/d2FH4-)
    endif ()
endif ()

# macOS deployment
if (APPLE)
    message(STATUS "-- Building for architecture: ${CMAKE_OSX_ARCHITECTURES}")
    message(STATUS "Checking which MACOSX_DEPLOYMENT_TARGET to use - ${CMAKE_OSX_DEPLOYMENT_TARGET}")
    set(MACOSX_DEPLOYMENT_TARGET ${CMAKE_OSX_DEPLOYMENT_TARGET})
endif ()

# Warnings for non-Windows
if (NOT WIN32)
    add_definitions(-Wall -Wextra)
endif ()

# C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ------------------------------------------------------------------------------
# Dependencies: BLAS, LAPACK, LAPACKE, Eigen, OpenMP, Boost
# ------------------------------------------------------------------------------

# BLAS, LAPACK and LAPACKE
include(EnableBLASLapack)

# OpenMP
include(EnableOpenMP)

# Boost
find_package(Boost REQUIRED COMPONENTS multiprecision)

# Eigen
find_package(Eigen3 REQUIRED CONFIG)
include_directories(SYSTEM ${EIGEN3_INCLUDE_DIR})
message(STATUS "Eigen3 include directory: ${EIGEN3_INCLUDE_DIR}")

# ------------------------------------------------------------------------------
# Python & pybind11 setup
# ------------------------------------------------------------------------------

set(Python_FIND_IMPLEMENTATIONS CPython PyPy)
find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
set(PYBIND11_FINDPYTHON ON)
add_subdirectory(pybind11)
message(STATUS "Python include dirs: ${Python_INCLUDE_DIRS}")

# ------------------------------------------------------------------------------
# RPATH settings
# ------------------------------------------------------------------------------

include(GNUInstallDirs)
set(CMAKE_INSTALL_RPATH "$ORIGIN/${CMAKE_INSTALL_LIBDIR}")
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)

# ------------------------------------------------------------------------------
# Export targets
# ------------------------------------------------------------------------------
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

# Generate version file
write_basic_package_version_file(
        "${CMAKE_CURRENT_BINARY_DIR}/EGTToolsConfigVersion.cmake"
        VERSION ${PROJECT_VERSION}
        COMPATIBILITY SameMajorVersion
)

configure_package_config_file(
        "${CMAKE_SOURCE_DIR}/cmake/EGTToolsConfig.cmake.in"
        "${CMAKE_CURRENT_BINARY_DIR}/EGTToolsConfig.cmake"
        INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/EGTTools
)

# Install generated config + version files
install(FILES
        "${CMAKE_CURRENT_BINARY_DIR}/EGTToolsConfig.cmake"
        "${CMAKE_CURRENT_BINARY_DIR}/EGTToolsConfigVersion.cmake"
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/EGTTools
)

# Install exported targets
install(EXPORT EGTToolsTargets
        FILE EGTToolsTargets.cmake
        NAMESPACE EGTTools::
        DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/EGTTools"
)

# ------------------------------------------------------------------------------
# Subdirectories (main source, tests, docs)
# ------------------------------------------------------------------------------

include_directories(${PROJECT_SOURCE_DIR}/cpp/include)
add_subdirectory(${PROJECT_SOURCE_DIR}/cpp/src)
add_subdirectory(${PROJECT_SOURCE_DIR}/docs)

if (BUILD_NUMERICAL_TESTS)
    add_subdirectory(tests)
endif ()

# ------------------------------------------------------------------------------
# scikit-build install logic (bundling .so/.dylib if needed)
# ------------------------------------------------------------------------------
if (SKBUILD)
    install(TARGETS numerical_ LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX})

    if (WIN32)
        install(FILES ${CMAKE_INSTALL_SYSTEM_RUNTIME_LIBS} DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)
    else ()
        # OpenMP
        include(InstallOpenMP)
        # BLAS / LAPACK
        include(InstallBLASLAPACK)
    endif ()
endif ()

# ------------------------------------------------------------------------------
# Build information
# ------------------------------------------------------------------------------
include(AddBuildInfo)



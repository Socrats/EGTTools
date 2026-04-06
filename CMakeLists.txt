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

cmake_minimum_required(VERSION 3.20...4.2)

# ------------------------------------------------------------------------------
# Parse version from version.h
# ------------------------------------------------------------------------------

file(READ "${CMAKE_SOURCE_DIR}/cpp/src/version.h" VERSION_HEADER_CONTENT)

string(REGEX MATCH "#define EGTTOOLS_VERSION_MAJOR[ \t]+([0-9]+)" _ "${VERSION_HEADER_CONTENT}")
set(EGTTOOLS_VERSION_MAJOR "${CMAKE_MATCH_1}")

string(REGEX MATCH "#define EGTTOOLS_VERSION_MINOR[ \t]+([0-9]+)" _ "${VERSION_HEADER_CONTENT}")
set(EGTTOOLS_VERSION_MINOR "${CMAKE_MATCH_1}")

string(REGEX MATCH "#define EGTTOOLS_VERSION_PATCH[ \t]+([0-9]+)" _ "${VERSION_HEADER_CONTENT}")
set(EGTTOOLS_VERSION_PATCH "${CMAKE_MATCH_1}")

string(REGEX MATCH "#define EGTTOOLS_VERSION_TWEAK[ \t]+([0-9]+)" _ "${VERSION_HEADER_CONTENT}")
set(EGTTOOLS_VERSION_TWEAK "${CMAKE_MATCH_1}")

set(EGTTOOLS_VERSION
        "${EGTTOOLS_VERSION_MAJOR}.${EGTTOOLS_VERSION_MINOR}.${EGTTOOLS_VERSION_PATCH}.${EGTTOOLS_VERSION_TWEAK}"
)

# ------------------------------------------------------------------------------
# Project definition
# ------------------------------------------------------------------------------

project(egttools VERSION ${EGTTOOLS_VERSION} LANGUAGES CXX)

# ------------------------------------------------------------------------------
# Options
# ------------------------------------------------------------------------------

option(USE_OPENMP "Enable OpenMP multithreading" ON)
option(BUILD_NUMERICAL_TESTS "Build C++ tests in tests/" ON)
option(SKIP_VCPKG "Skip vcpkg toolchain setup" OFF)

# Silence third-party CMake noise if desired
option(EGTTOOLS_SUPPRESS_THIRD_PARTY_CMAKE_WARNINGS "Suppress third-party CMake dev/deprecation warnings" ON)

if (EGTTOOLS_SUPPRESS_THIRD_PARTY_CMAKE_WARNINGS)
    set(CMAKE_SUPPRESS_DEVELOPER_WARNINGS 1 CACHE INTERNAL "" FORCE)
    set(CMAKE_WARN_DEPRECATED OFF CACHE INTERNAL "" FORCE)
endif ()

# ------------------------------------------------------------------------------
# Toolchain setup: SKIP_VCPKG vs vcpkg
# ------------------------------------------------------------------------------

if (NOT SKIP_VCPKG AND NOT DEFINED CMAKE_TOOLCHAIN_FILE)
    set(CMAKE_TOOLCHAIN_FILE
            "${CMAKE_SOURCE_DIR}/vcpkg/scripts/buildsystems/vcpkg.cmake"
            CACHE STRING "Toolchain file for vcpkg"
    )
    message(STATUS "Using vcpkg toolchain: ${CMAKE_TOOLCHAIN_FILE}")
elseif (DEFINED CMAKE_TOOLCHAIN_FILE)
    message(STATUS "Toolchain file already set: ${CMAKE_TOOLCHAIN_FILE}")
else ()
    message(STATUS "Using SKIP_VCPKG/system configuration — not setting vcpkg toolchain")
endif ()

if (SKIP_VCPKG AND DEFINED CMAKE_TOOLCHAIN_FILE)
    message(FATAL_ERROR "SKIP_VCPKG=ON but CMAKE_TOOLCHAIN_FILE is defined")
endif ()

# ------------------------------------------------------------------------------
# Policies / module path
# ------------------------------------------------------------------------------

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake/Modules")

if (POLICY CMP0167)
    cmake_policy(SET CMP0167 NEW)
endif ()

# ------------------------------------------------------------------------------
# Build type / compiler settings
# ------------------------------------------------------------------------------

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif ()

include(CheckIPOSupported)
check_ipo_supported(RESULT LTO_SUPPORTED OUTPUT LTO_ERROR)
if (LTO_SUPPORTED)
    message(STATUS "IPO / LTO enabled")
else ()
    message(STATUS "IPO / LTO not supported: <${LTO_ERROR}>")
endif ()

if (MSVC)
    add_compile_options(/permissive- /Zc:inline /utf-8)
    add_compile_definitions(_CRT_SECURE_NO_WARNINGS UNICODE NOMINMAX)

    if (MSVC_VERSION GREATER_EQUAL 1920)
        add_compile_options(/d2FH4-)
    endif ()
else ()
    add_compile_options(-Wall -Wextra)
endif ()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ------------------------------------------------------------------------------
# macOS deployment target enforcement
# ------------------------------------------------------------------------------

if (APPLE)
    if (NOT DEFINED CMAKE_OSX_DEPLOYMENT_TARGET OR CMAKE_OSX_DEPLOYMENT_TARGET STREQUAL "")
        set(CMAKE_OSX_DEPLOYMENT_TARGET "11.0" CACHE STRING "Minimum macOS deployment target" FORCE)
        message(WARNING
                "[EGTTools] CMAKE_OSX_DEPLOYMENT_TARGET was not set. Defaulting to 11.0."
        )
    else ()
        message(STATUS "[EGTTools] CMAKE_OSX_DEPLOYMENT_TARGET=${CMAKE_OSX_DEPLOYMENT_TARGET}")
    endif ()

    message(STATUS "-- Building for architecture: ${CMAKE_OSX_ARCHITECTURES}")
    message(STATUS "Checking which MACOSX_DEPLOYMENT_TARGET to use - ${CMAKE_OSX_DEPLOYMENT_TARGET}")
    set(MACOSX_DEPLOYMENT_TARGET "${CMAKE_OSX_DEPLOYMENT_TARGET}")
endif ()

# ------------------------------------------------------------------------------
# Dependencies
# ------------------------------------------------------------------------------

include(EnableBLASLapack)
include(EnableOpenMP)

# Use config package directly instead of FindBoost
find_package(boost_multiprecision CONFIG REQUIRED)

find_package(Eigen3 CONFIG REQUIRED)
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
# Global include dirs
# ------------------------------------------------------------------------------

include_directories(${PROJECT_SOURCE_DIR}/cpp/include)
include_directories(SYSTEM ${EIGEN3_INCLUDE_DIR})

# ------------------------------------------------------------------------------
# RPATH settings
# ------------------------------------------------------------------------------

include(GNUInstallDirs)
set(CMAKE_INSTALL_RPATH "$ORIGIN/${CMAKE_INSTALL_LIBDIR}")
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)

# ------------------------------------------------------------------------------
# Export targets
# ------------------------------------------------------------------------------

include(CMakePackageConfigHelpers)

write_basic_package_version_file(
        "${CMAKE_CURRENT_BINARY_DIR}/EGTToolsConfigVersion.cmake"
        VERSION ${PROJECT_VERSION}
        COMPATIBILITY SameMajorVersion
)

configure_package_config_file(
        "${CMAKE_SOURCE_DIR}/cmake/EGTToolsConfig.cmake.in"
        "${CMAKE_CURRENT_BINARY_DIR}/EGTToolsConfig.cmake"
        INSTALL_DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/EGTTools"
)

if (NOT SKBUILD)
    install(FILES
            "${CMAKE_CURRENT_BINARY_DIR}/EGTToolsConfig.cmake"
            "${CMAKE_CURRENT_BINARY_DIR}/EGTToolsConfigVersion.cmake"
            DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/EGTTools"
    )

    install(EXPORT EGTToolsTargets
            FILE EGTToolsTargets.cmake
            NAMESPACE EGTTools::
            DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/EGTTools"
    )
endif ()

# ------------------------------------------------------------------------------
# Subdirectories
# ------------------------------------------------------------------------------

add_subdirectory(${PROJECT_SOURCE_DIR}/cpp/src)
add_subdirectory(${PROJECT_SOURCE_DIR}/docs)

if (BUILD_NUMERICAL_TESTS)
    add_subdirectory(tests)
endif ()

# ------------------------------------------------------------------------------
# scikit-build install logic
# ------------------------------------------------------------------------------

if (SKBUILD)
    install(TARGETS numerical_ LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX})

    if (WIN32)
        install(FILES ${CMAKE_INSTALL_SYSTEM_RUNTIME_LIBS} DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)
    else ()
        include(InstallOpenMP)
        include(InstallBLASLAPACK)
    endif ()
endif ()

# ------------------------------------------------------------------------------
# Build information
# ------------------------------------------------------------------------------

include(AddBuildInfo)
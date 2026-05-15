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

# SKIP_VCPKG guard — must come before project() because CMAKE_TOOLCHAIN_FILE
# is loaded by CMakeDetermineSystem.cmake *during* project(), before any code
# below project() can run. A stale cmake cache from a prior vcpkg build injects
# the vcpkg toolchain even when -DSKIP_VCPKG=ON is passed on the command line.
# Unsetting the cache variable here prevents project() from loading it.
if (SKIP_VCPKG AND DEFINED CMAKE_TOOLCHAIN_FILE AND CMAKE_TOOLCHAIN_FILE MATCHES "vcpkg")
    message(STATUS "[EGTtools] SKIP_VCPKG=ON: clearing stale vcpkg CMAKE_TOOLCHAIN_FILE before project()")
    unset(CMAKE_TOOLCHAIN_FILE CACHE)
endif ()

# ------------------------------------------------------------------------------
# Version detection: git tag → EGTTOOLS_VERSION env var → fallback
# ------------------------------------------------------------------------------

find_package(Git QUIET)
set(_EGTTOOLS_VERSION "0.0.0")

if (GIT_FOUND AND EXISTS "${CMAKE_SOURCE_DIR}/.git")
    execute_process(
            COMMAND ${GIT_EXECUTABLE} describe --tags --match "v*" --abbrev=0
            WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
            OUTPUT_VARIABLE _GIT_TAG
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET
            RESULT_VARIABLE _GIT_RESULT
    )
    if (_GIT_RESULT EQUAL 0 AND _GIT_TAG MATCHES "^v[0-9]")
        string(REGEX REPLACE "^v" "" _EGTTOOLS_VERSION "${_GIT_TAG}")
        message(STATUS "[EGTtools] Version from git tag: ${_EGTTOOLS_VERSION}")
    endif ()
endif ()

if (DEFINED ENV{EGTTOOLS_VERSION})
    set(_EGTTOOLS_VERSION "$ENV{EGTTOOLS_VERSION}")
    message(STATUS "[EGTtools] Version from EGTTOOLS_VERSION env: ${_EGTTOOLS_VERSION}")
endif ()

if (_EGTTOOLS_VERSION STREQUAL "0.0.0")
    message(WARNING
            "[EGTtools] Could not detect version from git tag or EGTTOOLS_VERSION env var. "
            "Using 0.0.0 as fallback. Set EGTTOOLS_VERSION=<version> to override."
    )
endif ()

# ------------------------------------------------------------------------------
# Project definition
# ------------------------------------------------------------------------------

project(egttools VERSION ${_EGTTOOLS_VERSION} LANGUAGES CXX)

# ------------------------------------------------------------------------------
# Options
# ------------------------------------------------------------------------------

option(USE_OPENMP "Enable OpenMP multithreading" ON)
option(BUILD_NUMERICAL_TESTS "Build C++ tests in tests/" ON)
option(SKIP_VCPKG "Skip vcpkg toolchain setup" OFF)
option(EGTTOOLS_ENABLE_ARPACK "Build native ARPACK eigensolver (requires arpack-ng)" OFF)
option(EGTTOOLS_ENABLE_PETSC "Build MPI-distributed PETSc/SLEPc eigensolver (numerical_mpi_ module)" OFF)

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
    if (CMAKE_TOOLCHAIN_FILE MATCHES "vcpkg")
        # Stale cmake cache or setup.py injected the vcpkg toolchain before
        # SKIP_VCPKG was honoured.  Clear it so the build proceeds without vcpkg.
        message(STATUS "SKIP_VCPKG=ON: clearing cached vcpkg CMAKE_TOOLCHAIN_FILE")
        unset(CMAKE_TOOLCHAIN_FILE CACHE)
    else ()
        message(FATAL_ERROR
                "SKIP_VCPKG=ON but a non-vcpkg CMAKE_TOOLCHAIN_FILE is already defined: "
                "${CMAKE_TOOLCHAIN_FILE}"
        )
    endif ()
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

# Boost (header-only multiprecision)
# - vcpkg installs per-component cmake configs  → find by component name
# - conda/system installs BoostConfig.cmake     → find parent package
if (SKIP_VCPKG)
    find_package(Boost CONFIG REQUIRED)
    if (NOT TARGET Boost::multiprecision)
        # Older conda boost or plain system install: synthesise the INTERFACE target
        find_path(BOOST_MULTIPRECISION_INCLUDE_DIR
                NAMES boost/multiprecision/cpp_bin_float.hpp
                HINTS ${CMAKE_PREFIX_PATH}
                PATH_SUFFIXES include
        )
        if (NOT BOOST_MULTIPRECISION_INCLUDE_DIR)
            message(FATAL_ERROR "Could not find Boost multiprecision headers. "
                    "Set CMAKE_PREFIX_PATH to the Boost installation root.")
        endif ()
        add_library(Boost::multiprecision INTERFACE IMPORTED GLOBAL)
        set_target_properties(Boost::multiprecision PROPERTIES
                INTERFACE_INCLUDE_DIRECTORIES "${BOOST_MULTIPRECISION_INCLUDE_DIR}"
        )
        message(STATUS "[Boost] Synthesised Boost::multiprecision from ${BOOST_MULTIPRECISION_INCLUDE_DIR}")
    endif ()
else ()
    # vcpkg provides individual component cmake configs
    find_package(boost_multiprecision CONFIG REQUIRED)
endif ()

# Eigen3
# - vcpkg installs Eigen3Config.cmake (config mode)
# - conda-forge (eigen >=3.4.0 / Eigen 5.x) also provides Eigen3Config.cmake
#   at share/eigen3/cmake/ — use CONFIG mode for both cases so our legacy
#   FindEigen3.cmake (which reads Macros.h, now gone in Eigen 5) is bypassed.
find_package(Eigen3 CONFIG REQUIRED)

# Eigen3Config.cmake only creates the Eigen3::Eigen target; extract the include
# directory from the target so the legacy include_directories() call below works.
if (NOT EIGEN3_INCLUDE_DIR AND TARGET Eigen3::Eigen)
    get_target_property(EIGEN3_INCLUDE_DIR Eigen3::Eigen INTERFACE_INCLUDE_DIRECTORIES)
endif ()
message(STATUS "Eigen3 include directory: ${EIGEN3_INCLUDE_DIR}")

# arpack-ng (optional — enabled by EGTTOOLS_ENABLE_ARPACK=ON)
# conda-forge package name: arpackng  →  target: ARPACK::ARPACK
# vcpkg / system installs may use:    arpack-ng  →  target: arpack-ng::arpack-ng
# HPC / EasyBuild installs (foss toolchain): no CMake config, but arpack.pc is available
if (EGTTOOLS_ENABLE_ARPACK)
    find_package(arpackng QUIET)
    if (arpackng_FOUND AND TARGET ARPACK::ARPACK)
        set(EGTTOOLS_ARPACK_TARGET ARPACK::ARPACK)
        message(STATUS "[EGTtools] ARPACK: found via arpackng (ARPACK::ARPACK)")
    else ()
        find_package(arpack-ng QUIET)
        if (arpack-ng_FOUND AND TARGET arpack-ng::arpack-ng)
            set(EGTTOOLS_ARPACK_TARGET arpack-ng::arpack-ng)
            message(STATUS "[EGTtools] ARPACK: found via arpack-ng (arpack-ng::arpack-ng)")
        else ()
            # Fallback for HPC environments (EasyBuild/LMOD) where arpack-ng ships a
            # pkg-config .pc file but no CMake config package.
            find_package(PkgConfig QUIET)
            if (PKG_CONFIG_FOUND)
                pkg_check_modules(ARPACK QUIET arpack)
                if (ARPACK_FOUND)
                    # EasyBuild arpack.pc sets includedir=${prefix}/include/arpack.
                    # Headers are included as <arpack/arpack.hpp>, so we need
                    # ${prefix}/include. Strip the trailing /arpack component if present.
                    set(_arpack_sys_includes)
                    foreach(_d ${ARPACK_INCLUDE_DIRS})
                        get_filename_component(_leaf "${_d}" NAME)
                        if (_leaf STREQUAL "arpack")
                            get_filename_component(_d "${_d}" DIRECTORY)
                        endif ()
                        list(APPEND _arpack_sys_includes "${_d}")
                    endforeach()
                    list(REMOVE_DUPLICATES _arpack_sys_includes)

                    # Check whether arpack/arpack.hpp is present in the system install.
                    # EasyBuild Autotools builds omit the C++ wrapper; conda-forge CMake
                    # builds include it. If absent, use the vendored copy in cpp/include/.
                    find_path(_ARPACK_HPP_DIR "arpack/arpack.hpp"
                        HINTS ${_arpack_sys_includes}
                        NO_DEFAULT_PATH)
                    if (_ARPACK_HPP_DIR)
                        set(_arpack_hpp_root "${_ARPACK_HPP_DIR}")
                        message(STATUS "[EGTtools] ARPACK: arpack.hpp found at ${_ARPACK_HPP_DIR}")
                    else ()
                        # Fall back to vendored arpack.hpp / arpack.h in cpp/include/arpack/.
                        # Those headers use <arpack/arpackdef.h> so arpackdef.h is still
                        # resolved from the system install via _arpack_sys_includes.
                        set(_arpack_hpp_root "${PROJECT_SOURCE_DIR}/cpp/include")
                        message(STATUS "[EGTtools] ARPACK: arpack.hpp not in system install — using vendored copy")
                    endif ()

                    add_library(_egttools_arpack INTERFACE IMPORTED)
                    target_link_libraries(_egttools_arpack INTERFACE ${ARPACK_LINK_LIBRARIES})
                    # _arpack_hpp_root: parent of arpack/ (for <arpack/arpack.hpp>)
                    # _arpack_sys_includes: system install base (for <arpack/arpackdef.h>)
                    target_include_directories(_egttools_arpack INTERFACE
                        "${_arpack_hpp_root}" ${_arpack_sys_includes})
                    target_link_directories(_egttools_arpack INTERFACE ${ARPACK_LIBRARY_DIRS})
                    set(EGTTOOLS_ARPACK_TARGET _egttools_arpack)
                    message(STATUS "[EGTtools] ARPACK: pkg-config ${ARPACK_VERSION}, hpp root: ${_arpack_hpp_root}, sys: ${_arpack_sys_includes}")

                    # Check whether the C-API wrappers (dnaupd_c etc.) are present in
                    # libarpack. EasyBuild Autotools builds compile only the Fortran core
                    # and omit them. Use nm to probe the installed library.
                    find_library(_ARPACK_LIB_FILE arpack
                        HINTS ${ARPACK_LIBRARY_DIRS} NO_DEFAULT_PATH)
                    set(_ARPACK_HAS_C_WRAPPERS FALSE)
                    if (_ARPACK_LIB_FILE)
                        execute_process(
                            COMMAND nm "${_ARPACK_LIB_FILE}"
                            COMMAND grep "dnaupd_c"
                            RESULT_VARIABLE _nm_exit
                            OUTPUT_QUIET ERROR_QUIET
                        )
                        if (_nm_exit EQUAL 0)
                            set(_ARPACK_HAS_C_WRAPPERS TRUE)
                        endif ()
                    endif ()

                    if (NOT _ARPACK_HAS_C_WRAPPERS)
                        message(STATUS "[EGTtools] ARPACK: dnaupd_c absent — compiling built-in Fortran→C wrappers")
                        enable_language(C)
                        add_library(_egttools_arpack_cw STATIC
                            "${PROJECT_SOURCE_DIR}/cpp/src/arpack_cwrapper.c")
                        target_include_directories(_egttools_arpack_cw PRIVATE
                            ${_arpack_sys_includes})
                        target_link_libraries(_egttools_arpack INTERFACE _egttools_arpack_cw)
                    endif ()
                else ()
                    message(WARNING "[EGTtools] EGTTOOLS_ENABLE_ARPACK=ON but arpack-ng not found — disabling")
                endif ()
            else ()
                message(WARNING "[EGTtools] EGTTOOLS_ENABLE_ARPACK=ON but arpack-ng not found — disabling")
            endif ()
        endif ()
    endif ()
endif ()

# PETSc + SLEPc + MPI (optional — enabled by EGTTOOLS_ENABLE_PETSC=ON)
# conda-forge installation:
#   conda install -c conda-forge petsc slepc petsc4py mpi4py openmpi
# Build with:
#   python setup.py build_ext --inplace -- -DEGTTOOLS_ENABLE_PETSC=ON
#     -DCMAKE_PREFIX_PATH=/path/to/egtenv
if (EGTTOOLS_ENABLE_PETSC)
    find_package(PkgConfig REQUIRED)
    # PETSc.pc and SLEPc.pc land in $PREFIX/lib/pkgconfig (conda-forge).
    pkg_check_modules(PETSC REQUIRED IMPORTED_TARGET PETSc>=3.20)
    pkg_check_modules(SLEPC REQUIRED IMPORTED_TARGET SLEPc>=3.20)
    find_package(MPI REQUIRED)
    set(EGTTOOLS_PETSC_TARGET PkgConfig::PETSC)
    set(EGTTOOLS_SLEPC_TARGET PkgConfig::SLEPC)
    message(STATUS "[EGTtools] PETSc ${PETSC_VERSION}, SLEPc ${SLEPC_VERSION}")
endif ()

# ------------------------------------------------------------------------------
# Python & pybind11 setup
# ------------------------------------------------------------------------------

set(Python_FIND_IMPLEMENTATIONS CPython PyPy)
find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)

set(PYBIND11_FINDPYTHON ON)
# Use the bundled pybind11 git submodule when available (local dev, cibuildwheel).
# Fall back to find_package() so that conda-forge builds work: GitHub release
# tarballs do not include submodules, and pybind11 is a conda host dependency.
if (EXISTS "${CMAKE_SOURCE_DIR}/pybind11/CMakeLists.txt")
    add_subdirectory(pybind11)
    message(STATUS "pybind11: using bundled submodule")
else ()
    find_package(pybind11 REQUIRED)
    message(STATUS "pybind11: using system installation (${pybind11_VERSION})")
endif ()

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
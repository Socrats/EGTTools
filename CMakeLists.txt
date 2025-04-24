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
project(egttools LANGUAGES CXX)

option(HPC "Disable HPC configuration - Use VCPKG" OFF)

# Set the vcpkg toolchain file
if (NOT DEFINED CMAKE_TOOLCHAIN_FILE AND NOT HPC)
    set(CMAKE_TOOLCHAIN_FILE "${CMAKE_SOURCE_DIR}/vcpkg/scripts/buildsystems/vcpkg.cmake" CACHE STRING "")
endif ()

# For additional Find library scripts
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/cmake/Modules/")

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build, options are: None(CMAKE_CXX_FLAGS or CMAKE_C_FLAGS used) Debug Release RelWithDebInfo MinSizeRel." FORCE)
endif ()

# Check for Link Time Optimizations with this compiler
include(CheckIPOSupported)
check_ipo_supported(RESULT LTO_SUPPORTED OUTPUT LTO_ERROR)

if (LTO_SUPPORTED)
    message(STATUS "IPO / LTO enabled")
else ()
    message(STATUS "IPO / LTO not supported: <${LTO_ERROR}>")
endif ()

# In case of windows (This piece of code was copied from https://github.com/YannickJadoul/Parselmouth/blob/master/CMakeLists.txt
if (MSVC)
    add_compile_options(/permissive- /Zc:inline) # Please follow the standard more closely, MSVC (e.g. adds ciso646 alternative tokens for logical operators)
    add_compile_options(/utf-8) # because UTF-8 is not default enough for MSVC to read and compile these files correctly
    add_compile_definitions(_CRT_SECURE_NO_WARNINGS UNICODE NOMINMAX) # Windows no "safe alternatives" warning, Windows unicode API by default, and no <windows.h> 'min' and 'max' macros
    if (MSVC_VERSION GREATER_EQUAL 1920)
        add_compile_options(/d2FH4-) # Because we don't want to link against VCRUNTIME140_1.dll (see https://cibuildwheel.readthedocs.io/en/stable/faq/#importerror-dll-load-failed-the-specific-module-could-not-be-found-error-on-windows)
    endif ()
endif ()

if (APPLE)
    message("-- Building for architecture ${CMAKE_OSX_ARCHITECTURES}")
    set(_msg "Checking which MACOSX_DEPLOYMENT_TARGET to use")
    message(STATUS "${_msg}")
    set(MACOSX_DEPLOYMENT_TARGET ${CMAKE_OSX_DEPLOYMENT_TARGET})
    message(STATUS "${_msg} - ${MACOSX_DEPLOYMENT_TARGET}")
endif ()

if (NOT WIN32)
    add_definitions(
            -Wall
            -Wextra
    )
endif ()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Find BLAS/LAPACK and Eigen
if (APPLE)
    find_library(ACCELERATE_FRAMEWORK Accelerate)
    find_library(LAPACKE_LIB lapacke PATHS /opt/local/lib ${LAPACK_DIR}/lib)
    if (ACCELERATE_FRAMEWORK AND LAPACKE_LIB)
        message(STATUS "Using Accelerate framework and LAPACKE")
        add_compile_definitions(EIGEN_USE_BLAS EIGEN_USE_LAPACKE)
    else ()
        message(FATAL_ERROR "Accelerate framework or LAPACKE library not found")
    endif ()
else ()
    # Find BLAS and LAPACK (these are usually bundled together)
    find_package(BLAS REQUIRED)
    find_package(LAPACK REQUIRED)

    # Optionally try to find LAPACKE
    find_library(LAPACKE_LIB
            NAMES lapacke
            HINTS ${CMAKE_PREFIX_PATH}
            PATH_SUFFIXES lib lib64
    )

    if (BLAS_FOUND AND LAPACK_FOUND)
        message(STATUS "BLAS and LAPACK found")
        add_compile_definitions(EIGEN_USE_BLAS)

        if (LAPACKE_LIB)
            message(STATUS "LAPACKE found")
            add_compile_definitions(EIGEN_USE_LAPACKE)
        else()
            message(STATUS "LAPACKE not found — using LAPACK directly")
        endif()

    else()
        message(FATAL_ERROR "BLAS or LAPACK not found — required for Eigen acceleration")
    endif()

endif ()

# Necessary for certain MacOX versions to find homebrew OpenMP
# Optionally allow disabling OpenMP
option(USE_OPENMP "Enable OpenMP multithreading" ON)

if (USE_OPENMP)
    find_package(OpenMP)
endif()

find_package(Eigen3 ${EIGEN_VERSION_REQUIRED} QUIET CONFIG)
if (NOT EIGEN3_FOUND)
    find_package(Eigen3 ${EIGEN_VERSION_REQUIRED} REQUIRED)
    if (NOT EIGEN3_FOUND)
        message(FATAL_ERROR "Please point the environment variable EIGEN3_INCLUDE_DIR to the include directory of your Eigen3 installation.")
    endif ()
else ()
    message(STATUS "Eigen3 found: ${Eigen3_DIR}")
    message(STATUS "Eigen3 include directory: ${EIGEN3_INCLUDE_DIR}")
endif ()

if (EIGEN3_FOUND)
    include_directories(SYSTEM ${EIGEN3_INCLUDE_DIR})
    message(STATUS "Eigen found")
endif ()

#set(Boost_DEBUG ON)
find_package(Boost REQUIRED COMPONENTS multiprecision)

set(Python_FIND_IMPLEMENTATIONS CPython PyPy)
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
set(PYBIND11_FINDPYTHON ON)
# Add external dependencies
add_subdirectory(pybind11)


message(STATUS "${Python_INCLUDE_DIRS}")

include_directories(${PROJECT_SOURCE_DIR}/cpp/include)
add_subdirectory(${PROJECT_SOURCE_DIR}/cpp/src)
add_subdirectory(${PROJECT_SOURCE_DIR}/tests)
add_subdirectory(${PROJECT_SOURCE_DIR}/docs)

if (SKBUILD)
    if (OpenMP_CXX_FOUND)
        install(FILES ${LIBOMP_DIR}/lib/libomp.dylib DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)
    endif ()
    install(TARGETS numerical_ LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX})
endif ()

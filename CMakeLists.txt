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

cmake_minimum_required(VERSION 3.30)
project(egttools LANGUAGES CXX)


#set(SKIP_OPENMP TRUE)

# Set PYTHONPATH

#set(PYTHONPATH src/:${PYTHONPATH})

# Set the vcpkg toolchain file
if(NOT DEFINED CMAKE_TOOLCHAIN_FILE)
    set(CMAKE_TOOLCHAIN_FILE "${CMAKE_SOURCE_DIR}/vcpkg/scripts/buildsystems/vcpkg.cmake" CACHE STRING "")
endif()

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

# Necessary for certain MacOX versions to find homebrew OpenMP
#find_package(OpenMP)
if (APPLE AND NOT SKIP_OPENMP)
    if (CMAKE_C_COMPILER_ID MATCHES "Clang\$")
        set(OpenMP_C_FLAGS "-Xpreprocessor -fopenmp -I${LIBOMP_DIR}/include")
        set(OpenMP_C_LIB_NAMES "omp")
        set(OpenMP_omp_LIBRARY ${LIBOMP_DIR}/lib/libomp.dylib)
        set(LDFLAGS "$LDFLAGS -Wl,-rpath,${LIBOMP_DIR}/lib -L${LIBOMP_DIR}/lib -lomp")
        set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
        set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
    endif ()

    if (CMAKE_CXX_COMPILER_ID MATCHES "Clang\$")
        set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp -I${LIBOMP_DIR}/include")
        set(OpenMP_CXX_LIB_NAMES "omp")
        set(OpenMP_omp_LIBRARY ${LIBOMP_DIR}/lib/libomp.dylib)
        set(LDFLAGS "$LDFLAGS -Wl,-rpath,${LIBOMP_DIR}/lib -L${LIBOMP_DIR}/lib -lomp")
        set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
        set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
    endif ()
#    set(CPPFLAGS "${CPPFLAGS} -Xpreprocessor -fopenmp")
#    set(CFLAGS "${CFLAGS} -I${LIBOMP_DIR}/include")
#    set(CXXFLAGS "${CXXFLAGS} -I${LIBOMP_DIR}/include")
#    set(LDFLAGS "${LDFLAGS} -Wl,-rpath,${LIBOMP_DIR}/lib -L${LIBOMP_DIR}/lib -lomp")
endif ()
#find_package(OpenMP)

if (NOT WIN32)
    add_definitions(
            -Wall
            -Wextra
    )
endif ()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

#set(EIGEN_VERSION_REQUIRED 3.3)

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

set(Boost_DEBUG ON)
find_package(Boost REQUIRED COMPONENTS multiprecision)

#include_directories(SYSTEM ${PYTHON_INCLUDE_DIRS})

# Support for OpenMP parallelization - not on windows
if (NOT MSVC AND NOT SKIP_OPENMP)
    find_package(OpenMP)
    if (OpenMP_CXX_FOUND)
        include_directories(SYSTEM ${OPENMP_INCLUDE_DIR})
        message(STATUS "OpenMP enabled")
        if (NOT APPLE)
            SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
            SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS} -fopenmp")
        else ()
            SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}")
            SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
        endif ()
    endif ()
endif ()

# Currently, scikit-build does not support FindPython.
#if (SKBUILD)
#    set(Python_EXECUTABLE "${PYTHON_EXECUTABLE}")
#    set(Python_INCLUDE_DIR "${PYTHON_INCLUDE_DIR}")
#    unset(PYTHON_EXECUTABLE)
#    unset(PYTHON_INCLUDE_DIR)
#    unset(PYTHON_LIBRARY)
#    unset(PYTHON_VERSION_STRING)
#endif ()

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
    install(FILES ${LIBOMP_DIR}/lib/libomp.dylib DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)
    install(TARGETS numerical_ LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX})
endif ()

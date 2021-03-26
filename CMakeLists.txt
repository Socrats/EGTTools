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

cmake_minimum_required(VERSION 3.17)
project(egttools LANGUAGES CXX)

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

# Necessary for certain MacOX versions to find homebrew OpenMP
if(APPLE)
    execute_process(COMMAND brew --prefix libomp OUTPUT_VARIABLE LIBOMP_DIR)
    string(STRIP ${LIBOMP_DIR} LIBOMP_DIR)

    if(CMAKE_C_COMPILER_ID MATCHES "Clang\$")
        set(OpenMP_C_FLAGS "-Xpreprocessor -fopenmp -I${LIBOMP_DIR}/include")
        set(OpenMP_C_LIB_NAMES "omp")
        set(OpenMP_omp_LIBRARY ${LIBOMP_DIR}/lib/libomp.a)
    endif()

    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang\$")
        set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp -I${LIBOMP_DIR}/include")
        set(OpenMP_CXX_LIB_NAMES "omp")
        set(OpenMP_omp_LIBRARY ${LIBOMP_DIR}/lib/libomp.a)
    endif()

endif()

if (NOT WIN32)
    add_definitions(
            -Wall
            -Wextra
    )
endif ()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(EIGEN_VERSION_REQUIRED 3.3)
#set(PYTHON_VERSION 3)
#set(PYBIND11_CPP_STANDARD -std=c++1z) # Experimental C++17 support

find_package(Eigen3 ${EIGEN_VERSION_REQUIRED} QUIET CONFIG)
if (NOT EIGEN3_FOUND)
    find_package(Eigen3 ${EIGEN_VERSION_REQUIRED} REQUIRED)
    if (NOT EIGEN3_FOUND)
        message(FATAL_ERROR "Please point the environment variable EIGEN3_INCLUDE_DIR to the include directory of your Eigen3 installation.")
    endif()
endif()

if (EIGEN3_FOUND)
    include_directories(SYSTEM ${EIGEN3_INCLUDE_DIR})
    message(STATUS "Eigen found")
endif ()

include_directories(SYSTEM ${PYTHON_INCLUDE_DIRS})

# Support for OpenMP parallelization
find_package(OpenMP)
if (OPENMP_FOUND)
    include_directories(SYSTEM ${OPENMP_INCLUDE_DIR})
    message(STATUS "OpenMP enabled")
endif ()

# Currently, scikit-build does not support FindPython.
if(SKBUILD)
    set(Python_EXECUTABLE "${PYTHON_EXECUTABLE}")
    set(Python_INCLUDE_DIR "${PYTHON_INCLUDE_DIR}")
    unset(PYTHON_EXECUTABLE)
    unset(PYTHON_INCLUDE_DIR)
    unset(PYTHON_LIBRARY)
    unset(PYTHON_VERSION_STRING)
endif()

set(Python_FIND_IMPLEMENTATIONS CPython PyPy)
find_package(Python3 COMPONENTS Interpreter Development.Module REQUIRED)
if (PYTHON3_FOUND)
    message("-- Using Python version ${PYTHON_VERSION_STRING}")
endif ()
add_subdirectory(pybind11)

include_directories(${PROJECT_SOURCE_DIR}/include)
add_subdirectory(${PROJECT_SOURCE_DIR}/src)
#add_subdirectory(${PROJECT_SOURCE_DIR}/tests)
#add_subdirectory(${PROJECT_SOURCE_DIR}/docs)

if(SKBUILD)
    install(TARGETS numerical LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX})
endif()

# Python bindings
#if (MAKE_PYTHON)
#    # Wrappers -- if unused macros will skip them
#    set(PYTHON_FILES
#            src/python/EGTtoolsWrapper.cpp
#            )
#    add_subdirectory(pybind11)
#    pybind11_add_module(EGTtools SHARED ${PYTHON_FILES})
#    target_link_libraries(EGTtools PRIVATE EGTTools)
#endif ()



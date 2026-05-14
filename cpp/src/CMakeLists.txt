# Copyright (c) 2021  Elias Fernandez
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
# Generate version.h from template (single source of truth: git tag / CMake project version)
# ------------------------------------------------------------------------------

configure_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/version.h.in"
        "${CMAKE_CURRENT_BINARY_DIR}/version.h"
        @ONLY
)

# ------------------------------------------------------------------------------
# Python extension module using pybind11
# ------------------------------------------------------------------------------

pybind11_add_module(numerical_
        egttools_pybind11_distributed.cpp
        pybind11_files/distributions.cpp
        pybind11_files/games.cpp
        pybind11_files/structure.cpp
        pybind11_files/behaviors.cpp
        pybind11_files/methods.cpp
        pybind11_files/random.cpp
        pybind11_files/datastructures.cpp
)

# ------------------------------------------------------------------------------
# Add main C++ sources
# ------------------------------------------------------------------------------

add_subdirectory(egttools)

# ------------------------------------------------------------------------------
# Includes and warnings
# ------------------------------------------------------------------------------

target_include_directories(numerical_ PRIVATE . "${CMAKE_CURRENT_BINARY_DIR}")

if (UNIX)
    target_compile_options(numerical_ PRIVATE
            -Wall
            -Wextra
            -Wno-unknown-pragmas
    )
endif ()

# ------------------------------------------------------------------------------
# Target properties (standard, LTO, visibility)
# ------------------------------------------------------------------------------

set_target_properties(numerical_ PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED TRUE
        CXX_EXTENSIONS FALSE
        INTERPROCEDURAL_OPTIMIZATION ${LTO_SUPPORTED}
        POSITION_INDEPENDENT_CODE TRUE
        # Fixing "ld: warning: direct access in function '...' from file '...' to global weak symbol '...' from file '...' means the weak symbol cannot
        # be overridden at runtime. This was likely caused by different translation units being compiled with different visibility settings."
        C_VISIBILITY_PRESET hidden
        CXX_VISIBILITY_PRESET hidden
)

# ------------------------------------------------------------------------------
# Boost
# ------------------------------------------------------------------------------

get_target_property(Boost_INCLUDE_DIRS Boost::multiprecision INTERFACE_INCLUDE_DIRECTORIES)
message(STATUS "Boost Include Directories: ${Boost_INCLUDE_DIRS}")
target_include_directories(numerical_ PRIVATE ${Boost_INCLUDE_DIRS})

if (APPLE)
    target_link_libraries(numerical_ PRIVATE Boost::multiprecision "-framework Accelerate")
else ()
    target_link_libraries(numerical_ PRIVATE Boost::multiprecision)
endif ()

target_compile_definitions(numerical_ PRIVATE -DHAS_BOOST)

# ------------------------------------------------------------------------------
# OpenMP
# ------------------------------------------------------------------------------

if (OpenMP_CXX_FOUND)
    message(STATUS "[OpenMP] Linking to target: numerical_")
    target_link_libraries(numerical_ PRIVATE OpenMP::OpenMP_CXX)
    if (NOT APPLE)
        #        SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
        #        SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS} -fopenmp")
        target_compile_options(numerical_ PRIVATE ${OpenMP_CXX_FLAGS})
    endif ()
else ()
    message(WARNING "[OpenMP] Not found or disabled for target: numerical_")
endif ()

# ------------------------------------------------------------------------------
# BLAS / LAPACK
# ------------------------------------------------------------------------------

egttools_enable_blas_lapack(numerical_)

# ------------------------------------------------------------------------------
# ARPACK (optional)
# ------------------------------------------------------------------------------

if (DEFINED EGTTOOLS_ARPACK_TARGET)
    target_link_libraries(numerical_ PRIVATE ${EGTTOOLS_ARPACK_TARGET})
    target_compile_definitions(numerical_ PRIVATE HAS_ARPACK=1)
    message(STATUS "[EGTtools] numerical_: linking ${EGTTOOLS_ARPACK_TARGET} (HAS_ARPACK=1)")
endif ()

# ------------------------------------------------------------------------------
# PETSc / SLEPc / MPI — optional numerical_mpi_ module
# ------------------------------------------------------------------------------

if (EGTTOOLS_ENABLE_PETSC)
    target_compile_definitions(numerical_ PRIVATE HAS_PETSC_MODULE=1)
    message(STATUS "[EGTtools] numerical_: HAS_PETSC_MODULE=1 (PETSc/SLEPc available)")

    # Compile PairwisePetscOperator into a static lib linked by numerical_mpi_.
    add_subdirectory(egttools/mpi)

    pybind11_add_module(numerical_mpi_
        egttools_pybind11_mpi.cpp
        pybind11_files/mpi_operators.cpp
    )

    target_include_directories(numerical_mpi_ PRIVATE
        .
        "${CMAKE_CURRENT_BINARY_DIR}"
        "${PROJECT_SOURCE_DIR}/cpp/include"
    )

    target_link_libraries(numerical_mpi_ PRIVATE
        egttools_mpi_core
        ${EGTTOOLS_PETSC_TARGET}
        ${EGTTOOLS_SLEPC_TARGET}
        MPI::MPI_CXX
        Eigen3::Eigen
    )

    set_target_properties(numerical_mpi_ PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED TRUE
        POSITION_INDEPENDENT_CODE TRUE
        C_VISIBILITY_PRESET hidden
        CXX_VISIBILITY_PRESET hidden
    )

    if (SKBUILD)
        install(TARGETS numerical_mpi_
                LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX})
    endif ()

    message(STATUS "[EGTtools] numerical_mpi_: PETSc/SLEPc MPI module enabled")
endif ()

# ------------------------------------------------------------------------------
# Create EGTToolsTargets export
# ------------------------------------------------------------------------------

if (NOT SKBUILD)
    install(TARGETS numerical_
            EXPORT EGTToolsTargets
            LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
            ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
            RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    )
endif ()
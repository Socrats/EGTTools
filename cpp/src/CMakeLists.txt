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

pybind11_add_module(numerical_
        egttools_pybind11_distributed.cpp
        pybind11_files/distributions.cpp
        pybind11_files/games.cpp
        pybind11_files/structure.cpp
        pybind11_files/behaviors.cpp
        pybind11_files/methods.cpp
        pybind11_files/datastructures.cpp)
add_subdirectory(egttools)

target_include_directories(numerical_ PRIVATE .)

if (UNIX)
    target_compile_options(numerical_ PRIVATE
            -Wall
            -Wextra
            -Wno-unknown-pragmas
            )
endif ()

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

target_include_directories(numerical_ SYSTEM INTERFACE ${CMAKE_SOURCE_DIR})

if (OPENMP_FOUND)
    message("-- linking OpenMP to target numerical_")
    # TODO: Make sure that having dynamic linking will work on Pypi
    target_link_libraries(numerical_ PRIVATE OpenMP::OpenMP_CXX)
else ()
    target_link_libraries(numerical_)
endif ()
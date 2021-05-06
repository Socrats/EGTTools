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

add_custom_target(pytest
        COMMAND ${CMAKE_COMMAND} -E env PYTHONPATH=$<TARGET_FILE_DIR:numerical> ${Python_EXECUTABLE} -m pytest
        DEPENDS numerical
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
add_custom_target(tests DEPENDS pytest)

add_library(egttoolsNumerical
        ../src/egttools/SeedGenerator.cpp
        ../src/egttools/Distributions.cpp
        ../src/egttools/Data.cpp
        ../src/egttools/finite_populations/Utils.cpp
        ../src/egttools/finite_populations/games/NormalFormGame.cpp
        ../src/egttools/finite_populations/behaviors/NFGStrategies.cpp
        )

set_target_properties(egttoolsNumerical PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED TRUE
        CXX_EXTENSIONS FALSE
#        INTERPROCEDURAL_OPTIMIZATION ${LTO_SUPPORTED}
#        POSITION_INDEPENDENT_CODE TRUE
        # Fixing "ld: warning: direct access in function '...' from file '...' to global weak symbol '...' from file '...' means the weak symbol cannot
        # be overridden at runtime. This was likely caused by different translation units being compiled with different visibility settings."
        C_VISIBILITY_PRESET hidden
#        CXX_VISIBILITY_PRESET hidden
        )

if (OPENMP_FOUND)
    target_link_libraries(egttoolsNumerical OpenMP::OpenMP_CXX)
endif ()
target_link_libraries(egttoolsNumerical)

add_executable(testPMrun test_PairwiseMoran_run.cpp)
#set_target_properties(testPM PROPERTIES INTERPROCEDURAL_OPTIMIZATION ${LTO_SUPPORTED})
target_link_libraries(testPMrun egttoolsNumerical)

add_executable(testPMBstationary_distribution test_PairwiseMoran_stationary_distribution.cpp)
#set_target_properties(testPM PROPERTIES INTERPROCEDURAL_OPTIMIZATION ${LTO_SUPPORTED})
target_link_libraries(testPMBstationary_distribution egttoolsNumerical)
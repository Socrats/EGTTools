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

add_custom_target(docs
        COMMAND ${CMAKE_COMMAND} -E env PYTHONPATH=${CMAKE_INSTALL_PREFIX} ${Python_EXECUTABLE} -m sphinx -M html ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR}
        BYPRODUCTS doctrees/ html/
        DEPENDS numerical
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

add_custom_target(docs_linkcheck
        COMMAND ${CMAKE_COMMAND} -E env PYTHONPATH=${CMAKE_INSTALL_PREFIX} ${Python_EXECUTABLE} -m sphinx -M linkcheck ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR}
        BYPRODUCTS linkcheck/
        DEPENDS docs
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
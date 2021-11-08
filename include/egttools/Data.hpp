/** Copyright (c) 2019-2021  Elias Fernandez
  *
  * This file is part of EGTtools.
  *
  * EGTtools is free software: you can redistribute it and/or modify
  * it under the terms of the GNU General Public License as published by
  * the Free Software Foundation, either version 3 of the License, or
  * (at your option) any later version.
  *
  * EGTtools is distributed in the hope that it will be useful,
  * but WITHOUT ANY WARRANTY; without even the implied warranty of
  * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  * GNU General Public License for more details.
  *
  * You should have received a copy of the GNU General Public License
  * along with EGTtools.  If not, see <http://www.gnu.org/licenses/>
*/
#pragma once
#ifndef EGTTOOLS_DATA_HPP
#define EGTTOOLS_DATA_HPP

#include <egttools/Types.h>

#include <utility>
#include <vector>

namespace egttools::DataStructures {
    struct DataTable {
        size_t nb_rows, nb_columns;
        std::vector<std::string> header;
        std::vector<std::string> column_types;
        Matrix2D data;
        /* TODO: either create a simplified pandas-like interface for this structure
         * or try to get this structure transformed into a pandas dataframe in Python.  */
        DataTable(size_t nb_rows,
                  size_t nb_columns,
                  std::vector<std::string> &headers,
                  std::vector<std::string> &column_types);
    };
}// namespace egttools::DataStructures

#endif//EGTTOOLS_DATA_HPP

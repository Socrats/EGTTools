/** Copyright (c) 2020-2021  Elias Fernandez
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

#include <egttools/Data.hpp>

egttools::DataStructures::DataTable::DataTable(size_t nb_rows, size_t nb_columns,
                                               std::vector<std::string> &headers, std::vector<std::string> &column_types)
    : nb_rows(nb_rows),
      nb_columns(nb_columns),
      header(std::move(headers)),
      column_types(std::move(column_types)) {
    data = Matrix2D::Zero(static_cast<long>(nb_rows), static_cast<long>(nb_columns));
}
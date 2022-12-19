/** Copyright (c) 2022-2023  Elias Fernandez
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
#include "datastructures.hpp"

void init_datastructures(py::module_ &mData) {
    mData.attr("__init__") = py::str("The `egttools.numerical.DataStructures` submodule contains helpful data structures.");

    py::class_<egttools::DataStructures::DataTable>(mData, "DataTable")
            .def(py::init<size_t,
                          size_t,
                          std::vector<std::string> &,
                          std::vector<std::string> &>(),
                 "Data structure that allows to store information in table format. Headers give the ",
                 py::arg("nb_rows"),
                 py::arg("nb_columns"),
                 py::arg("headers"),
                 py::arg("column_types"))
            .def_readonly("rows", &egttools::DataStructures::DataTable::nb_rows, "returns the number of rows")
            .def_readonly("cols", &egttools::DataStructures::DataTable::nb_columns, "returns the number of columns")
            .def_readwrite("data", &egttools::DataStructures::DataTable::data, py::return_value_policy::reference_internal)
            .def_readwrite("headers", &egttools::DataStructures::DataTable::header,
                           py::return_value_policy::reference_internal)
            .def_readwrite("column_types", &egttools::DataStructures::DataTable::column_types,
                           py::return_value_policy::reference_internal);
}
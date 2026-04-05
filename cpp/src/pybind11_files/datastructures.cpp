/** Copyright (c) 2022-2026  Elias Fernandez
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
    mData.doc() =
            "The `egttools.numerical.DataStructures` submodule contains helper data structures.";

    py::class_<egttools::DataStructures::DataTable>(
                mData,
                "DataTable",
                R"pbdoc(
            Store tabular data together with column headers and column types.
        )pbdoc"
            )
            .def(
                py::init<
                    size_t,
                    size_t,
                    std::vector<std::string> &,
                    std::vector<std::string> &>(),
                R"pbdoc(
                Create a data table.

                Parameters
                ----------
                nb_rows : int
                    Number of rows in the table.
                nb_columns : int
                    Number of columns in the table.
                headers : list[str]
                    Column names.
                column_types : list[str]
                    Type label for each column.

                Examples
                --------
                >>> from egttools.numerical.DataStructures import DataTable
                >>> table = DataTable(
                ...     2,
                ...     2,
                ...     ["name", "value"],
                ...     ["str", "float"]
                ... )
            )pbdoc",
                py::arg("nb_rows"),
                py::arg("nb_columns"),
                py::arg("headers"),
                py::arg("column_types")
            )
            .def_readonly(
                "rows",
                &egttools::DataStructures::DataTable::nb_rows,
                "Number of rows in the table."
            )
            .def_readonly(
                "cols",
                &egttools::DataStructures::DataTable::nb_columns,
                "Number of columns in the table."
            )
            .def_readwrite(
                "data",
                &egttools::DataStructures::DataTable::data,
                py::return_value_policy::reference_internal,
                R"pbdoc(
                Table contents.

                Returns
                -------
                numpy.ndarray
                    Two-dimensional array storing the table values.
            )pbdoc"
            )
            .def_readwrite(
                "headers",
                &egttools::DataStructures::DataTable::header,
                py::return_value_policy::reference_internal,
                R"pbdoc(
                Column names.

                Returns
                -------
                list[str]
                    Header for each column.
            )pbdoc"
            )
            .def_readwrite(
                "column_types",
                &egttools::DataStructures::DataTable::column_types,
                py::return_value_policy::reference_internal,
                R"pbdoc(
                Column type labels.

                Returns
                -------
                list[str]
                    Type label for each column.
            )pbdoc"
            );
}

//
// Created by Elias Fernandez on 03/03/2021.
//
#include <egttools/Types.h>

#include <egttools/Data.hpp>
#include <utility>

egttools::DataStructures::DataTable::DataTable(size_t nb_rows, size_t nb_columns,
                                               std::vector<std::string> &headers, std::vector<std::string> &column_types)
    : nb_rows(nb_rows),
      nb_columns(nb_columns),
      header(std::move(headers)),
      column_types(std::move(column_types)) {
    data = Matrix2D::Zero(nb_rows, nb_columns);
}
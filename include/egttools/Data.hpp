//
// Created by Elias Fernandez on 03/03/2021.
//

#ifndef EGTTOOLS_DATA_HPP
#define EGTTOOLS_DATA_HPP

namespace egttools { namespace DataStructures {
    struct DataTable {
        size_t nb_rows, nb_columns;
        std::vector<std::string> header;
        std::vector<std::string> column_types;
        Matrix2D data;

        DataTable(size_t nb_rows,
                  size_t nb_columns,
                  std::vector<std::string> &headers,
                  std::vector<std::string> &column_types);
    };
} }

#endif//EGTTOOLS_DATA_HPP

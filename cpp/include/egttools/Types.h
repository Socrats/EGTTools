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

//
// Adapted from https://github.com/Svalorzen/AI-Toolbox/
//
#pragma once
#ifndef EGTTOOLS_TYPES_H
#define EGTTOOLS_TYPES_H

#include <vector>
#include <unordered_map>
#include <random>
#include <Eigen/Core>
#include <Eigen/SparseCore>

//#undef EIGEN_DEFAULT_DENSE_INDEX_TYPE
//#define EIGEN_DEFAULT_DENSE_INDEX_TYPE signed long

namespace egttools {
    // This should have decent properties.
    using RandomEngine = std::mt19937_64;

    using Factors = std::vector<size_t>;

    using Vector = Eigen::Matrix<double, Eigen::Dynamic, 1>;
    using VectorXi = Eigen::Matrix<int, Eigen::Dynamic, 1>;
    using VectorXli = Eigen::Matrix<int_fast64_t, Eigen::Dynamic, 1>;
    using VectorXui = Eigen::Matrix<size_t, Eigen::Dynamic, 1>;
    using SparseVectorXi = Eigen::SparseVector<int, Eigen::RowMajor>;
    using SparseVectorXui =  Eigen::SparseVector<size_t, Eigen::RowMajor>;
    using SparseVector =  Eigen::SparseVector<double, Eigen::RowMajor>;
    using Vector2d = Eigen::Matrix<double, 2, 1>;
    using Vector3d = Eigen::Matrix<double, 3, 1>;

    using MatrixXd       = Eigen::MatrixXd;
    using Matrix2D       = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor | Eigen::AutoAlign>;
    using SparseMatrix2D = Eigen::SparseMatrix<double, Eigen::RowMajor, signed long>;
    using SparseMatrix2DXi = Eigen::SparseMatrix<long, Eigen::RowMajor, signed long>;
    using SparseMatrix2DXui = Eigen::SparseMatrix<size_t, Eigen::RowMajor, signed long>;
    using MatrixXui2D = Eigen::Matrix<size_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor | Eigen::AutoAlign>;
    using MatrixXl2D = Eigen::Matrix<int_fast64_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor | Eigen::AutoAlign>;

    using Matrix3D       = std::vector<Matrix2D>;
    using SparseMatrix3D = std::vector<SparseMatrix2D>;

//    using Matrix4D       = boost::multi_array<Matrix2D, 2>;
//    using SparseMatrix4D = boost::multi_array<SparseMatrix2D, 2>;

    using Table2D = Eigen::Matrix<long, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor | Eigen::AutoAlign>;
    using Table3D = std::vector<Table2D>;

    using SparseTable2D = Eigen::SparseMatrix<long, Eigen::RowMajor>;
    using SparseTable3D = std::vector<SparseTable2D>;

    // This is used to store a probability vector (sums to one, every element >= 0, <= 1)
    using ProbabilityVector = Vector;

//    using DumbMatrix2D = boost::multi_array<double, 2>;
//    using DumbMatrix3D = boost::multi_array<double, 3>;
//    using DumbTable2D  = boost::multi_array<long, 2>;
//    using DumbTable3D  = boost::multi_array<long, 3>;

    /**
     * @brief This is used to tag functions that avoid runtime checks.
     */
    inline struct NoCheck {
    } NO_CHECK;
}

#endif //EGTTOOLS_TYPES_H

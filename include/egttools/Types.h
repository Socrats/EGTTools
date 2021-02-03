//
// Adapted from https://github.com/Svalorzen/AI-Toolbox/
//

#ifndef EGTTOOLS_TYPES_H
#define EGTTOOLS_TYPES_H

#include <vector>
#include <unordered_map>
#include <random>
#include <Eigen/Core>
#include <Eigen/SparseCore>

namespace egttools {
    // This should have decent properties.
    using RandomEngine = std::mt19937;

    using Factors = std::vector<size_t>;

    using Vector = Eigen::Matrix<double, Eigen::Dynamic, 1>;
    using VectorXi = Eigen::Matrix<int, Eigen::Dynamic, 1>;
    using VectorXui = Eigen::Matrix<size_t, Eigen::Dynamic, 1>;
    using Vector2d = Eigen::Matrix<double, 2, 1>;
    using Vector3d = Eigen::Matrix<double, 3, 1>;

    using MatrixXd       = Eigen::MatrixXd;
    using Matrix2D       = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor | Eigen::AutoAlign>;
    using SparseMatrix2D = Eigen::SparseMatrix<double, Eigen::RowMajor>;
    using MatrixXui2D = Eigen::Matrix<size_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor | Eigen::AutoAlign>;

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

#endif //DYRWIN_TYPES_H

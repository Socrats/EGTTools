#define EIGEN_USE_THREADS
#define EIGEN_USE_OPENMP
#include <Eigen/Sparse>
#include <Eigen/SparseLU>
#include <Eigen/IterativeLinearSolvers>
#include <random>
#include <iostream>
#include <iomanip> // for std::setprecision
#include <chrono>
#include <cmath>

using namespace Eigen;
using SparseMatrix2D = SparseMatrix<double, RowMajor>;
using Vector1d = Matrix<double, Dynamic, 1>;
using Matrix2D = Matrix<double, Dynamic, Dynamic, RowMajor | AutoAlign>;

Vector1d calculateStationaryDistributionSparse(SparseMatrix2D &transitionMatrix) {
    const auto n = transitionMatrix.rows();

    // Validate input
    if (transitionMatrix.cols() != n) {
        throw std::invalid_argument("Transition matrix must be square.");
    }

    // Normalize rows to sum to 1
    for (int k = 0; k < transitionMatrix.outerSize(); ++k) {
        long double rowSum = 0.0;
        for (SparseMatrix2D::InnerIterator it(transitionMatrix, k); it; ++it) {
            rowSum += it.value();
        }
        if (rowSum > 0) {
            for (SparseMatrix2D::InnerIterator it(transitionMatrix, k); it; ++it) {
                it.valueRef() /= rowSum;
            }
        }
    }

    // Create left-hand side: A = P^T - I
    SparseMatrix2D A = transitionMatrix.transpose();
    A -= SparseMatrix2D(Matrix2D::Identity(n, n).sparseView());

    // Replace last row with the constraint: sum(pi) = 1
    for (int j = 0; j < n; ++j) {
        A.coeffRef(n - 1, j) = 1.0;
    }

    // Right-hand side vector
    Vector1d b = Vector1d::Zero(n);
    b(n - 1) = 1.0; // Corresponding to the sum constraint

    // Solve with BiCGSTAB (robust for sparse systems)
    BiCGSTAB<SparseMatrix2D> solver;
    solver.compute(A);

    if (solver.info() != Success) {
        throw std::runtime_error("Decomposition failed.");
    }

    Vector1d pi = solver.solve(b);

    if (solver.info() != Success) {
        throw std::runtime_error("Solving failed.");
    }

    return pi;
}

Vector1d calculateStationaryDistributionSparseOptimised(const SparseMatrix2D& transitionMatrix) {
    const auto n = transitionMatrix.rows();

    // Validate that it's a square matrix
    if (transitionMatrix.cols() != n) {
        throw std::invalid_argument("Transition matrix must be square.");
    }

    // Create the left-hand side matrix: A = P^T - I
    SparseMatrix2D A = transitionMatrix.transpose();
    A -= SparseMatrix2D(MatrixXd::Identity(n, n).sparseView());

    // Add the constraint row: sum(pi) = 1
    SparseMatrix2D augmentedA(n, n);
    augmentedA = A;
    augmentedA.row(n - 1) = SparseMatrix2D(VectorXd::Ones(n).transpose().sparseView());

    // Right-hand side vector
    VectorXd b = VectorXd::Zero(n);
    b(n - 1) = 1.0;  // Constraint: sum(pi) = 1

    // Solve the sparse linear system using SparseLU
    SparseLU<SparseMatrix2D> solver;
    solver.compute(augmentedA);

    if (solver.info() != Success) {
        throw std::runtime_error("Decomposition failed.");
    }

    Vector1d pi = solver.solve(b);

    if (solver.info() != Success) {
        throw std::runtime_error("Solving failed.");
    }

    return pi;
}

SparseMatrix2D generateLargeSparseTransitionMatrix(const int n, const double sparsity = 0.01) {
    SparseMatrix2D matrix(n, n);

    std::default_random_engine generator;
    std::uniform_real_distribution<double> valueDist(0.1, 1.0); // Random values [0.1, 1.0)
    std::uniform_real_distribution<double> sparseDist(0.0, 1.0); // Sparsity control

    std::vector<Triplet<double> > triplets;
    triplets.reserve(ceil(n * sparsity));

    for (int i = 0; i < n; ++i) {
        double rowSum = 0.0;
        std::vector<int> nonZeroIndices;

        // Generate random sparse row
        for (int j = 0; j < n; ++j) {
            if (sparseDist(generator) < sparsity) {
                double value = valueDist(generator);
                triplets.emplace_back(i, j, value);
                rowSum += value;
            }
        }

        // Ensure the row has at least one non-zero entry
        if (rowSum == 0) {
            double value = valueDist(generator);
            triplets.emplace_back(i, i, value);
            rowSum += value;
        }

        // Normalize the row
        for (auto &t: triplets) {
            if (t.row() == i) {
                t = Triplet<double>(t.row(), t.col(), t.value() / rowSum);
            }
        }
    }

    matrix.setFromTriplets(triplets.begin(), triplets.end());
    return matrix;
}

// Function to verify the stationary distribution
Vector1d computeStationaryDistributionDirect(const SparseMatrix2D &transitionMatrix) {
    const auto n = transitionMatrix.rows();
    VectorXd pi = VectorXd::Ones(n) / n; // Uniform initial guess
    constexpr int iterations = 10000; // Number of iterations

    for (int iter = 0; iter < iterations; ++iter) {
        VectorXd nextPi = transitionMatrix.transpose() * pi;
        if (constexpr double tolerance = 1e-17; (nextPi - pi).norm() < tolerance) {
            return nextPi;
        }
        pi = nextPi;
    }
    throw std::runtime_error("Failed to converge to the stationary distribution.");
}

int main() {
    std::cout << nbThreads() << " threads available.\n";
    initParallel();
    // Example sparse transition matrix
    constexpr int n = 5000;
    constexpr double sparsity = 0.001; // Fraction of non-zero entries

    // Generate large sparse transition matrix
    SparseMatrix2D transitionMatrix = generateLargeSparseTransitionMatrix(n, sparsity);

    std::cout << "Generated transition matrix with " << transitionMatrix.nonZeros() << " non-zero entries.\n";

    // Increase precision
    std::cout << std::setprecision(10);

    // Compute the stationary distribution
    try {
        auto start = std::chrono::high_resolution_clock::now();
        VectorXd stationaryDistributionDirect = computeStationaryDistributionDirect(transitionMatrix);
        auto end = std::chrono::high_resolution_clock::now();
        std::cout << "Stationary Distribution (first 10 values):\n"
                << stationaryDistributionDirect.head(10).transpose() << std::endl;
        std::cout << "Elapsed time: "
                << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
                << " ms\n";
    } catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }

    try {
        auto start = std::chrono::high_resolution_clock::now();
        Vector1d stationaryDistribution = calculateStationaryDistributionSparse(transitionMatrix);
        auto end = std::chrono::high_resolution_clock::now();
        std::cout << "Stationary Distribution:\n" << stationaryDistribution.head(10).transpose() << std::endl;
        std::cout << "Elapsed time: "
                << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
                << " ms\n";
    } catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }

    return 0;
}

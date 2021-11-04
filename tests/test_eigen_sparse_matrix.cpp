//
// Created by Elias Fernandez on 06/05/2021.
//
#include <egttools/Types.h>

#include <chrono>
#include <iostream>
using namespace std::chrono;

int main() {
//    auto start = high_resolution_clock::now();
//    int nrows = 10;
//    int ncols = 15;
//    egttools::SparseMatrix2DXui mat(nrows, ncols);
////    Eigen::SparseMatrix<double> mat(nrows,ncols);
//    mat.makeCompressed();
//    auto values =  egttools::VectorXi::Constant(ncols, 100);
//    mat.reserve(values);
//
//    for (int i = 0; i < nrows; i++) {
//        for (int j = 0; j < 1000000; j++) {
//            mat.coeffRef(i, i) += 1;
//        }
//    }
//
//    nrows = 10;
//    ncols = 1;
//    egttools::SparseMatrix2DXui mat2(nrows, ncols);
//////    Eigen::SparseMatrix<double> mat(nrows,ncols);
////    mat.makeCompressed();
//    auto values2 =  egttools::VectorXi::Constant(ncols, 3);
//    mat.reserve(values2);
//
//    mat2.coeffRef(5, 0) += 1;
//    mat2.coeffRef(0, 0) += 1;
//    mat2.coeffRef(2, 0) += 1;
//
//    auto stop = high_resolution_clock::now();
//    auto duration = duration_cast<microseconds>(stop - start);
//    std::cout << "Time taken by function: "
//              << duration.count() << " microseconds" << std::endl;
//
//    std::cout << mat << std::endl;
//    std::cout << mat2 << std::endl;

    egttools::SparseMatrix2D Test(2, 3);
    Test.insert(0, 1) = 34;
    Test.insert(1, 2) = 56;
    for (int k = 0; k < Test.outerSize(); ++k){
        for (egttools::SparseMatrix2D::InnerIterator it(Test, k); it; ++it){
            std::cout << it.row() <<"\t";
            std::cout << it.col() << "\t";
            std::cout << it.value() << std::endl;
        }
    }
}
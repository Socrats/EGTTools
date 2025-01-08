//
// Created by Elias Fernandez on 08/01/2025.
//
#include <iostream>
#include <Accelerate/Accelerate.h>

int main() {
    int n = 2;
    double A[] = {1.0, 2.0, 3.0, 4.0};
    double B[] = {5.0, 6.0};
    double C[2];
    cblas_dgemv(CblasRowMajor, CblasNoTrans, n, n, 1.0, A, n, B, 1, 0.0, C, 1);

    std::cout << "Result: " << C[0] << ", " << C[1] << std::endl;
    return 0;
}

// macOS only: checks that the BLAS from Accelerate, which egttools_core links on macOS, is
// linked and gives the right result.
#include <Accelerate/Accelerate.h>

#include <catch2/catch_test_macros.hpp>

TEST_CASE("Accelerate cblas_dgemv computes a matrix-vector product", "[build][blas]") {
    // [[1, 2], [3, 4]] * [5, 6] = [17, 39], exact in floating point.
    const double A[] = {1.0, 2.0, 3.0, 4.0};
    const double x[] = {5.0, 6.0};
    double y[] = {0.0, 0.0};
    cblas_dgemv(CblasRowMajor, CblasNoTrans, 2, 2, 1.0, A, 2, x, 1, 0.0, y, 1);
    CHECK(y[0] == 17.0);
    CHECK(y[1] == 39.0);
}

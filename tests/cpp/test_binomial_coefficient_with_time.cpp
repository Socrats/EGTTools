//
// Created by Elias Fernandez on 07/11/2022.
//
// Checks egttools::binomial_precision and egttools::multivariateHypergeometricPDF
// against exact values, and reports how long the hypergeometric PDF takes.
// Reference values were computed with Python's math.comb (exact integers).
#include <cassert>
#include <cmath>
#include <ctime>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

#include <egttools/Distributions.h>
#include <egttools/Types.h>

#include <boost/multiprecision/cpp_int.hpp>

using boost::multiprecision::uint128_t;

namespace {
    bool close(const double value, const double reference, const double rel_tol = 1e-9) {
        return std::abs(value - reference) <= rel_tol * std::abs(reference);
    }
}// namespace

int main() {
    // binomial_precision: exact while C(n, k) fits in 128 bits. The old implementation
    // overflowed in the intermediate product from n = 126 onwards, even when the
    // result fits (e.g. C(126, 63) and C(131, 65)).
    assert(egttools::binomial_precision(50, 10) == uint128_t("10272278170"));
    assert(egttools::binomial_precision(490, 481) == uint128_t("4167813029162990130"));
    assert(egttools::binomial_precision(126, 63) == uint128_t("6034934435761406706427864636568328000"));
    assert(egttools::binomial_precision(131, 65) == uint128_t("188694833082770476622296176145946360850"));
    assert(egttools::binomial_precision(10, 11) == 0);

    // C(150, 100) ~ 2.0e40 does not fit in 128 bits: it must throw, not wrap around.
    bool threw = false;
    try {
        static_cast<void>(egttools::binomial_precision(150, 100));
    } catch (const std::overflow_error &) {
        threw = true;
    }
    assert(threw);

    // A negative int passed by mistake arrives as SIZE_MAX: no wrap-around, and a large k
    // must throw quickly instead of looping ~2^63 times.
    constexpr size_t size_max = std::numeric_limits<size_t>::max();
    assert(egttools::binomial_precision(size_max, 1) == uint128_t(size_max));
    threw = false;
    try {
        static_cast<void>(egttools::binomial_precision(size_max, size_max / 2));
    } catch (const std::overflow_error &) {
        threw = true;
    }
    assert(threw);

    // Multivariate hypergeometric PDF: population of 499 with counts {9, 490}, groups of 9.
    // P({7, 2}) = C(9, 7) C(490, 2) / C(499, 9), P({5, 4}) = C(9, 5) C(490, 4) / C(499, 9).
    const std::vector<size_t> population_counts{9, 490};
    const std::vector<size_t> group_counts1{7, 2};
    const std::vector<size_t> group_counts2{5, 4};

    const auto start = std::clock();
    const double p1 = egttools::multivariateHypergeometricPDF(499, 2, 9, group_counts1, population_counts);
    const auto end = std::clock();
    const double p2 = egttools::multivariateHypergeometricPDF(499, 2, 9, group_counts2, population_counts);

    std::cout << "P({7, 2}) = " << p1 << ", P({5, 4}) = " << p2 << std::endl;
    std::cout << "Elapsed time for multivariateHypergeometricPDF = "
              << static_cast<double>(end - start) / CLOCKS_PER_SEC << " s" << std::endl;

    assert(close(p1, 8.772974388576907e-13));
    assert(close(p2, 6.08110417043393e-08));

    std::cout << "binomial_precision and multivariateHypergeometricPDF tests passed.\n";
    return 0;
}

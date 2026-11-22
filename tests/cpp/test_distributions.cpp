// Binomial coefficients, state counts and the multivariate hypergeometric PDF (Distributions.h).
//
// References: Pascal's triangle built by addition (a recurrence independent of the multiplicative
// one under test), and exact values computed with Python's math.comb and fractions.Fraction.
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <boost/multiprecision/cpp_int.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <egttools/Distributions.h>

using boost::multiprecision::uint128_t;
using Catch::Matchers::WithinRel;

namespace {
    struct Entry {
        std::uint64_t value;
        bool too_big;// exceeds 2^64 - 1
    };

    std::vector<std::vector<Entry>> pascal_triangle(const std::uint64_t max_n) {
        constexpr auto u64_max = std::numeric_limits<std::uint64_t>::max();
        std::vector<std::vector<Entry>> pascal(max_n + 1);
        for (std::uint64_t n = 0; n <= max_n; ++n) {
            pascal[n].resize(n + 1);
            pascal[n][0] = pascal[n][n] = {1, false};
            for (std::uint64_t k = 1; k < n; ++k) {
                const auto &a = pascal[n - 1][k - 1];
                const auto &b = pascal[n - 1][k];
                const bool too_big = a.too_big || b.too_big || a.value > u64_max - b.value;
                pascal[n][k] = {too_big ? 0 : a.value + b.value, too_big};
            }
        }
        return pascal;
    }

    template<typename Output>
    bool throws_overflow(std::uint64_t n, std::uint64_t k) {
        try {
            static_cast<void>(egttools::binomialCoeff<Output, std::uint64_t>(n, k));
        } catch (const std::overflow_error &) {
            return true;
        }
        return false;
    }

    // For every n <= 80: the exact value when it fits in Output, std::overflow_error otherwise,
    // and 0 for k > n. Returns the number of mismatches, so a failure reports how many.
    template<typename Output>
    int mismatches_against_pascal(const std::vector<std::vector<Entry>> &pascal) {
        const auto max_value = static_cast<std::uint64_t>(std::numeric_limits<Output>::max());
        int mismatches = 0;
        for (std::uint64_t n = 0; n < pascal.size(); ++n) {
            for (std::uint64_t k = 0; k <= n + 1; ++k) {
                if (k > n) {
                    mismatches += egttools::binomialCoeff<Output, std::uint64_t>(n, k) != 0;
                    continue;
                }
                const auto &entry = pascal[n][k];
                if (entry.too_big || entry.value > max_value)
                    mismatches += !throws_overflow<Output>(n, k);
                else
                    mismatches += static_cast<std::uint64_t>(egttools::binomialCoeff<Output, std::uint64_t>(n, k)) !=
                                  entry.value;
            }
        }
        return mismatches;
    }
}// namespace

TEST_CASE("binomialCoeff is exact or throws for 64-bit outputs", "[distributions][binomial]") {
    const auto pascal = pascal_triangle(80);
    CHECK(mismatches_against_pascal<std::int64_t>(pascal) == 0);
    CHECK(mismatches_against_pascal<std::uint64_t>(pascal) == 0);
}

TEST_CASE("binomialCoeff at the 64-bit boundaries", "[distributions][binomial]") {
    // C(66, 33) is the largest central coefficient that fits in int64_t, C(67, 33) fits only in
    // uint64_t, C(68, 34) fits in neither.
    CHECK(egttools::binomialCoeff<std::int64_t, std::uint64_t>(66, 33) == 7219428434016265740LL);
    CHECK(throws_overflow<std::int64_t>(67, 33));
    CHECK(egttools::binomialCoeff<std::uint64_t, std::uint64_t>(67, 33) == 14226520737620288370ULL);
    CHECK(throws_overflow<std::uint64_t>(68, 34));
}

TEST_CASE("binomialCoeff handles huge and negative arguments", "[distributions][binomial]") {
    constexpr auto u64_max = std::numeric_limits<std::uint64_t>::max();
    // Huge n (e.g. a negative int converted to size_t): exact for k = 1, a fast throw otherwise.
    CHECK(egttools::binomialCoeff<std::uint64_t, std::uint64_t>(u64_max, 1) == u64_max);
    CHECK(throws_overflow<std::int64_t>(u64_max, 1));
    CHECK(throws_overflow<std::uint64_t>(u64_max, u64_max / 2));
    // Negative arguments give 0.
    CHECK(egttools::binomialCoeff<std::int64_t, int>(-1, 0) == 0);
    CHECK(egttools::binomialCoeff<std::int64_t, int>(5, -1) == 0);
}

TEST_CASE("binomialCoeff with a 32-bit output is exact or throws", "[distributions][binomial]") {
    // C(141, 5) computed in int broke PairwiseComparison before the overflow checks.
    CHECK(egttools::binomialCoeff<int, int>(141, 5) == 432295143);
    CHECK_THROWS_AS((egttools::binomialCoeff<int, int>(128, 28)), std::overflow_error);
}

TEST_CASE("binomialCoeff with a floating-point output", "[distributions][binomial]") {
    CHECK_THAT((egttools::binomialCoeff<double, std::int64_t>(10, 3)), WithinRel(120.0, 1e-15));
}

TEST_CASE("checked_nb_states counts the states C(Z + S - 1, S - 1)", "[distributions][binomial]") {
    // Values that were wrong when computed in int.
    CHECK(egttools::checked_nb_states(136, 6) == 432295143);
    CHECK(egttools::checked_nb_states(52, 8) == 341149446);
    CHECK(egttools::checked_nb_states(31, 10) == 273438880);
    CHECK(egttools::checked_nb_states(0, 3) == 1);
    // A population whose number of states does not fit in int64_t: the message names Z and S.
    CHECK_THROWS_WITH(egttools::checked_nb_states(100, 29), Catch::Matchers::ContainsSubstring("Z = 100 and S = 29"));
}

TEST_CASE("binomial_precision is exact while C(n, k) fits in 128 bits", "[distributions][binomial]") {
    CHECK(egttools::binomial_precision(50, 10) == uint128_t("10272278170"));
    CHECK(egttools::binomial_precision(490, 481) == uint128_t("4167813029162990130"));
    // The former implementation overflowed in the intermediate product from n = 126 onwards,
    // even when the result fits.
    CHECK(egttools::binomial_precision(126, 63) == uint128_t("6034934435761406706427864636568328000"));
    CHECK(egttools::binomial_precision(131, 65) == uint128_t("188694833082770476622296176145946360850"));
    CHECK(egttools::binomial_precision(10, 11) == 0);
}

TEST_CASE("binomial_precision throws when C(n, k) does not fit in 128 bits", "[distributions][binomial]") {
    // C(150, 100) ~ 2.0e40 does not fit: it must throw, not wrap around.
    CHECK_THROWS_AS(egttools::binomial_precision(150, 100), std::overflow_error);
    // A negative int passed by mistake arrives as SIZE_MAX: no wrap-around for k = 1, and a large
    // k must throw quickly instead of looping ~2^63 times.
    constexpr size_t size_max = std::numeric_limits<size_t>::max();
    CHECK(egttools::binomial_precision(size_max, 1) == uint128_t(size_max));
    CHECK_THROWS_AS(egttools::binomial_precision(size_max, size_max / 2), std::overflow_error);
}

TEST_CASE("multivariateHypergeometricPDF matches exact values", "[distributions][hypergeometric]") {
    // P(sample) = prod_i C(K_i, k_i) / C(M, n), evaluated exactly with fractions.Fraction.
    const std::vector<size_t> population{9, 490};
    // C(9, 7) C(490, 2) / C(499, 9)
    CHECK_THAT(egttools::multivariateHypergeometricPDF(499, 2, 9, std::vector<size_t>{7, 2}, population),
               WithinRel(8.772974388576907e-13, 1e-9));
    // C(9, 5) C(490, 4) / C(499, 9)
    CHECK_THAT(egttools::multivariateHypergeometricPDF(499, 2, 9, std::vector<size_t>{5, 4}, population),
               WithinRel(6.08110417043393e-08, 1e-9));
    // C(498, 10) C(2, 0) / C(500, 10)
    CHECK_THAT(egttools::multivariateHypergeometricPDF(500, 2, 10, std::vector<size_t>{10, 0},
                                                       std::vector<size_t>{498, 2}),
               WithinRel(0.9603607214428858, 1e-12));
    // Three types: C(4, 1) C(3, 1) C(5, 2) / C(12, 4) = 120 / 495
    CHECK_THAT(egttools::multivariateHypergeometricPDF(12, 3, 4, std::vector<size_t>{1, 1, 2},
                                                       std::vector<size_t>{4, 3, 5}),
               WithinRel(120.0 / 495.0, 1e-12));
}

TEST_CASE("multivariateHypergeometricPDF sums to one over all samples", "[distributions][hypergeometric]") {
    // Population {4, 3, 5} (M = 12), samples of n = 4: every composition (k0, k1, k2).
    const std::vector<size_t> population{4, 3, 5};
    double total = 0.0;
    for (size_t k0 = 0; k0 <= 4; ++k0)
        for (size_t k1 = 0; k1 <= 4 - k0; ++k1) {
            const size_t k2 = 4 - k0 - k1;
            if (k0 > population[0] || k1 > population[1] || k2 > population[2]) continue;
            total += egttools::multivariateHypergeometricPDF(12, 3, 4, std::vector<size_t>{k0, k1, k2}, population);
        }
    CHECK_THAT(total, WithinRel(1.0, 1e-12));
}

// Tests the integer path of egttools::binomialCoeff (exact or std::overflow_error) and
// egttools::checked_nb_states.
//
// Reference: Pascal's triangle built by addition (an independent recurrence from the
// multiplicative one under test), with an "exceeds 2^64 - 1" flag per entry. For every
// n <= 80 and both int64_t and uint64_t outputs, binomialCoeff must return the exact value
// when it fits and throw std::overflow_error when it does not.
#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <egttools/Distributions.h>

namespace {
    struct Entry {
        std::uint64_t value;
        bool too_big;// exceeds 2^64 - 1
    };

    template<typename Output>
    bool throws_overflow(std::uint64_t n, std::uint64_t k) {
        try {
            static_cast<void>(egttools::binomialCoeff<Output, std::uint64_t>(n, k));
        } catch (const std::overflow_error &) {
            return true;
        }
        return false;
    }

    template<typename Output>
    void check_against_pascal(const std::vector<std::vector<Entry>> &pascal) {
        const auto max_value = static_cast<std::uint64_t>(std::numeric_limits<Output>::max());
        for (std::uint64_t n = 0; n < pascal.size(); ++n) {
            for (std::uint64_t k = 0; k <= n + 1; ++k) {
                if (k > n) {
                    assert((egttools::binomialCoeff<Output, std::uint64_t>(n, k) == 0));
                    continue;
                }
                const auto &entry = pascal[n][k];
                if (entry.too_big || entry.value > max_value) {
                    assert(throws_overflow<Output>(n, k));
                } else {
                    assert((static_cast<std::uint64_t>(egttools::binomialCoeff<Output, std::uint64_t>(n, k)) ==
                            entry.value));
                }
            }
        }
    }
}// namespace

int main() {
    constexpr std::uint64_t max_n = 80;
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

    check_against_pascal<std::int64_t>(pascal);
    check_against_pascal<std::uint64_t>(pascal);

    // Boundaries: C(66, 33) is the largest central coefficient that fits in int64_t,
    // C(67, 33) fits only in uint64_t, C(68, 34) fits in neither.
    assert((egttools::binomialCoeff<std::int64_t, std::uint64_t>(66, 33) == 7219428434016265740LL));
    assert(throws_overflow<std::int64_t>(67, 33));
    assert((egttools::binomialCoeff<std::uint64_t, std::uint64_t>(67, 33) == 14226520737620288370ULL));
    assert(throws_overflow<std::uint64_t>(68, 34));

    // Huge n (e.g. a negative int converted to size_t): exact for k = 1, a fast throw otherwise.
    assert((egttools::binomialCoeff<std::uint64_t, std::uint64_t>(u64_max, 1) == u64_max));
    assert(throws_overflow<std::int64_t>(u64_max, 1));
    assert(throws_overflow<std::uint64_t>(u64_max, u64_max / 2));

    // Negative arguments give 0, as before.
    assert((egttools::binomialCoeff<std::int64_t, int>(-1, 0)) == 0);
    assert((egttools::binomialCoeff<std::int64_t, int>(5, -1)) == 0);

    // 32-bit output: the case that broke PairwiseComparison (C(141, 5) computed in int).
    assert((egttools::binomialCoeff<int, int>(141, 5)) == 432295143);
    bool int_overflow = false;
    try {
        static_cast<void>(egttools::binomialCoeff<int, int>(128, 28));
    } catch (const std::overflow_error &) {
        int_overflow = true;
    }
    assert(int_overflow);

    // Non-integer outputs keep the unchecked loop.
    assert((std::abs(egttools::binomialCoeff<double, std::int64_t>(10, 3) - 120.0) < 1e-12));

    // Number of states C(Z + S - 1, S - 1): values that were wrong when computed in int,
    // and a population whose count does not fit in int64_t.
    assert(egttools::checked_nb_states(136, 6) == 432295143);
    assert(egttools::checked_nb_states(52, 8) == 341149446);
    assert(egttools::checked_nb_states(31, 10) == 273438880);
    assert(egttools::checked_nb_states(0, 3) == 1);
    bool overflow = false;
    try {
        static_cast<void>(egttools::checked_nb_states(100, 29));
    } catch (const std::overflow_error &error) {
        overflow = std::string(error.what()).find("Z = 100 and S = 29") != std::string::npos;
    }
    assert(overflow);

    std::cout << "binomialCoeff (integer path) and checked_nb_states tests passed.\n";
    return 0;
}

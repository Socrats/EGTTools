// State indexing and random sampling helpers (finite_populations/Utils.hpp, Sampling.h).
#include <random>
#include <set>
#include <unordered_set>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include <egttools/Distributions.h>
#include <egttools/Sampling.h>
#include <egttools/Types.h>
#include <egttools/finite_populations/Utils.hpp>

using egttools::VectorXui;
using egttools::FinitePopulations::calculate_state;
using egttools::FinitePopulations::sample_simplex;

TEST_CASE("sample_simplex and calculate_state are inverse bijections", "[sampling][state]") {
    // Every index in [0, C(Z + S - 1, S - 1)) maps to a distinct state with the right total, and
    // calculate_state maps it back to the same index.
    for (const auto &[Z, S]: std::vector<std::pair<size_t, size_t>>{{1, 2}, {6, 2}, {5, 3}, {4, 4}}) {
        const auto nb_states = egttools::starsBars<size_t, int64_t>(Z, S);
        std::set<std::vector<size_t>> seen;
        for (int64_t i = 0; i < nb_states; ++i) {
            const VectorXui state = sample_simplex(static_cast<size_t>(i), Z, S);
            REQUIRE(static_cast<size_t>(state.size()) == S);
            CHECK(state.sum() == Z);
            CHECK(calculate_state(Z, state) == static_cast<size_t>(i));
            seen.insert(std::vector<size_t>(state.data(), state.data() + state.size()));
        }
        CHECK(seen.size() == static_cast<size_t>(nb_states));
    }
}

TEST_CASE("sample_simplex orders two-strategy states by the count of strategy 0", "[sampling][state]") {
    // The convention the games' payoff tables rely on: with two strategies, index i is the state
    // (Z - i, i) (Utils.cpp).
    constexpr size_t Z = 5;
    for (size_t i = 0; i <= Z; ++i) {
        const VectorXui state = sample_simplex(i, Z, 2);
        CHECK(state(0) == Z - i);
        CHECK(state(1) == i);
    }
}

// Known bug, kept as an expected failure until it is fixed (roadmap: MEM-12): when Floyd's
// algorithm draws a value v that is already in the set, sample_without_replacement inserts v
// again instead of the upper bound r, so the sample can have fewer than k elements.
TEST_CASE("sample_without_replacement returns k distinct values", "[sampling][!shouldfail]") {
    std::mt19937_64 generator(3610063510ULL);
    bool always_k = true;
    for (int trial = 0; trial < 1000 && always_k; ++trial) {
        std::unordered_set<long> sample;
        // k = 9 values from [0, 10): collisions are frequent, so the bug shows quickly.
        egttools::sampling::sample_without_replacement<long, long>(0L, 10L, 9L, sample, generator);
        always_k = sample.size() == 9;
    }
    CHECK(always_k);
}

TEST_CASE("sample_simplex_direct_method returns a state of the right size and total", "[sampling]") {
    std::mt19937_64 generator(3610063510ULL);
    egttools::VectorXli state = egttools::VectorXli::Zero(3);
    for (int trial = 0; trial < 100; ++trial) {
        egttools::FinitePopulations::sample_simplex_direct_method<long, long, egttools::VectorXli, std::mt19937_64>(
                3, 100, state, generator);
        CHECK(state.sum() == 100);
        CHECK(state.minCoeff() >= 0);
    }
}

// Known disagreement with the documentation (roadmap: MEM-12): the method is documented to
// sample the simplex uniformly, but with distinct cut points drawn from [0, Z) only the first
// strategy can have a count of 0, a bias of order S / Z. For Z = 2 and three strategies the cut
// points are always {0, 1}, so it returns (0, 1, 1) every time instead of each of the 6 states.
// The exact method (Smith & Tromble: cut points from {1, ..., Z + S - 1}, minus 1) costs the
// same, so this is not a speed trade-off; MEM-12 either fixes it or documents the bias.
TEST_CASE("sample_simplex_direct_method can return every state", "[sampling][!shouldfail]") {
    std::mt19937_64 generator(3610063510ULL);
    egttools::VectorXli state = egttools::VectorXli::Zero(3);
    std::set<std::vector<long>> seen;
    for (int trial = 0; trial < 2000; ++trial) {
        egttools::FinitePopulations::sample_simplex_direct_method<long, long, egttools::VectorXli, std::mt19937_64>(
                3, 2, state, generator);
        seen.insert({state(0), state(1), state(2)});
    }
    CHECK(seen.size() == 6);
}

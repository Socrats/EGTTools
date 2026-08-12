//
// C++-level smoke test for GarciaGroup and MLS<GarciaGroup> (Garcia & van den Bergh 2011).
//
#include <egttools/Types.h>
#include <egttools/finite_populations/structure/GarciaGroup.hpp>
#include <egttools/finite_populations/evolvers/MLS.hpp>

#include <cassert>
#include <cmath>
#include <iostream>

using egttools::Matrix2D;
using egttools::VectorXui;
using egttools::FinitePopulations::GarciaGroup;
using egttools::FinitePopulations::MLS;

static Matrix2D donation_game(double b, double c = 1.0) {
    Matrix2D A(2, 2);
    A << b - c, -c,
         b, 0.0;
    return A;
}

static void test_garcia_group_basic_invariants() {
    Matrix2D payoff_in = donation_game(3.0);
    Matrix2D payoff_out = donation_game(3.0);
    VectorXui init_strategies(2);
    init_strategies << 2, 2;

    GarciaGroup group(2, 4, 0.1, init_strategies, payoff_in, payoff_out);
    assert(group.nb_strategies() == 2);
    assert(group.max_group_size() == 4);
    assert(group.group_size() == 4);
    assert(!group.isGroupOversize());

    std::cout << "test_garcia_group_basic_invariants passed\n";
}

static void test_mls_garcia_fixation_probability() {
    Matrix2D payoff_in = donation_game(3.0);
    Matrix2D payoff_out = donation_game(3.0);

    MLS<GarciaGroup> evolver(2000, 2, 4, 3);
    assert(evolver.nb_strategies() == 2);
    assert(evolver.max_pop_size() == 12);

    double fp = evolver.fixationProbability(
        0, 1, 100,
        /*q=*/0.01, /*lambda=*/0.0, /*w=*/0.1,
        /*alpha=*/0.8, /*kappa=*/0.0, /*z=*/0.0,
        payoff_in, payoff_out);
    assert(fp >= 0.0 && fp <= 1.0);

    std::cout << "test_mls_garcia_fixation_probability passed\n";
}

static void test_mls_garcia_invalid_strategy_index_throws() {
    Matrix2D payoff_in = donation_game(3.0);
    Matrix2D payoff_out = donation_game(3.0);
    MLS<GarciaGroup> evolver(2000, 2, 4, 3);

    bool threw = false;
    try {
        evolver.fixationProbability(5, 1, 10, 0.01, 0.0, 0.1, 0.8, 0.0, 0.0, payoff_in, payoff_out);
    } catch (const std::invalid_argument &) {
        threw = true;
    }
    assert(threw);

    std::cout << "test_mls_garcia_invalid_strategy_index_throws passed\n";
}

int main() {
    test_garcia_group_basic_invariants();
    test_mls_garcia_fixation_probability();
    test_mls_garcia_invalid_strategy_index_throws();
    return 0;
}

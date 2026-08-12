//
// C++-level smoke test for Group and MLS<Group> (Traulsen & Nowak 2006).
//
#include <egttools/Types.h>
#include <egttools/finite_populations/structure/Group.hpp>
#include <egttools/finite_populations/evolvers/MLS.hpp>

#include <cassert>
#include <iostream>
#include <random>

using egttools::Matrix2D;
using egttools::VectorXui;
using egttools::FinitePopulations::Group;
using egttools::FinitePopulations::MLS;

static Matrix2D donation_game(double b, double c = 1.0) {
    Matrix2D A(2, 2);
    A << b - c, -c,
         b, 0.0;
    return A;
}

static void test_group_basic_invariants() {
    Matrix2D payoffs = donation_game(3.0);
    VectorXui init_strategies(2);
    init_strategies << 2, 2;

    Group group(2, 4, 0.1, init_strategies, payoffs);
    assert(group.nb_strategies() == 2);
    assert(group.max_group_size() == 4);
    assert(group.group_size() == 4);
    assert(!group.isGroupOversize());

    group.totalPayoff();
    std::mt19937_64 generator(42);
    auto [group_increased, dying_strategy] = group.createOffspring(generator);
    (void) group_increased;
    (void) dying_strategy;
    assert(group.isGroupOversize());

    std::cout << "test_group_basic_invariants passed\n";
}

static void test_mls_traulsen_evolve() {
    Matrix2D payoffs = donation_game(3.0);
    Eigen::VectorXd freq(2);
    freq << 0.5, 0.5;

    MLS<Group> evolver(2000, 2, 4, 3, 0.1, freq, payoffs);
    assert(evolver.nb_strategies() == 2);
    assert(evolver.max_pop_size() == 12);

    VectorXui init_state(2);
    init_state << 6, 6;
    auto final_freqs = evolver.evolve(0.0, 0.1, init_state);
    assert(static_cast<size_t>(final_freqs.size()) == 2);
    assert(std::abs(final_freqs.sum() - 1.0) < 1e-9);
    for (Eigen::Index i = 0; i < final_freqs.size(); ++i) {
        assert(final_freqs(i) >= 0.0 && final_freqs(i) <= 1.0);
    }

    std::cout << "test_mls_traulsen_evolve passed\n";
}

static void test_mls_traulsen_fixation_probability() {
    Matrix2D payoffs = donation_game(3.0);
    Eigen::VectorXd freq(2);
    freq << 0.5, 0.5;

    MLS<Group> evolver(2000, 2, 4, 3, 0.1, freq, payoffs);
    double fp = evolver.fixationProbability(0, 1, 100, 0.0, 0.1);
    assert(fp >= 0.0 && fp <= 1.0);

    double fp_migration = evolver.fixationProbability(0, 1, 100, 0.0, 0.05, 0.1);
    assert(fp_migration >= 0.0 && fp_migration <= 1.0);

    std::cout << "test_mls_traulsen_fixation_probability passed\n";
}

static void test_mls_traulsen_gradient_of_selection() {
    Matrix2D payoffs = donation_game(3.0);
    Eigen::VectorXd freq(2);
    freq << 0.5, 0.5;

    MLS<Group> evolver(2000, 2, 4, 3, 0.1, freq, payoffs);
    auto gradient = evolver.gradientOfSelection(0, 1, 50, 0.1);
    assert(static_cast<size_t>(gradient.size()) == evolver.max_pop_size() + 1);

    std::cout << "test_mls_traulsen_gradient_of_selection passed\n";
}

int main() {
    test_group_basic_invariants();
    test_mls_traulsen_evolve();
    test_mls_traulsen_fixation_probability();
    test_mls_traulsen_gradient_of_selection();
    return 0;
}

// Multilevel selection: Group (Traulsen & Nowak 2006, PNAS 103, 10952), GarciaGroup (García &
// Traulsen 2012, PLoS ONE 7, e35962) and the MLS evolver.
//
// Group payoffs are checked by hand: an individual's fitness is 1 - w + w pi, with pi its average
// payoff against the other n - 1 members of its group. The group invariant
// group_size() == strategies().sum() <= max_group_size() (except transiently, after an offspring
// is created and before a member dies) is checked through every public operation, since the
// library only asserts it.
#include <random>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <egttools/Types.h>
#include <egttools/finite_populations/evolvers/MLS.hpp>
#include <egttools/finite_populations/structure/GarciaGroup.hpp>
#include <egttools/finite_populations/structure/Group.hpp>

using Catch::Matchers::WithinAbs;
using egttools::Matrix2D;
using egttools::VectorXui;
using egttools::FinitePopulations::GarciaGroup;
using egttools::FinitePopulations::Group;
using egttools::FinitePopulations::MLS;

namespace {
    // Donation game: a cooperator pays c for the co-player to receive b. Strategy 0 is C.
    Matrix2D donation_game(const double b, const double c = 1.0) {
        Matrix2D A(2, 2);
        A << b - c, -c,
                b, 0.0;
        return A;
    }
}// namespace

TEST_CASE("Group accessors after construction", "[mls][group]") {
    const Matrix2D payoffs = donation_game(3.0);
    VectorXui init(2);
    init << 2, 2;
    Group group(2, 4, 0.1, init, payoffs);
    CHECK(group.nb_strategies() == 2);
    CHECK(group.max_group_size() == 4);
    CHECK(group.group_size() == 4);
    CHECK_FALSE(group.isGroupOversize());
    CHECK_FALSE(group.isPopulationMonomorphic());
}

TEST_CASE("Group totalPayoff", "[mls][group]") {
    // Two C and two D, b = 3, c = 1, w = 0.1. A C meets one C and two D: pi = (2 - 2) / 3 = 0,
    // fitness 0.9. A D meets two C and one D: pi = 6 / 3 = 2, fitness 0.9 + 0.2 = 1.1.
    // Total 2 * 0.9 + 2 * 1.1 = 4.
    const Matrix2D payoffs = donation_game(3.0);
    VectorXui init(2);
    init << 2, 2;
    Group group(2, 4, 0.1, init, payoffs);
    CHECK_THAT(group.totalPayoff(), WithinAbs(4.0, 1e-14));
}

TEST_CASE("Group invariant through offspring, deaths and additions", "[mls][group]") {
    const Matrix2D payoffs = donation_game(3.0);
    VectorXui init(2);
    init << 2, 2;
    Group group(2, 4, 0.1, init, payoffs);
    std::mt19937_64 generator(42);
    int broken = 0;
    for (int step = 0; step < 1000; ++step) {
        group.totalPayoff();
        const auto [oversize, new_strategy] = group.createOffspring(generator);
        // A full group becomes oversize by one until a member dies.
        broken += !oversize || !group.isGroupOversize() || group.group_size() != 5 || new_strategy >= 2;
        broken += group.group_size() != group.strategies().sum();
        group.deleteMember(generator);
        broken += group.isGroupOversize() || group.group_size() != 4 || group.group_size() != group.strategies().sum();
    }
    CHECK(broken == 0);

    // addMember reports whether the group still fits; deleteMember(strategy) refuses an absent strategy.
    group.setPopulationHomogeneous(0);
    CHECK(group.isPopulationMonomorphic());
    CHECK_FALSE(group.deleteMember(size_t{1}));
    CHECK(group.deleteMember(size_t{0}));
    CHECK(group.addMember(1));
    CHECK_FALSE(group.addMember(1));
    CHECK(group.group_size() == group.strategies().sum());
}

// Known bug (roadmap: MEM-13): set_strategy_count throws when the new counts fit in the group
// (its condition is inverted) and accepts counts that do not, and it does not update group_size().
TEST_CASE("Group set_strategy_count accepts counts that fit", "[mls][group][!shouldfail]") {
    const Matrix2D payoffs = donation_game(3.0);
    VectorXui init(2);
    init << 2, 2;
    Group group(2, 4, 0.1, init, payoffs);
    VectorXui counts(2);
    counts << 1, 2;
    CHECK_NOTHROW(group.set_strategy_count(counts));
    CHECK(group.group_size() == 3);
    VectorXui too_many(2);
    too_many << 4, 4;
    CHECK_THROWS_AS(group.set_strategy_count(too_many), std::invalid_argument);
}

TEST_CASE("Group constructor validates its arguments", "[mls][group][validation]") {
    const Matrix2D payoffs = donation_game(3.0);
    VectorXui three(3);
    three << 1, 1, 1;
    CHECK_THROWS_AS(Group(2, 4, 0.1, three, payoffs), std::invalid_argument);
    CHECK_THROWS_AS(Group(3, 4, 0.1, three, payoffs), std::invalid_argument);
}

TEST_CASE("GarciaGroup accessors after construction", "[mls][garcia]") {
    const Matrix2D payoffs = donation_game(3.0);
    VectorXui init(2);
    init << 2, 2;
    GarciaGroup group(2, 4, 0.1, init, payoffs, payoffs);
    CHECK(group.nb_strategies() == 2);
    CHECK(group.max_group_size() == 4);
    CHECK(group.group_size() == 4);
    CHECK_FALSE(group.isGroupOversize());
}

TEST_CASE("MLS<Group> evolve returns frequencies", "[mls]") {
    const Matrix2D payoffs = donation_game(3.0);
    Eigen::VectorXd frequencies(2);
    frequencies << 0.5, 0.5;
    MLS<Group> evolver(2000, 2, 4, 3, 0.1, frequencies, payoffs);
    CHECK(evolver.nb_strategies() == 2);
    CHECK(evolver.max_pop_size() == 12);
    VectorXui init_state(2);
    init_state << 6, 6;
    const auto final_frequencies = evolver.evolve(0.0, 0.1, init_state);
    REQUIRE(final_frequencies.size() == 2);
    CHECK_THAT(final_frequencies.sum(), WithinAbs(1.0, 1e-12));
    CHECK(final_frequencies.minCoeff() >= 0.0);
}

TEST_CASE("MLS<Group> fixation probabilities and gradient", "[mls]") {
    const Matrix2D payoffs = donation_game(3.0);
    Eigen::VectorXd frequencies(2);
    frequencies << 0.5, 0.5;
    MLS<Group> evolver(2000, 2, 4, 3, 0.1, frequencies, payoffs);
    const double rho = evolver.fixationProbability(0, 1, 100, 0.0, 0.1);
    CHECK((rho >= 0.0 && rho <= 1.0));
    const double rho_migration = evolver.fixationProbability(0, 1, 100, 0.0, 0.05, 0.1);
    CHECK((rho_migration >= 0.0 && rho_migration <= 1.0));
    const auto gradient = evolver.gradientOfSelection(0, 1, 50, 0.1);
    CHECK(static_cast<size_t>(gradient.size()) == evolver.max_pop_size() + 1);
}

TEST_CASE("MLS<GarciaGroup> fixation probability", "[mls][garcia]") {
    const Matrix2D payoffs = donation_game(3.0);
    MLS<GarciaGroup> evolver(2000, 2, 4, 3);
    CHECK(evolver.nb_strategies() == 2);
    CHECK(evolver.max_pop_size() == 12);
    const double rho = evolver.fixationProbability(0, 1, 100, /*q=*/0.01, /*lambda=*/0.0, /*w=*/0.1,
                                                   /*alpha=*/0.8, /*kappa=*/0.0, /*z=*/0.0, payoffs, payoffs);
    CHECK((rho >= 0.0 && rho <= 1.0));
    CHECK_THROWS_AS(evolver.fixationProbability(5, 1, 10, 0.01, 0.0, 0.1, 0.8, 0.0, 0.0, payoffs, payoffs),
                    std::invalid_argument);
}

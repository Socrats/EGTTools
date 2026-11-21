// Collective-risk dilemmas: OneShotCRD, CRDGame (fixed number of rounds) and CRDGameTU (random
// number of rounds). Expected payoffs are worked out by hand from the rules of each game.
#include <cmath>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <egttools/SeedGenerator.h>
#include <egttools/finite_populations/behaviors/CRDStrategies.h>
#include <egttools/finite_populations/games/CRDGame.hpp>
#include <egttools/finite_populations/games/CRDGameTU.hpp>
#include <egttools/finite_populations/games/OneShotCRD.hpp>

using Catch::Matchers::WithinAbs;
using egttools::FinitePopulations::StrategyCounts;
using CRDPlayer = egttools::FinitePopulations::behaviors::CRD::CRDMemoryOnePlayer;

// One-shot CRD (Santos & Pacheco 2011, PNAS 108, 10421): with endowment b, cost c, risk r and
// threshold M, a group with k cooperators succeeds if k >= M. Defectors get b on success and
// b (1 - r) on failure; cooperators get the defectors' payoff minus c b.
// Strategy 0 is D, strategy 1 is C; compositions are (#D, #C).
TEST_CASE("OneShotCRD payoffs", "[games][crd]") {
    constexpr double b = 1.0, c = 0.1, r = 0.9;
    constexpr int N = 6, M = 3;
    egttools::FinitePopulations::OneShotCRD game(b, c, r, N, M);
    REQUIRE(game.payoffs().cols() == N + 1);
    for (int k = 0; k <= N; ++k) {
        const StrategyCounts group{static_cast<size_t>(N - k), static_cast<size_t>(k)};
        const double pi_d = k >= M ? b : b * (1 - r);
        if (k < N) CHECK_THAT(game.payoff(0, group), WithinAbs(pi_d, 1e-14));
        if (k > 0) CHECK_THAT(game.payoff(1, group), WithinAbs(pi_d - c * b, 1e-14));
    }
}

TEST_CASE("OneShotCRD group achievement in a population", "[games][crd]") {
    // Probability that a group of N = 2 drawn from Z = 4 with 2 cooperators has at least M = 1
    // cooperator: 1 - C(2, 2) / C(4, 2) = 5 / 6.
    egttools::FinitePopulations::OneShotCRD game(1.0, 0.1, 0.5, 2, 1);
    egttools::VectorXui state(2);
    state << 2, 2;
    CHECK_THAT(game.calculate_population_group_achievement(4, state), WithinAbs(5.0 / 6.0, 1e-14));
}

// Multi-round CRD (Milinski et al. 2008, PNAS 105, 2291): each player starts with the endowment
// and contributes every round; the game ends as soon as the public account reaches the threshold
// (players keep what they have left) or after the last round, and if the threshold was not
// reached everyone loses what is left with probability r. Payoffs are expected values.
// Always-x players contribute x every round.
namespace {
    struct CRDSetup {
        CRDPlayer always0{10, 0, 0, 0, 0};
        CRDPlayer always2{10, 2, 2, 2, 2};
        CRDPlayer always4{10, 4, 4, 4, 4};
        egttools::FinitePopulations::CRDStrategyVector strategies{&always0, &always2, &always4};
    };
}// namespace

TEST_CASE("CRDGame payoffs of always-x groups", "[games][crd]") {
    // Endowment 40, threshold 120, 10 rounds, groups of 6, risk 0.9.
    CRDSetup setup;
    egttools::FinitePopulations::CRDGame game(40, 120, 10, 6, 0.9, 1.0, setup.strategies);
    // Six always-0: nothing is contributed, everyone keeps 40 with probability 0.1.
    CHECK_THAT(game.payoff(0, StrategyCounts{6, 0, 0}), WithinAbs(4.0, 1e-12));
    // Six always-2: 12 per round, the threshold is reached in round 10, each keeps 40 - 20.
    CHECK_THAT(game.payoff(1, StrategyCounts{0, 6, 0}), WithinAbs(20.0, 1e-12));
    // Six always-4: 24 per round, reached in round 5, each keeps 40 - 20.
    CHECK_THAT(game.payoff(2, StrategyCounts{0, 0, 6}), WithinAbs(20.0, 1e-12));
    // Three always-0 and three always-4: 12 per round, reached in round 10.
    CHECK_THAT(game.payoff(0, StrategyCounts{3, 0, 3}), WithinAbs(40.0, 1e-12));
    CHECK_THAT(game.payoff(2, StrategyCounts{3, 0, 3}), WithinAbs(0.0, 1e-12));
    // Five always-0 and one always-4: 4 per round, 40 after 10 rounds, the threshold is missed.
    CHECK_THAT(game.payoff(0, StrategyCounts{5, 0, 1}), WithinAbs(40.0 * 0.1, 1e-12));
    CHECK_THAT(game.payoff(2, StrategyCounts{5, 0, 1}), WithinAbs(0.0, 1e-12));
}

TEST_CASE("CRDGame payoff table follows the group-composition order", "[games][crd]") {
    // payoffs()(strategy, i) is the payoff in the group sample_simplex(i, N, S).
    CRDSetup setup;
    egttools::FinitePopulations::CRDGame game(40, 120, 10, 6, 0.9, 1.0, setup.strategies);
    const auto &table = game.payoffs();
    REQUIRE(table.cols() == egttools::starsBars<int64_t>(6, 3));
    double max_error = 0.0;
    StrategyCounts group(3, 0);
    for (int64_t i = 0; i < table.cols(); ++i) {
        egttools::FinitePopulations::sample_simplex(static_cast<size_t>(i), 6, 3, group);
        for (int s = 0; s < 3; ++s)
            if (group[s] > 0) max_error = std::max(max_error, std::abs(table(s, i) - game.payoff(s, group)));
    }
    CHECK(max_error == 0.0);
}

// CRD with timing uncertainty (Fernández Domingos et al. 2020, iScience 23, 101752): the game
// lasts min_rounds + G rounds, G geometric with parameter p. CRDGameTU estimates each expected
// payoff from 10000 simulated games, so it is compared with the exact expectation within 5
// standard errors of that estimate.
TEST_CASE("CRDGameTU payoffs of always-x groups", "[games][crd][timing_uncertainty]") {
    egttools::Random::SeedGenerator::getInstance().setMainSeed(3610063510);
    CRDSetup setup;
    constexpr double p = 1.0 / 3, risk = 0.9;
    const auto tu = egttools::utils::TimingUncertainty(p);
    egttools::FinitePopulations::games::CRDGameTU game(40, 120, 8, 6, risk, tu, setup.strategies);

    // Always-0 and always-4 groups do not depend on the number of rounds (always-4 reaches the
    // threshold in round 5 < 8), so their estimates are exact.
    CHECK_THAT(game.payoff(0, StrategyCounts{6, 0, 0}), WithinAbs(4.0, 1e-12));
    CHECK_THAT(game.payoff(2, StrategyCounts{0, 0, 6}), WithinAbs(20.0, 1e-12));

    // Always-2 needs 10 rounds: the game lasts at least 10 rounds (G >= 2) with probability
    // (1 - p)^2, and each player keeps 20; otherwise it lasts 8 (G = 0, probability p) or 9
    // (G = 1, probability p (1 - p)) rounds, and each keeps 24 or 22 with probability 1 - r.
    const double p_success = (1 - p) * (1 - p), p8 = p, p9 = p * (1 - p);
    const double values[] = {20.0, 24.0 * (1 - risk), 22.0 * (1 - risk)};
    const double probabilities[] = {p_success, p8, p9};
    double mean = 0.0, second_moment = 0.0;
    for (int i = 0; i < 3; ++i) {
        mean += probabilities[i] * values[i];
        second_moment += probabilities[i] * values[i] * values[i];
    }
    const double standard_error = std::sqrt((second_moment - mean * mean) / 10000);
    CHECK_THAT(game.payoff(1, StrategyCounts{0, 6, 0}), WithinAbs(mean, 5 * standard_error));
}

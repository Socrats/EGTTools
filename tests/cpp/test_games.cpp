// Matrix games and the N-player stag hunt: payoff tables and fitness.
//
// Fitness is checked against Definitions 2.1 and 2.2 of docs/theory/pairwise_comparison.md
// (tests/cpp/reference.hpp), at every state of a small population.
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <egttools/Types.h>
#include <egttools/finite_populations/Utils.hpp>
#include <egttools/finite_populations/behaviors/NFGStrategies.hpp>
#include <egttools/finite_populations/games/Matrix2PlayerGameHolder.hpp>
#include <egttools/finite_populations/games/MatrixNPlayerGameHolder.hpp>
#include <egttools/finite_populations/games/NPlayerStagHunt.hpp>
#include <egttools/finite_populations/games/NormalFormGame.h>

#include "reference.hpp"

using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;
using egttools::Matrix2D;
using egttools::VectorXui;
namespace twoActions = egttools::FinitePopulations::behaviors::twoActions;

namespace {
    // Largest |game fitness - reference| over every state of a population of size Z and every
    // strategy present in it. The game receives the counts without the focal individual.
    template<typename Game, typename Reference>
    double max_fitness_error(Game &game, const size_t Z, const size_t S, Reference &&reference_fitness) {
        double max_error = 0.0;
        const auto nb_states = egttools::starsBars<size_t, int64_t>(Z, S);
        for (int64_t index = 0; index < nb_states; ++index) {
            const VectorXui state = egttools::FinitePopulations::sample_simplex(static_cast<size_t>(index), Z, S);
            const std::vector<long> n(state.data(), state.data() + state.size());
            for (int l = 0; l < static_cast<int>(S); ++l) {
                if (n[l] == 0) continue;
                VectorXui others = state;
                others(l) -= 1;
                const double f = game.calculate_fitness(l, Z, others);
                max_error = std::max(max_error, std::abs(f - reference_fitness(n, l)));
            }
        }
        return max_error;
    }
}// namespace

TEST_CASE("NormalFormGame without strategies uses the payoff matrix", "[games][normal_form]") {
    Matrix2D A(2, 2);
    A << 3.0, 0.0, 5.0, 1.0;
    egttools::FinitePopulations::NormalFormGame game(1, A);
    CHECK(game.nb_strategies() == 2);
    CHECK(game.payoffs().isApprox(A));
    CHECK(max_fitness_error(game, 7, 2, [&A](const auto &n, int l) {
              return reference::fitness_two_player(A, n, l);
          }) < 1e-13);
}

TEST_CASE("NormalFormGame averages the payoffs of repeated rounds", "[games][normal_form]") {
    // A(own action, other action), with DEFECT = 0 and COOPERATE = 1.
    Matrix2D A(2, 2);
    A << 1.0, 5.0, 0.0, 3.0;
    twoActions::Cooperator all_c;
    twoActions::Defector all_d;
    twoActions::TitForTat tft;
    egttools::FinitePopulations::NFGStrategyVector strategies{&all_c, &all_d, &tft};
    egttools::FinitePopulations::NormalFormGame game(3, A, strategies);
    // expected_payoffs()(i, j): average payoff of strategy i against strategy j (payoffs() is A).
    const auto &P = game.expected_payoffs();
    REQUIRE(P.rows() == 3);
    REQUIRE(P.cols() == 3);
    // AllC against AllD: (C, D) every round.
    CHECK_THAT(P(0, 1), WithinAbs(A(1, 0), 1e-14));
    CHECK_THAT(P(1, 0), WithinAbs(A(0, 1), 1e-14));
    // TFT cooperates with AllC every round.
    CHECK_THAT(P(2, 0), WithinAbs(A(1, 1), 1e-14));
    // TFT against AllD: C then D, D; AllD defects every round.
    CHECK_THAT(P(2, 1), WithinAbs((A(1, 0) + 2 * A(0, 0)) / 3, 1e-14));
    CHECK_THAT(P(1, 2), WithinAbs((A(0, 1) + 2 * A(0, 0)) / 3, 1e-14));
    // Fitness follows Definition 2.1 with the averaged payoffs.
    const Matrix2D averaged = P;
    CHECK(max_fitness_error(game, 6, 3, [&averaged](const auto &n, int l) {
              return reference::fitness_two_player(averaged, n, l);
          }) < 1e-13);
}

TEST_CASE("Matrix2PlayerGameHolder fitness follows Definition 2.1", "[games][holders]") {
    Matrix2D A(3, 3);
    A << 0.0, -1.0, 2.0,
            1.5, 0.5, -0.5,
            -2.0, 3.0, 1.0;
    egttools::FinitePopulations::Matrix2PlayerGameHolder game(3, A);
    CHECK(game.payoffs().isApprox(A));
    CHECK(max_fitness_error(game, 8, 3, [&A](const auto &n, int l) {
              return reference::fitness_two_player(A, n, l);
          }) < 1e-13);
}

TEST_CASE("MatrixNPlayerGameHolder fitness follows Definition 2.2", "[games][holders]") {
    // Groups of N = 3 with two strategies: column i of the payoff table is the group with i
    // members of strategy 1 (focal included).
    constexpr int N = 3;
    Matrix2D table(2, N + 1);
    table << 0.5, 2.0, -1.0, 4.0,
            -3.0, 1.0, 0.25, 2.5;
    egttools::FinitePopulations::MatrixNPlayerGameHolder game(2, N, table);
    CHECK(game.payoffs().isApprox(table));
    // A focal l-strategist with k co-players of strategy 1 is in column k + (l == 1).
    const auto payoff = [&table](int l, long k) { return table(l, k + (l == 1 ? 1 : 0)); };
    CHECK(max_fitness_error(game, 9, 2, [&payoff](const auto &n, int l) {
              return reference::fitness_n_player_two_strategies(payoff, N, n, l);
          }) < 1e-13);
}

TEST_CASE("MatrixNPlayerGameHolder rejects groups of two and tables of the wrong shape", "[games][holders]") {
    // Two-player games use Matrix2PlayerGameHolder.
    CHECK_THROWS_AS(egttools::FinitePopulations::MatrixNPlayerGameHolder(2, 2, Matrix2D::Zero(2, 3)),
                    std::invalid_argument);
    CHECK_THROWS_AS(egttools::FinitePopulations::MatrixNPlayerGameHolder(2, 3, Matrix2D::Zero(2, 3)),
                    std::invalid_argument);
}

// N-player stag hunt (Pacheco, Santos, Souza & Skyrms 2009, Proc. R. Soc. B 276, 315-321):
// with k cooperators in a group of N, threshold M, enhancement F and cost c,
//   Pi_D(k) = k F c / N * theta(k - M),   Pi_C(k) = Pi_D(k) - c.
// Strategy 0 is D, strategy 1 is C; compositions are (#D, #C).
namespace {
    double stag_hunt_defector(const long k, const long N, const long M, const double F, const double c) {
        return k >= M ? static_cast<double>(k) * F * c / static_cast<double>(N) : 0.0;
    }

    double max_stag_hunt_error(const int N, const int M, const double F, const double c) {
        egttools::FinitePopulations::NPlayerStagHunt game(N, M, F, c);
        double max_error = 0.0;
        for (long k = 0; k <= N; ++k) {
            const egttools::FinitePopulations::StrategyCounts group{static_cast<size_t>(N - k), static_cast<size_t>(k)};
            const double pi_d = stag_hunt_defector(k, N, M, F, c);
            if (k < N) max_error = std::max(max_error, std::abs(game.payoff(0, group) - pi_d));
            if (k > 0) max_error = std::max(max_error, std::abs(game.payoff(1, group) - (pi_d - c)));
        }
        return max_error;
    }
}// namespace

TEST_CASE("NPlayerStagHunt payoffs with unit cost", "[games][stag_hunt]") {
    CHECK(max_stag_hunt_error(6, 3, 3.0, 1.0) < 1e-14);
}

// Known bug (roadmap: API-13): in mixed groups the defectors' payoff is k F / N instead of
// k F c / N, so the payoffs match the paper only when c = 1 (the Python NPlayerStagHunt in
// egttools.games has the same formula). The groups of cooperators only are right.
TEST_CASE("NPlayerStagHunt payoffs with a cost other than 1", "[games][stag_hunt][!shouldfail]") {
    CHECK(max_stag_hunt_error(6, 3, 3.0, 0.5) < 1e-14);
}

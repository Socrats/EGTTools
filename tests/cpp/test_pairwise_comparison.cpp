// The analytical pairwise comparison process (analytical/PairwiseComparison.hpp) and the
// matrix-free transition operator (PairwiseComparisonTransitionOperator.hpp).
//
// References: Definitions 2.1, 3.1, 4.1 and 5.1 of docs/theory/pairwise_comparison.md, and the
// closed-form fixation probability of a birth-death chain (tests/cpp/reference.hpp).
#include <cmath>
#include <limits>
#include <random>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <egttools/Types.h>
#include <egttools/finite_populations/PairwiseComparisonTransitionOperator.hpp>
#include <egttools/finite_populations/Utils.hpp>
#include <egttools/finite_populations/analytical/PairwiseComparison.hpp>
#include <egttools/finite_populations/games/Matrix2PlayerGameHolder.hpp>
#include <egttools/finite_populations/games/NormalFormGame.h>

#include "reference.hpp"

using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;
using egttools::Matrix2D;
using egttools::Vector;
using egttools::VectorXui;
using egttools::FinitePopulations::calculate_state;
using egttools::FinitePopulations::sample_simplex;
using PairwiseComparison = egttools::FinitePopulations::analytical::PairwiseComparison;

namespace {
    Matrix2D rock_paper_scissors_like() {
        // An asymmetric 3-strategy game, so no transition probabilities coincide by symmetry.
        Matrix2D A(3, 3);
        A << 0.0, -1.0, 2.0,
                1.5, 0.5, -0.5,
                -2.0, 3.0, 1.0;
        return A;
    }

    std::vector<long> to_counts(const VectorXui &state) {
        return {state.data(), state.data() + state.size()};
    }

    std::vector<double> reference_fitness(const Matrix2D &A, const std::vector<long> &n) {
        std::vector<double> f(n.size(), 0.0);
        for (int l = 0; l < static_cast<int>(n.size()); ++l)
            if (n[l] > 0) f[l] = reference::fitness_two_player(A, n, l);
        return f;
    }

    // Definition 5.1 from Definition 4.1: G_l = sum_{j != l} [T_{j->l} - T_{l->j}].
    std::vector<double> reference_gradient(const Matrix2D &A, const std::vector<long> &n, const double beta,
                                           const double mu) {
        const auto f = reference_fitness(A, n);
        const int k = static_cast<int>(n.size());
        std::vector<double> G(static_cast<size_t>(k), 0.0);
        for (int l = 0; l < k; ++l)
            for (int j = 0; j < k; ++j) {
                if (j == l) continue;
                if (n[j] > 0) G[l] += reference::transition(n, f, j, l, beta, mu);
                if (n[l] > 0) G[l] -= reference::transition(n, f, l, j, beta, mu);
            }
        return G;
    }
}// namespace

TEST_CASE("PairwiseComparison sizes", "[pairwise_comparison]") {
    egttools::FinitePopulations::Matrix2PlayerGameHolder game(3, rock_paper_scissors_like());
    PairwiseComparison pc(8, game, 256);
    CHECK(pc.population_size() == 8);
    CHECK(pc.nb_strategies() == 3);
    CHECK(pc.nb_states() == egttools::starsBars<int64_t>(8, 3));
}

TEST_CASE("calculate_transition_matrix follows Definition 4.1", "[pairwise_comparison]") {
    constexpr int Z = 5;
    constexpr double beta = 0.7, mu = 0.05;
    const Matrix2D A = rock_paper_scissors_like();
    egttools::FinitePopulations::Matrix2PlayerGameHolder game(3, A);
    PairwiseComparison pc(Z, game, 1000);
    const Matrix2D P = pc.calculate_transition_matrix(beta, mu);// P(from, to)

    double max_error = 0.0, max_row_sum_error = 0.0;
    for (int64_t from = 0; from < pc.nb_states(); ++from) {
        const VectorXui state = sample_simplex(static_cast<size_t>(from), Z, 3);
        const auto n = to_counts(state);
        const auto f = reference_fitness(A, n);
        Vector expected_row = Vector::Zero(pc.nb_states());
        for (int j = 0; j < 3; ++j) {
            if (n[j] == 0) continue;
            for (int l = 0; l < 3; ++l) {
                if (l == j) continue;
                VectorXui destination = state;
                destination(j) -= 1;
                destination(l) += 1;
                expected_row(static_cast<Eigen::Index>(calculate_state(Z, destination))) +=
                        reference::transition(n, f, j, l, beta, mu);
            }
        }
        expected_row(from) = 1.0 - expected_row.sum();
        max_error = std::max(max_error, (P.row(from).transpose() - expected_row).cwiseAbs().maxCoeff());
        max_row_sum_error = std::max(max_row_sum_error, std::abs(P.row(from).sum() - 1.0));
    }
    CHECK(max_error < 1e-13);
    CHECK(max_row_sum_error < 1e-13);
}

TEST_CASE("calculate_gradient_of_selection follows Definition 5.1", "[pairwise_comparison]") {
    constexpr int Z = 7;
    constexpr double beta = 1.3;
    const Matrix2D A = rock_paper_scissors_like();
    egttools::FinitePopulations::Matrix2PlayerGameHolder game(3, A);
    PairwiseComparison pc(Z, game, 1000);

    double max_error = 0.0, max_error_mu = 0.0, max_sum = 0.0;
    for (int64_t index = 0; index < pc.nb_states(); ++index) {
        const VectorXui state = sample_simplex(static_cast<size_t>(index), Z, 3);
        const auto n = to_counts(state);
        const Vector G = pc.calculate_gradient_of_selection(beta, state);
        const Vector G_mu = pc.calculate_gradient_of_selection_with_mutation(beta, 0.1, state);
        const auto expected = reference_gradient(A, n, beta, 0.0);
        const auto expected_mu = reference_gradient(A, n, beta, 0.1);
        for (int l = 0; l < 3; ++l) {
            max_error = std::max(max_error, std::abs(G(l) - expected[static_cast<size_t>(l)]));
            max_error_mu = std::max(max_error_mu, std::abs(G_mu(l) - expected_mu[static_cast<size_t>(l)]));
        }
        // Each individual that leaves one strategy joins another.
        max_sum = std::max(max_sum, std::abs(G.sum()));
    }
    CHECK(max_error < 1e-14);
    CHECK(max_error_mu < 1e-14);
    CHECK(max_sum < 1e-14);
}

TEST_CASE("calculate_fixation_probability matches the birth-death closed form", "[pairwise_comparison]") {
    // Prisoner's dilemma: D (index 1) dominates C (index 0).
    constexpr int Z = 20;
    Matrix2D A(2, 2);
    A << 3.0, 0.0, 5.0, 1.0;
    egttools::FinitePopulations::NormalFormGame game(1, A);
    PairwiseComparison pc(Z, game, 1000);

    for (const double beta: {0.0, 0.1, 1.0, 5.0}) {
        for (const auto &[invader, resident]: std::vector<std::pair<int, int>>{{0, 1}, {1, 0}}) {
            const auto delta_f = [&, invader = invader, resident = resident](long i) {
                std::vector<long> n(2, 0);
                n[static_cast<size_t>(invader)] = i;
                n[static_cast<size_t>(resident)] = Z - i;
                return reference::fitness_two_player(A, n, invader) - reference::fitness_two_player(A, n, resident);
            };
            const double expected = reference::fixation_probability(delta_f, Z, beta);
            CAPTURE(beta, invader, resident);
            CHECK_THAT(pc.calculate_fixation_probability(invader, resident, beta), WithinRel(expected, 1e-12));
            CHECK_THAT(pc.calculate_fixation_probability_boost(invader, resident, beta), WithinRel(expected, 1e-12));
        }
    }
    // Neutral selection: 1 / Z.
    CHECK_THAT(pc.calculate_fixation_probability(0, 1, 0.0), WithinRel(1.0 / Z, 1e-14));
}

TEST_CASE("calculate_log_fixation_probability stays finite when the probability underflows",
          "[pairwise_comparison]") {
    // C invading D with Z = 200 and beta = 50: rho is far below the smallest double.
    Matrix2D A(2, 2);
    A << 3.0, 0.0, 5.0, 1.0;
    egttools::FinitePopulations::NormalFormGame game(1, A);
    PairwiseComparison pc(200, game, 1000);
    CHECK(pc.calculate_fixation_probability(0, 1, 50.0) == 0.0);
    const double log_rho = pc.calculate_log_fixation_probability(0, 1, 50.0);
    CHECK(std::isfinite(log_rho));
    CHECK(log_rho < std::log(std::numeric_limits<double>::denorm_min()));
    // Where rho is representable, the two agree.
    CHECK_THAT(pc.calculate_log_fixation_probability(1, 0, 0.5),
               WithinRel(std::log(pc.calculate_fixation_probability(1, 0, 0.5)), 1e-12));
}

TEST_CASE("calculate_transition_and_fixation_matrix_sml", "[pairwise_comparison][sml]") {
    // Small-mutation limit (Fudenberg & Imhof 2006): fixation(i, j) is the probability that one
    // j-mutant fixates in a population of i, and T(i, j) = fixation(i, j) / (S - 1) for i != j.
    constexpr int Z = 10;
    constexpr double beta = 0.5;
    egttools::FinitePopulations::Matrix2PlayerGameHolder game(3, rock_paper_scissors_like());
    PairwiseComparison pc(Z, game, 1000);
    const auto [T, fixation] = pc.calculate_transition_and_fixation_matrix_sml(beta);
    REQUIRE(T.rows() == 3);
    REQUIRE(T.cols() == 3);
    for (int i = 0; i < 3; ++i) {
        CHECK_THAT(T.row(i).sum(), WithinAbs(1.0, 1e-14));
        for (int j = 0; j < 3; ++j) {
            if (i == j) continue;
            CAPTURE(i, j);
            CHECK_THAT(fixation(i, j), WithinRel(pc.calculate_fixation_probability(j, i, beta), 1e-14));
            CHECK_THAT(T(i, j), WithinRel(fixation(i, j) / 2.0, 1e-14));
        }
    }
}

TEST_CASE("PairwiseComparison rejects invalid strategy indices and states", "[pairwise_comparison][validation]") {
    Matrix2D A(2, 2);
    A << 3.0, 0.0, 5.0, 1.0;
    egttools::FinitePopulations::NormalFormGame game(1, A);
    PairwiseComparison pc(10, game, 100);
    // Validation.hpp: an index out of range is std::out_of_range (IndexError in Python).
    CHECK_THROWS_AS(pc.calculate_fixation_probability(2, 0, 1.0), std::out_of_range);
    CHECK_THROWS_AS(pc.calculate_fixation_probability(0, 0, 1.0), std::invalid_argument);
    VectorXui wrong_total(2);
    wrong_total << 3, 3;
    CHECK_THROWS_AS(pc.calculate_gradient_of_selection(1.0, wrong_total), std::invalid_argument);
}

TEST_CASE("PairwiseComparisonTransitionOperator applies the transition matrix", "[pairwise_comparison][operator]") {
    // The operator enumerates transitions on the fly; it must agree with the sparse matrix.
    constexpr int Z = 6;
    constexpr double beta = 0.7, mu = 0.05;
    egttools::FinitePopulations::Matrix2PlayerGameHolder game(3, rock_paper_scissors_like());
    PairwiseComparison pc(Z, game, 1000);
    const egttools::SparseMatrix2D P = pc.calculate_transition_matrix(beta, mu);
    const egttools::FinitePopulations::PairwiseComparisonTransitionOperator op(Z, game, beta, mu);
    REQUIRE(op.size() == pc.nb_states());

    std::mt19937_64 generator(3610063510ULL);
    std::uniform_real_distribution<double> uniform(-1.0, 1.0);
    Vector x(op.size());
    for (Eigen::Index i = 0; i < x.size(); ++i) x(i) = uniform(generator);

    Vector y(op.size()), y_transpose(op.size()), residual(op.size());
    op.apply(x, y);
    op.apply_transpose(x, y_transpose);
    op.apply_residual(x, residual);
    const Vector Px = P * x;
    const Vector Ptx = P.transpose() * x;
    CHECK((y - Px).cwiseAbs().maxCoeff() < 1e-13);
    CHECK((y_transpose - Ptx).cwiseAbs().maxCoeff() < 1e-13);
    CHECK((residual - (x - Ptx)).cwiseAbs().maxCoeff() < 1e-13);
    // P is row-stochastic: P 1 = 1.
    Vector ones = Vector::Ones(op.size()), p_ones(op.size());
    op.apply(ones, p_ones);
    CHECK((p_ones - ones).cwiseAbs().maxCoeff() < 1e-13);
}

TEST_CASE("PairwiseComparisonTransitionOperator rejects invalid parameters", "[pairwise_comparison][operator]") {
    egttools::FinitePopulations::Matrix2PlayerGameHolder game(3, rock_paper_scissors_like());
    using Operator = egttools::FinitePopulations::PairwiseComparisonTransitionOperator;
    CHECK_THROWS_AS(Operator(1, game, 1.0, 0.1), std::invalid_argument);
    CHECK_THROWS_AS(Operator(5, game, -1.0, 0.1), std::invalid_argument);
    CHECK_THROWS_AS(Operator(5, game, 1.0, 1.5), std::invalid_argument);
    const Operator op(5, game, 1.0, 0.1);
    Vector x = Vector::Zero(op.size() + 1), y = Vector::Zero(op.size());
    CHECK_THROWS_AS(op.apply(x, y), std::invalid_argument);
}

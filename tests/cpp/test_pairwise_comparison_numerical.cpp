// Monte Carlo simulation of the pairwise comparison process (PairwiseComparisonNumerical.hpp).
//
// Trajectories are checked against the rules of the process (Definition 3.2 of
// docs/theory/pairwise_comparison.md); estimates are compared with exact values of the
// two-strategy birth-death chain (tests/cpp/reference.hpp). The rigorous statistical
// validation of the estimators against the exact chain is in the Python tests (MC-09).
#include <cmath>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <egttools/SeedGenerator.h>
#include <egttools/Types.h>
#include <egttools/finite_populations/PairwiseComparisonNumerical.hpp>
#include <egttools/finite_populations/games/NormalFormGame.h>

#include "reference.hpp"

using Catch::Matchers::WithinAbs;
using egttools::Matrix2D;
using egttools::VectorXui;

namespace {
    Matrix2D prisoners_dilemma() {
        Matrix2D A(2, 2);
        A << 3.0, 0.0, 5.0, 1.0;// strategy 1 (D) dominates strategy 0 (C)
        return A;
    }

    // Fitness of strategy l with k individuals of strategy 1 in a population of Z (Definition 2.1).
    double fitness(const Matrix2D &A, const long Z, const long k, const int l) {
        return reference::fitness_two_player(A, {Z - k, k}, l);
    }
}// namespace

TEST_CASE("run returns a trajectory that moves one individual at a time", "[pairwise_comparison][numerical]") {
    egttools::Random::SeedGenerator::getInstance().setMainSeed(3610063510);
    constexpr size_t Z = 30;
    egttools::FinitePopulations::NormalFormGame game(1, prisoners_dilemma());
    egttools::FinitePopulations::PairwiseComparisonNumerical evolver(Z, game, 1000);
    VectorXui init_state(2);
    init_state << 15, 15;

    const auto trajectory = evolver.run(2000, 1.0, 0.01, init_state);
    REQUIRE(trajectory.rows() == 2001);
    REQUIRE(trajectory.cols() == 2);
    CHECK(trajectory.row(0).transpose() == init_state);
    int wrong_total = 0, jumps = 0;
    for (Eigen::Index t = 0; t < trajectory.rows(); ++t) {
        wrong_total += trajectory.row(t).sum() != Z;
        if (t > 0) {
            const auto moved = std::abs(static_cast<long>(trajectory(t, 0)) - static_cast<long>(trajectory(t - 1, 0)));
            jumps += moved > 1;
        }
    }
    CHECK(wrong_total == 0);
    CHECK(jumps == 0);
}

TEST_CASE("without mutation a monomorphic population never changes", "[pairwise_comparison][numerical]") {
    egttools::Random::SeedGenerator::getInstance().setMainSeed(3610063510);
    egttools::FinitePopulations::NormalFormGame game(1, prisoners_dilemma());
    egttools::FinitePopulations::PairwiseComparisonNumerical evolver(20, game, 1000);
    VectorXui init_state(2);
    init_state << 20, 0;
    const auto trajectory = evolver.run(500, 1.0, init_state);
    REQUIRE(trajectory.rows() == 501);
    CHECK((trajectory.col(0).array() == 20).all());
}

TEST_CASE("estimate_fixation_probability agrees with the closed form", "[pairwise_comparison][numerical]") {
    // A single D invading C in a population of 10 with beta = 0.2. Each run is a Bernoulli
    // trial, so the estimate has standard error sqrt(rho (1 - rho) / runs); fixed seed, 5 SE.
    egttools::Random::SeedGenerator::getInstance().setMainSeed(3610063510);
    constexpr long Z = 10;
    constexpr double beta = 0.2;
    constexpr size_t runs = 20000;
    const Matrix2D A = prisoners_dilemma();
    egttools::FinitePopulations::NormalFormGame game(1, A);
    egttools::FinitePopulations::PairwiseComparisonNumerical evolver(Z, game, 1000);

    const double rho = reference::fixation_probability(
            [&A](long i) { return fitness(A, Z, i, 1) - fitness(A, Z, i, 0); }, Z, beta);
    const double standard_error = std::sqrt(rho * (1 - rho) / runs);
    const double estimate = evolver.estimate_fixation_probability(1, 0, runs, 1000000, beta);
    CAPTURE(rho, standard_error);
    CHECK_THAT(estimate, WithinAbs(rho, 5 * standard_error));
}

TEST_CASE("estimate_stationary_distribution approaches the exact distribution", "[pairwise_comparison][numerical]") {
    // Two strategies with mutation: a birth-death chain on k = number of D (state index k), whose
    // stationary distribution follows from detailed balance with the rates of Definition 4.1.
    egttools::Random::SeedGenerator::getInstance().setMainSeed(3610063510);
    constexpr long Z = 10;
    constexpr double beta = 0.5, mu = 0.1;
    const Matrix2D A = prisoners_dilemma();
    egttools::FinitePopulations::NormalFormGame game(1, A);
    egttools::FinitePopulations::PairwiseComparisonNumerical evolver(Z, game, 1000);

    const auto rates = [&A](long k, int from, int to) {
        const std::vector<long> n{Z - k, k};
        const std::vector<double> f{n[0] > 0 ? fitness(A, Z, k, 0) : 0.0, n[1] > 0 ? fitness(A, Z, k, 1) : 0.0};
        return reference::transition(n, f, from, to, beta, mu);
    };
    const auto exact = reference::birth_death_stationary(
            [&rates](long k) { return rates(k, 0, 1); }, [&rates](long k) { return rates(k, 1, 0); }, Z);

    const egttools::Vector estimate = evolver.estimate_stationary_distribution(20, 100000, 1000, beta, mu);
    REQUIRE(estimate.size() == Z + 1);
    CHECK_THAT(estimate.sum(), WithinAbs(1.0, 1e-12));
    CHECK(estimate.minCoeff() >= 0.0);
    // Total variation distance, fixed seed. Measured: 0.002 for both estimators; the exact
    // distribution for a wrong mutation rate (mu = 0.05) is at distance 0.25, so the bound of
    // 0.02 still has power.
    double total_variation = 0.0;
    for (long k = 0; k <= Z; ++k) total_variation += std::abs(estimate(k) - exact[static_cast<size_t>(k)]) / 2;
    CHECK(total_variation < 0.02);

    // The sparse estimator returns the same kind of estimate (as a 1 x nb_states row).
    const auto sparse = evolver.estimate_stationary_distribution_sparse(20, 100000, 1000, beta, mu);
    REQUIRE(sparse.cols() == Z + 1);
    double sparse_variation = 0.0;
    for (long k = 0; k <= Z; ++k)
        sparse_variation += std::abs(sparse.coeff(0, k) - exact[static_cast<size_t>(k)]) / 2;
    CHECK(sparse_variation < 0.02);
}

TEST_CASE("PairwiseComparisonNumerical rejects invalid arguments", "[pairwise_comparison][numerical][validation]") {
    egttools::FinitePopulations::NormalFormGame game(1, prisoners_dilemma());
    egttools::FinitePopulations::PairwiseComparisonNumerical evolver(10, game, 100);
    VectorXui wrong_total(2);
    wrong_total << 4, 4;
    CHECK_THROWS_AS(evolver.estimate_mean_absorption_time(1.0, wrong_total, 10), std::invalid_argument);
    VectorXui state(2);
    state << 5, 5;
    CHECK_THROWS_AS(evolver.estimate_mean_absorption_time(-1.0, state, 10), std::invalid_argument);
    CHECK_THROWS_AS(evolver.estimate_absorption_probabilities(1.0, state, 0), std::invalid_argument);
}

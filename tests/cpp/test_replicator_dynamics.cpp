// Replicator dynamics in infinite populations (infinite_populations/ReplicatorDynamics.hpp):
//   dx_i/dt = x_i (f_i(x) - sum_j x_j f_j(x))   (Taylor & Jonker 1978; Hofbauer & Sigmund 1998).
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <egttools/Types.h>
#include <egttools/finite_populations/Utils.hpp>
#include <egttools/infinite_populations/ReplicatorDynamics.hpp>

using Catch::Matchers::WithinAbs;
using egttools::Matrix2D;
using egttools::Vector;
using egttools::infinite_populations::replicator_equation;
using egttools::infinite_populations::replicator_equation_n_player;

namespace {
    Matrix2D hawk_dove() {
        Matrix2D A(2, 2);
        A << 0.0, 3.0, 1.0, 2.0;
        return A;
    }
}// namespace

TEST_CASE("replicator_equation for a two-player game", "[replicator]") {
    // Hawk-dove at x = (1/4, 3/4): A x = (9/4, 7/4), mean fitness 15/8, so
    // dx/dt = (1/4 (9/4 - 15/8), 3/4 (7/4 - 15/8)) = (3/32, -3/32).
    Vector x(2);
    x << 0.25, 0.75;
    const Vector dx = replicator_equation(x, hawk_dove());
    CHECK_THAT(dx(0), WithinAbs(3.0 / 32, 1e-15));
    CHECK_THAT(dx(1), WithinAbs(-3.0 / 32, 1e-15));
}

TEST_CASE("replicator_equation vanishes at the vertices and the interior equilibrium", "[replicator]") {
    // Hawk-dove: (A x)_0 = (A x)_1 at x_0 = 1/2.
    const Matrix2D A = hawk_dove();
    for (const double x0: {0.0, 0.5, 1.0}) {
        Vector x(2);
        x << x0, 1 - x0;
        CAPTURE(x0);
        CHECK(replicator_equation(x, A).cwiseAbs().maxCoeff() < 1e-15);
    }
}

TEST_CASE("replicator_equation_n_player with linear payoffs matches the two-player game", "[replicator]") {
    // If a focal l-strategist's payoff in a group is the average of A(l, m) over its N - 1
    // co-players, its expected payoff is (A x)_l for any N, so both equations agree.
    // Column i of the table is the group sample_simplex(i, N, S), focal included.
    const Matrix2D A = [] {
        Matrix2D B(3, 3);
        B << 0.0, -1.0, 2.0, 1.5, 0.5, -0.5, -2.0, 3.0, 1.0;
        return B;
    }();
    constexpr size_t N = 4, S = 3;
    const auto nb_groups = egttools::starsBars<size_t, int64_t>(N, S);
    Matrix2D table = Matrix2D::Zero(S, nb_groups);
    for (int64_t i = 0; i < nb_groups; ++i) {
        const auto group = egttools::FinitePopulations::sample_simplex(static_cast<size_t>(i), N, S);
        for (size_t l = 0; l < S; ++l) {
            if (group(l) == 0) continue;
            double total = 0.0;
            for (size_t m = 0; m < S; ++m)
                total += A(l, m) * static_cast<double>(group(m) - (l == m ? 1 : 0));
            table(l, i) = total / static_cast<double>(N - 1);
        }
    }
    for (const auto &frequencies: std::vector<std::vector<double>>{{0.2, 0.3, 0.5}, {0.6, 0.1, 0.3}, {1.0, 0.0, 0.0}}) {
        const Vector x = Eigen::Map<const Vector>(frequencies.data(), 3);
        CHECK((replicator_equation_n_player(x, table, N) - replicator_equation(x, A)).cwiseAbs().maxCoeff() < 1e-14);
    }
}

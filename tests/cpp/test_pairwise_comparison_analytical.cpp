#include <cassert>
#include <cmath>
#include <iostream>

#include <egttools/Types.h>
#include <egttools/finite_populations/Utils.hpp>
#include <egttools/finite_populations/analytical/PairwiseComparison.hpp>
#include <egttools/finite_populations/games/NormalFormGame.h>

namespace {
    constexpr double kTol = 1e-10;

    void check_row_stochastic(const egttools::SparseMatrix2D &matrix) {
        assert(matrix.rows() == matrix.cols());

        for (int row = 0; row < matrix.rows(); ++row) {
            double row_sum = 0.0;
            bool has_diag = false;

            for (egttools::SparseMatrix2D::InnerIterator it(matrix, row); it; ++it) {
                const double value = it.value();
                assert(std::isfinite(value));
                assert(value >= -kTol);
                row_sum += value;
                if (it.col() == row) has_diag = true;
            }

            assert(has_diag);
            assert(std::abs(row_sum - 1.0) < 1e-8);
        }
    }

    void check_dense_probability_matrix(const egttools::Matrix2D &matrix) {
        for (int i = 0; i < matrix.rows(); ++i) {
            double row_sum = 0.0;
            for (int j = 0; j < matrix.cols(); ++j) {
                const double value = matrix(i, j);
                assert(std::isfinite(value));
                assert(value >= -kTol);
                assert(value <= 1.0 + kTol);
                row_sum += value;
            }
            assert(std::abs(row_sum - 1.0) < 1e-8);
        }
    }
}

int main() {
    constexpr int Z = 8;
    constexpr double beta = 1.0;
    constexpr double mu = 0.01;

    egttools::Matrix2D payoff_matrix(2, 2);
    payoff_matrix << 3.0, 0.0,
            5.0, 1.0;

    egttools::FinitePopulations::NormalFormGame game(1, payoff_matrix);

    egttools::FinitePopulations::analytical::PairwiseComparison pc(Z, game, 256);

    assert(pc.population_size() == Z);
    assert(pc.nb_strategies() == 2);
    assert(pc.nb_states() == egttools::starsBars(Z, 2));

    pc.pre_calculate_edge_fitnesses();

    egttools::VectorXui state(2);
    state << 3, 5;

    const auto grad = pc.calculate_gradient_of_selection(beta, state);
    assert(grad.size() == 2);
    assert(std::isfinite(grad(0)));
    assert(std::isfinite(grad(1)));
    assert(std::abs(grad.sum()) < 1e-12);

    // const auto grad_mu = pc.calculate_gradient_of_selection_with_mutation(beta, mu, state);
    // assert(grad_mu.size() == 2);
    // assert(std::isfinite(grad_mu(0)));
    // assert(std::isfinite(grad_mu(1)));
    // assert(std::abs(grad_mu.sum()) < 1e-12);

    const double rho_10 = pc.calculate_fixation_probability(1, 0, beta);
    const double rho_01 = pc.calculate_fixation_probability(0, 1, beta);
    assert(std::isfinite(rho_10));
    assert(std::isfinite(rho_01));
    assert(rho_10 >= 0.0 && rho_10 <= 1.0);
    assert(rho_01 >= 0.0 && rho_01 <= 1.0);

    const auto transition = pc.calculate_transition_matrix(beta, mu);
    assert(transition.rows() == pc.nb_states());
    assert(transition.cols() == pc.nb_states());
    check_row_stochastic(transition);

    const auto [sml_transition, fixation] = pc.calculate_transition_and_fixation_matrix_sml(beta);
    assert(sml_transition.rows() == pc.nb_strategies());
    assert(sml_transition.cols() == pc.nb_strategies());
    assert(fixation.rows() == pc.nb_strategies());
    assert(fixation.cols() == pc.nb_strategies());

    check_dense_probability_matrix(sml_transition);

    for (int i = 0; i < fixation.rows(); ++i) {
        for (int j = 0; j < fixation.cols(); ++j) {
            assert(std::isfinite(fixation(i, j)));
            assert(fixation(i, j) >= -kTol);
            assert(fixation(i, j) <= 1.0 + kTol);
        }
    }

    std::cout << "PairwiseComparison analytical test passed.\n";
    return 0;
}

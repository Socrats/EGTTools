//
// Created by Elias Fernandez on 11/11/2022.
//
#include <egttools/Types.h>

#include <Eigen/Eigenvalues>
#include <egttools/finite_populations/analytical/PairwiseComparison.hpp>
#include <egttools/finite_populations/games/Matrix2PlayerGameHolder.hpp>
#include <egttools/finite_populations/games/MatrixNPlayerGameHolder.hpp>
#include <iostream>
#include <memory>

using namespace std;

int main() {
    egttools::Matrix2D payoff_matrix1(2, 2);
    payoff_matrix1 << -0.5, 2, 0, 0;

    egttools::Matrix2D payoff_matrix2(2, 4);
    payoff_matrix2 << -0.5, 2, 0, 0, 0.5, 2, 1, 0;

    egttools::VectorXui init_state(2);
    init_state << 50, 50;

    // First let's test a 2-player game
    auto game_2player = egttools::FinitePopulations::Matrix2PlayerGameHolder(2, payoff_matrix1);
    auto game_nplayer = egttools::FinitePopulations::MatrixNPlayerGameHolder(2, 3, payoff_matrix2);

    auto evolver = egttools::FinitePopulations::analytical::PairwiseComparison(100, game_2player);

    std::cout << "2-player game ----" << std::endl;
    std::cout << "gradient of selection is = " << evolver.calculate_gradient_of_selection(1, init_state) << std::endl;

    auto evolver2 = egttools::FinitePopulations::analytical::PairwiseComparison(100, game_nplayer);

    std::cout << "N-player game ----" << std::endl;
    std::cout << "gradient of selection is = " << evolver2.calculate_gradient_of_selection(1, init_state) << std::endl;

//    auto evolver3 = egttools::FinitePopulations::analytical::PairwiseComparison(100, egttools::FinitePopulations::MatrixNPlayerGameHolder(2, 3, payoff_matrix2));
//    std::cout << "transition matrix ---" << std::endl;
//    auto transition_matrix = evolver3.calculate_transition_matrix(1, 0.001);
//    std::cout << transition_matrix << std::endl;
//
//    auto evolver4 = egttools::FinitePopulations::analytical::PairwiseComparison(100, egttools::FinitePopulations::Matrix2PlayerGameHolder(2, payoff_matrix1));
//    std::cout << "transition matrix ---" << std::endl;
//    auto transition_matrix2 = evolver3.calculate_transition_matrix(1, 0.001);
//    std::cout << transition_matrix << std::endl;
    //
    //    Eigen::EigenSolver<egttools::Matrix2D> eigensolver;
    //    egttools::Matrix2D transition_matrix_dense(transition_matrix);
    //    eigensolver.compute(transition_matrix_dense);
    //
    //    std::cout << "eigenvalues ---" << std::endl;
    //    std::cout << eigensolver.eigenvalues() << std::endl;
    //
    //    std::cout << "eigenvectors ---" << std::endl;
    //    std::cout << eigensolver.eigenvectors() << std::endl;
};
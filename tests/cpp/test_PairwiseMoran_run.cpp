//
// Created by Elias Fernandez on 2019-02-10.
//
#include <iostream>
#include <egttools/Types.h>
#include <egttools/SeedGenerator.h>
#include <egttools/finite_populations/games/NormalFormGame.h>
#include <egttools/finite_populations/PairwiseMoran.hpp>


using namespace std;

int main() {

    // First we define a vector of possible behaviors
    long nb_strategies = 2;
    size_t pop_size = 100;
    int nb_rounds = 1;
    egttools::VectorXui init_state(nb_strategies);
    egttools::Matrix2D payoff_matrix(2, 2);
    payoff_matrix << -0.5, 2, 0, 0;
    init_state << 50, 50;

    egttools::Random::SeedGenerator::getInstance().setMainSeed(3610063510);

    egttools::FinitePopulations::NormalFormGame game(nb_rounds, payoff_matrix);

    // Initialise selection mutation process
    auto smProcess = egttools::FinitePopulations::PairwiseMoran(pop_size, game, 1000000);

    auto dist = smProcess.run(1000, 1.0, 0.001, init_state);

    assert(dist.rows() == 1001);
    assert(dist.cols() == nb_strategies);

    return 0;
}

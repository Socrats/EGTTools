//
// Created by Elias Fernandez on 2019-02-10.
//
#include <iostream>
#include <egttools/Types.h>
#include <egttools/SeedGenerator.h>
#include <random>
#include <egttools/finite_populations/games/NormalFormGame.h>
#include <egttools/finite_populations/PairwiseMoran.hpp>


using namespace std;

int main() {

    // First we define a vector of possible behaviors
    long nb_strategies = 2;
    int64_t pop_size = 100;
    int nb_rounds = 1;
    egttools::VectorXui init_state(nb_strategies);
    egttools::Matrix2D payoff_matrix(2, 2);
    payoff_matrix << -0.5, 2, 0, 0;
    init_state << 50, 50;
    std::mt19937_64 gen{egttools::Random::SeedGenerator::getInstance().getSeed()};

    egttools::Random::SeedGenerator::getInstance().setMainSeed(3610063510);

    egttools::FinitePopulations::NormalFormGame game(nb_rounds, payoff_matrix);

    // Initialise selection mutation process
    auto smProcess = egttools::FinitePopulations::PairwiseMoran(pop_size, game, 1000);

    // let's run this many times
    size_t nb_runs = 1000;
    auto nb_states = egttools::starsBars<int64_t>(pop_size, nb_strategies);
    auto state_sampler = std::uniform_int_distribution<int64_t>(0, nb_states - 1);
    for (size_t i = 0; i < nb_runs; ++i) {
        auto state_index = state_sampler(gen);
        egttools::FinitePopulations::sample_simplex(state_index, pop_size, nb_strategies, init_state);
        auto dist = smProcess.run(1000, 1.0, 0.001, init_state);
        assert(dist.rows() == 1001);
        assert(dist.cols() == nb_strategies);
    }

//    auto dist = smProcess.run(1000, 1.0, 0.001, init_state);

    return 0;
}

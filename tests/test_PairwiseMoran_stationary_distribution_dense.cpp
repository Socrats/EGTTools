//
// Created by Elias Fernandez on 2019-02-10.
//
#include <iostream>
#include <chrono>
#include <egttools/Types.h>
#include <egttools/SeedGenerator.h>
#include <egttools/finite_populations/Utils.hpp>
#include <egttools/finite_populations/games/NormalFormGame.h>
#include <egttools/finite_populations/behaviors/NFGStrategies.hpp>
#include <egttools/finite_populations/PairwiseMoran.hpp>


using namespace std;
using namespace std::chrono;

int main() {

    // First we define a vector of possible behaviors

    egttools::FinitePopulations::NFGStrategyVector strategies;
    size_t pop_size = 100;
    int nb_rounds = 1;
    egttools::Matrix2D payoff_matrix(2, 2);
    payoff_matrix << -0.5, 2, 0, 0;

    egttools::Random::SeedGenerator::getInstance().setMainSeed(3610063510);

    // Instantiate strategies
    auto cooperator = egttools::FinitePopulations::behaviors::twoActions::Cooperator();
    strategies.push_back(&cooperator);
    auto defector = egttools::FinitePopulations::behaviors::twoActions::Defector();
    strategies.push_back(&defector);
//    auto tft = egttools::FinitePopulations::behaviors::twoActions::TitForTat();
//    strategies.push_back(&tft);
//    auto pavlov = egttools::FinitePopulations::behaviors::twoActions::Pavlov();
//    strategies.push_back(&pavlov);
//    auto randp = egttools::FinitePopulations::behaviors::twoActions::RandomPlayer();
//    strategies.push_back(&randp);
//    auto grim = egttools::FinitePopulations::behaviors::twoActions::GRIM();
//    strategies.push_back(&grim);
//    auto imptft = egttools::FinitePopulations::behaviors::twoActions::ImperfectTFT(0.3);
//    strategies.push_back(&imptft);
//    auto suspicioustft = egttools::FinitePopulations::behaviors::twoActions::SuspiciousTFT();
//    strategies.push_back(&suspicioustft);

    size_t nb_strategies = strategies.size();
    auto nb_states = egttools::starsBars(pop_size, nb_strategies);

    egttools::FinitePopulations::NormalFormGame game(nb_rounds, payoff_matrix, strategies);

    // Initialise selection mutation process
    auto smProcess = egttools::FinitePopulations::PairwiseMoran(pop_size, game, 1000000);

    auto start = high_resolution_clock::now();

    auto dist = smProcess.estimate_stationary_distribution(2, 1000000, 1000, 1, 1e-3);

    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    std::cout << "Time taken by function: "
              << duration.count() << " microseconds" << std::endl;

    assert(dist.rows() == static_cast<long>(nb_states));
    std::cout << "stationary_distribution: " << dist << std::endl;

    std::cout << "nb_strategies: " << nb_strategies << " nb_states: " << nb_states << std::endl;
    std::cout << "strategies: [";

    for (auto const &strategy: strategies) {
        std::cout << strategy->type() << ", ";
    }

    std::cout << "]" << std::endl;
    return 0;
}

//
// Created by Elias Fernandez on 2019-02-10.
//
#include <egttools/SeedGenerator.h>
#include <egttools/finite_populations/behaviors/CRDStrategies.h>

#include <chrono>
#include <egttools/finite_populations/analytical/PairwiseComparison.hpp>
#include <egttools/finite_populations/games/CRDGame.hpp>
#include <iostream>
#include <vector>


using namespace std;
using namespace std::chrono;

size_t myPow(size_t x, size_t p) {
    if (p == 0) return 1;
    if (p == 1) return x;
    return x * myPow(x, p - 1);
}

std::vector<int> combine(std::vector<int> &values, size_t length, size_t iteration) {
    auto output = std::vector<int>(length);
//    size_t max_combinations = myPow(values.size(), length);
//    if (iteration > max_combinations) throw std::invalid_argument(
//            "The size of the population must be a positive integer");

    if (iteration == 0) return output;

    for (size_t i = 1; i <= iteration; ++i) {
        for (size_t j = 0; j < length; ++j) {
            output[j] = values[(i / (myPow(length, j))) % values.size()];
        }
    }

    return output;
}

int main() {
    egttools::Random::SeedGenerator::getInstance().setMainSeed(3610063510);

    // First we define a vector of possible behaviors
    int pop_size = 100;
    int nb_rounds = 10;
    int group_size = 4;
    int endowment = 4 * nb_rounds;
    int target = group_size * nb_rounds * 2;
    int personal_threshold = (group_size - 1) * 2;
    double risk = 1;

    double beta = 1.0;

    std::vector<int> values(3);
    values[0] = 0;
    values[1] = 2;
    values[2] = 4;
    size_t length = 3;

    size_t max_combinations = myPow(values.size(), length);

//    auto result = combine(values, 3, 10);
//    std::cout << "testing combine [";
//    for (auto& el: result) {
//        std::cout << el << ",";
//    }
//    std::cout << "]" << std::endl;

    auto strategies = egttools::FinitePopulations::CRDStrategyVector();

    for (size_t i = 0; i < max_combinations; ++i) {
        auto result = combine(values, 3, 10);
        auto strategy = egttools::FinitePopulations::behaviors::CRD::CRDMemoryOnePlayer(personal_threshold,
                                                                                        2,
                                                                                        result[0],
                                                                                        result[1],
                                                                                        result[2]);
        strategies.push_back(&strategy);
    }

    auto always0 = egttools::FinitePopulations::behaviors::CRD::CRDMemoryOnePlayer(personal_threshold, 0, 0, 0, 0);
    strategies.push_back(&always0);
    auto always4 = egttools::FinitePopulations::behaviors::CRD::CRDMemoryOnePlayer(personal_threshold, 4, 4, 4, 4);
    strategies.push_back(&always4);


    auto game = egttools::FinitePopulations::CRDGame(endowment, target, nb_rounds, group_size, risk, 1.0, strategies);

    //    cout << game.payoffs() << endl;

    // Initialise selection mutation process
    auto evolver = egttools::FinitePopulations::analytical::PairwiseComparison(pop_size, game, 100000);

    auto start = high_resolution_clock::now();

    auto fixation = evolver.calculate_fixation_probability(1, 0, beta);

    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    std::cout << "Time taken by function: "
              << duration.count() << " microseconds" << std::endl;

    std::cout << "Fixation probability of strategy " << strategies[0]->type() << " over strategy " << strategies[1]->type() << " is " << fixation << std::endl;

    /* Second time */

    start = high_resolution_clock::now();

    fixation = evolver.calculate_fixation_probability(1, 0, beta);

    stop = high_resolution_clock::now();
    duration = duration_cast<microseconds>(stop - start);
    std::cout << "Time taken by function: "
              << duration.count() << " microseconds" << std::endl;

    std::cout << "Fixation probability of strategy " << strategies[0]->type() << " over strategy " << strategies[1]->type() << " is " << fixation << std::endl;

    return 0;
}

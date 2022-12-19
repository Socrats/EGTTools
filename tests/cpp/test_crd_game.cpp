//
// Created by Elias Fernandez on 27/07/2021.
//
#include <egttools/finite_populations/behaviors/CRDStrategies.h>

#include <egttools/finite_populations/games/CRDGame.hpp>
#include <iostream>

using namespace std;

int main() {
    int endowment = 40;
    int threshold = 120;
    int nb_rounds = 10;
    int group_size = 6;
    double risk = 0.9;

    auto strategies = egttools::FinitePopulations::CRDStrategyVector();
    auto always0 = egttools::FinitePopulations::behaviors::CRD::CRDMemoryOnePlayer(10, 0, 0, 0, 0);
    strategies.push_back(&always0);
    auto always2 = egttools::FinitePopulations::behaviors::CRD::CRDMemoryOnePlayer(10, 2, 2, 2, 2);
    strategies.push_back(&always2);
    auto always4 = egttools::FinitePopulations::behaviors::CRD::CRDMemoryOnePlayer(10, 4, 4, 4, 4);
    strategies.push_back(&always4);
    auto compensator = egttools::FinitePopulations::behaviors::CRD::CRDMemoryOnePlayer(10, 2, 4, 4, 0);
    strategies.push_back(&compensator);
    auto reciprocal = egttools::FinitePopulations::behaviors::CRD::CRDMemoryOnePlayer(10, 2, 0, 4, 4);
    strategies.push_back(&reciprocal);


    auto game = egttools::FinitePopulations::CRDGame(endowment, threshold, nb_rounds, group_size, risk, 1.0, strategies);

    cout << game.payoffs() << endl;
}
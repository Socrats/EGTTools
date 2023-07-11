//
// Created by Elias Fernandez on 08/01/2023.
//
#include <egttools/Types.h>
#include <egttools/finite_populations/games/NormalFormNetworkGame.h>

#include <egttools/finite_populations/evolvers/GeneralPopulationEvolver.hpp>
#include <egttools/finite_populations/structure/Network.hpp>
#include <iostream>

using NodeDictionary = egttools::FinitePopulations::structure::NodeDictionary;
using NormalFormNetworkGame = egttools::FinitePopulations::games::NormalFormNetworkGame;
using NetworkStructure = egttools::FinitePopulations::structure::Network<NormalFormNetworkGame>;
using Evolver = egttools::FinitePopulations::evolvers::GeneralPopulationEvolver;

int main() {
    double beta = 1, mu = 0;
    egttools::Matrix2D payoff_matrix(2, 2);
    payoff_matrix << 2, -1, 3, 0;

    std::cout << payoff_matrix << std::endl;

    NodeDictionary network;

    network[0] = std::vector<int>{1, 2, 3, 4, 5, 6, 7, 8, 9};
    network[1] = std::vector<int>{0, 2, 3, 4, 5, 6, 7, 8, 9};
    network[2] = std::vector<int>{0, 1, 3, 4, 5, 6, 7, 8, 9};
    network[3] = std::vector<int>{0, 1, 2, 4, 5, 6, 7, 8, 9};
    network[4] = std::vector<int>{0, 1, 2, 3, 5, 6, 7, 8, 9};
    network[5] = std::vector<int>{0, 1, 2, 3, 4, 6, 7, 8, 9};
    network[6] = std::vector<int>{0, 1, 2, 3, 4, 5, 7, 8, 9};
    network[7] = std::vector<int>{0, 1, 2, 3, 4, 5, 6, 8, 9};
    network[8] = std::vector<int>{0, 1, 2, 3, 4, 5, 6, 7, 9};
    network[9] = std::vector<int>{0, 1, 2, 3, 4, 5, 6, 7, 8};


    NormalFormNetworkGame game(1, payoff_matrix);
    NetworkStructure network_structure(2, beta, mu, network, game);

    Evolver evolver(network_structure);

    auto result = evolver.run(1000 + 10000, 10000);


    std::cout << result << std::endl;
}
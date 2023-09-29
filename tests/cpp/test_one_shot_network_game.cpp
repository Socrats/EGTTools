//
// Created by Elias Fernandez on 08/01/2023.
//
#include <egttools/Types.h>
#include <egttools/finite_populations/games/OneShotCRDNetworkGame.hpp>

#include <egttools/finite_populations/evolvers/GeneralPopulationEvolver.hpp>
#include <egttools/finite_populations/structure/Network.hpp>

using NodeDictionary = egttools::FinitePopulations::structure::NodeDictionary;
using OneShotCRDNetworkGame = egttools::FinitePopulations::games::OneShotCRDNetworkGame;
using NetworkStructure = egttools::FinitePopulations::structure::Network<OneShotCRDNetworkGame>;
using Evolver = egttools::FinitePopulations::evolvers::GeneralPopulationEvolver;

int main() {
    double beta = 1, mu = 0.01;

    double endowment = 1;
    double risk = 0;
    double cost = 0.1;
    int min_nb_cooperators = 7;

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


    OneShotCRDNetworkGame game(endowment, cost, risk, min_nb_cooperators);
    NetworkStructure network_structure(2, beta, mu, network, game);

    Evolver evolver(network_structure);

    auto result = evolver.run(100000 + 1000000, 100000);

    std::cout << result << std::endl;
}
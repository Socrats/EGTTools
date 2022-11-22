//
// Created by Elias Fernandez on 27/07/2021.
//
#include <egttools/finite_populations/games/OneShotCRD.hpp>
#include <iostream>

using namespace std;

int main() {
    double endowment = 1.0;
    double cost = 0.1;
    int group_size = 6;
    int min_nb_cooperators = 3;
    double risk = 0.9;


    auto game = egttools::FinitePopulations::OneShotCRD(endowment, cost, risk, group_size, min_nb_cooperators);

    cout << game.payoffs() << endl;
}
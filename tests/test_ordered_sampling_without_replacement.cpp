//
// Created by Elias Fernandez on 12/05/2021.
//
#include <egttools/SeedGenerator.h>
#include <egttools/Types.h>

#include <egttools/finite_populations/Utils.hpp>
#include <iostream>

using namespace std;

int main() {
    long int pop_size = 100;
    long int nb_strategies = 3;
    std::mt19937_64 generator(egttools::Random::SeedGenerator::getInstance().getSeed());
    egttools::VectorXli state = egttools::VectorXli::Zero(nb_strategies);

    // Test 100 values and make sure they always sum to the correct element
    for (int j = 0; j < 100; ++j) {
        egttools::FinitePopulations::sample_simplex_direct_method<long int, long int, egttools::VectorXli, std::mt19937_64>(nb_strategies, pop_size, state, generator);

        std::cout << "[\t";
        for (int i = 0; i < nb_strategies; ++i) {
            std::cout << state(i) << "\t";
        }
        std::cout << "]\t";
        std::cout << "sum = " << state.sum() << std::endl;
//        assert(state.sum() == pop_size);
    }
}
//
// Created by Elias Fernandez on 27/07/2021.
//
#include <egttools/SeedGenerator.h>

#include <cassert>
#include <egttools/utils/TimingUncertainty.hpp>
#include <random>

using namespace std;

int main() {

    std::mt19937_64 generator{egttools::Random::SeedGenerator::getInstance().getSeed()};
    auto tu = egttools::utils::TimingUncertainty(1. / 3);

    [[maybe_unused]] int avg_rounds = 0;

    for (int i = 0; i < 10000; ++i) {
        avg_rounds += tu.calculate_full_end(8, generator);
    }

    assert(avg_rounds / 10000.0 > 10 - 0.05);
    assert(avg_rounds / 10000.0 < 10 + 0.05);
}
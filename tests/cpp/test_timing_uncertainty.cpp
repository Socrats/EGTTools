//
// Created by Elias Fernandez on 27/07/2021.
//
// TimingUncertainty::calculate_full_end(min_rounds) returns min_rounds + G, with G geometric
// (number of failures before the first success, std::geometric_distribution) with parameter p:
// E[G] = (1 - p) / p and SD[G] = sqrt(1 - p) / p. For p = 1/3 and min_rounds = 8 the mean is 10.
//
// The sample mean of n draws is compared with the exact mean within k = 4 standard errors, with a
// fixed seed, so the test is reproducible. (Over random seeds a correct implementation fails about
// 0.1% of the time, measured; the former fixed tolerance of +-0.05, about 2 SE, failed 3.8%.)
// The same check must reject a wrong parameter (p = 0.3, mean 10.33), so the test has power.
#include <cassert>
#include <cmath>
#include <iostream>
#include <random>

#include <egttools/utils/TimingUncertainty.hpp>

namespace {
    double sample_mean(const double p, const int min_rounds, const int nb_samples, std::mt19937_64 &generator) {
        auto tu = egttools::utils::TimingUncertainty(p);
        long long total = 0;
        for (int i = 0; i < nb_samples; ++i)
            total += tu.calculate_full_end(min_rounds, generator);
        return static_cast<double>(total) / nb_samples;
    }
}// namespace

int main() {
    constexpr double p = 1.0 / 3;
    constexpr int min_rounds = 8;
    constexpr int nb_samples = 10000;
    constexpr double k = 4.0;

    const double expected_mean = min_rounds + (1 - p) / p;               // 10
    const double standard_error = std::sqrt(1 - p) / p / std::sqrt(nb_samples);// ~0.0245
    const double tolerance = k * standard_error;                        // ~0.098

    std::mt19937_64 generator{3610063510ULL};
    const double mean = sample_mean(p, min_rounds, nb_samples, generator);
    std::cout << "mean = " << mean << ", expected = " << expected_mean << " +- " << tolerance << '\n';
    assert(std::abs(mean - expected_mean) < tolerance);

    // Power: a wrong parameter must be rejected by the same check.
    const double wrong_mean = sample_mean(0.3, min_rounds, nb_samples, generator);
    assert(std::abs(wrong_mean - expected_mean) > tolerance);

    std::cout << "TimingUncertainty test passed.\n";
    return 0;
}

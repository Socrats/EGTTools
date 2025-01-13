//
// Created by Elias Fernandez on 07/11/2022.
//
#include <algorithm>
#include <iostream>
#include <egttools/Distributions.h>
#include <egttools/Types.h>
#include <chrono>

#include <boost/multiprecision/cpp_int.hpp>

using namespace std;
using namespace boost::multiprecision;


int main() {
    size_t n = 150;
    size_t k = 100;
    uint128_t correct_result = 4167813029162990130;

    // Since C(n, k) = C(n, n-k)
    int128_t n_terms = std::min(k, n - k);

    int128_t m = n + 1;

    int128_t res = 1;

    // Calculate value of [n * (n-1) * ... * (n-k+1)] / [k * (k-1) * ... * 1]
    for (uint128_t i = 1; i < n_terms + 1; ++i) {
        res *= m - i;
        res /= i;
    }

    std::cout << "res = " << res << std::endl;
    std::cout << "correct result = " << correct_result << std::endl;
    std::cout << "result = " << egttools::binomial_precision(n, k) << std::endl;

    // Now let's check the calculation of the multivariate hypergeometric
    double correct_solution1 = 8.772974388570533e-13;
    double correct_solution2 = 6.08110417043232e-08;

    std::vector<size_t> population_counts;
    population_counts.push_back(9);
    population_counts.push_back(150);
    std::vector<size_t> group_counts;
    group_counts.push_back(7);
    group_counts.push_back(2);

    auto start = std::clock();
    auto result_non_optimized = egttools::multivariateHypergeometricPDF(150, 2, 9, group_counts, population_counts);
    // std::cout << correct_solution2 << std::endl;
    auto end = std::clock();
    auto elapsed_non_optimized = static_cast<double>(end - start) / CLOCKS_PER_SEC;
    std::cout << "Result of multivariateHypergeometricPDF " << result_non_optimized << std::endl;
    std::cout << "Elapsed time for multivariateHypergeometricPDF = " << elapsed_non_optimized << std::endl;

    // start = std::clock();
    // auto result_optimized = egttools::multivariateHypergeometricPDF_optimized(150, 2, 9, group_counts, population_counts);
    // // std::cout << correct_solution1 << std::endl;
    // end = std::clock();
    // auto elapsed_optimized = static_cast<double>(end - start) / CLOCKS_PER_SEC;
    // std::cout << "Result of multivariateHypergeometricPDF_optimized " << result_optimized << std::endl;
    // std::cout << "Elapsed time for multivariateHypergeometricPDF_optimized = " << elapsed_optimized << std::endl;
    //
    // std::cout << "ratio = " << result_optimized / result_non_optimized << std::endl;
    // std::cout << "ratio_time = " << elapsed_optimized / elapsed_non_optimized << std::endl;

    // egttools::Vector example = egttools::Vector::Zero(3);

    // egttools::Vector new_example(example);
    //
    // example(1) = 3;
    //
    // std::cout << "original = " << example << std::endl;
    // std::cout << "new = " << new_example << std::endl;
    //
    // new_example = example;
    //
    // std::cout << "copy = " << new_example << std::endl;
    //
    // example(2) = 1;
    //
    // std::cout << "original = " << example << std::endl;
    // std::cout << "new = " << new_example << std::endl;
}

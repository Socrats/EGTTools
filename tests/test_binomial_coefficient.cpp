//
// Created by Elias Fernandez on 07/11/2022.
//
#include <algorithm>
#include <cassert>
#include <iostream>
//#include <boost/multiprecison/cpp_int.hpp>
#include <egttools/Distributions.h>
#include <egttools/Types.h>

#include <boost/multiprecision/cpp_int.hpp>

using namespace std;
using namespace boost::multiprecision;

double multivariateHypergeometricPDF(size_t m, size_t k, size_t n, const std::vector<size_t> &sample_counts,
                                     const std::vector<size_t> &population_counts) {

    uint128_t res = 1;
    // First we calculate the number of unordered samples of size n chosen from the population
    auto denominator = egttools::binomial_precision(m, n);

    // Then we calculate the multiplication of the number of all unordered subsets of a subset of the population
    // with only 1 type of object
    for (size_t i = 0; i < k; ++i) {
        res *= egttools::binomial_precision(population_counts[i], sample_counts[i]);
    }

    return static_cast<double>(res) / static_cast<double>(denominator);
}

int main() {
    size_t n = 490;
    size_t k = 481;
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
    population_counts.push_back(490);
    std::vector<size_t> group_counts;
    group_counts.push_back(7);
    group_counts.push_back(2);

    std::cout << multivariateHypergeometricPDF(499, 2, 9, group_counts, population_counts) << std::endl;
    std::cout << correct_solution1 << std::endl;

    std::vector<size_t> group_counts2;
    group_counts2.push_back(5);
    group_counts2.push_back(4);

    std::cout << multivariateHypergeometricPDF(499, 2, 9, group_counts2, population_counts) << std::endl;
    std::cout << correct_solution2 << std::endl;

    egttools::Vector example = egttools::Vector::Zero(3);

    egttools::Vector new_example(example);

    example(1) = 3;

    std::cout << "original = " << example << std::endl;
    std::cout << "new = " << new_example << std::endl;

    new_example = example;

    std::cout << "copy = " << new_example << std::endl;

    example(2) = 1;

    std::cout << "original = " << example << std::endl;
    std::cout << "new = " << new_example << std::endl;

}
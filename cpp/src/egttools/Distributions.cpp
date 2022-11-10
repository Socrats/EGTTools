/** Copyright (c) 2019-2021  Elias Fernandez
  *
  * This file is part of EGTtools.
  *
  * EGTtools is free software: you can redistribute it and/or modify
  * it under the terms of the GNU General Public License as published by
  * the Free Software Foundation, either version 3 of the License, or
  * (at your option) any later version.
  *
  * EGTtools is distributed in the hope that it will be useful,
  * but WITHOUT ANY WARRANTY; without even the implied warranty of
  * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  * GNU General Public License for more details.
  *
  * You should have received a copy of the GNU General Public License
  * along with EGTtools.  If not, see <http://www.gnu.org/licenses/>
*/

#include <egttools/Distributions.h>

//template<>
//size_t egttools::binomialCoeff<size_t, size_t>(size_t n, size_t k) {
//    if ((k > n) || (n == ULONG_MAX))
//        return 0;
//
//    // Since C(n, k) = C(n, n-k)
//    size_t n_terms = std::min(k, n - k);
//
//    size_t res = 1;
//    size_t m = n + 1;
//
//    // Calculate value of [n * (n-1) * ... * (n-k+1)] / [k * (k-1) * ... * 1]
//    for (size_t i = 1; i < n_terms + 1; ++i) {
//        res *= m - i;
//        res /= i;
//    }
//
//    return res;
//}

double
egttools::multivariateHypergeometricPDF(size_t m, size_t k, size_t n, const std::vector<size_t> &sample_counts,
                                        const std::vector<size_t> &population_counts) {

    double res = 1;
    // First we calculate the number of unordered samples of size n chosen from the population
    auto denominator = egttools::binomialCoeff<double, size_t>(m, n);

    // Then we calculate the multiplication of the number of all unordered subsets of a subset of the population
    // with only 1 type of object
    for (size_t i = 0; i < k; ++i) {
        res *= egttools::binomialCoeff<double, size_t>(population_counts[i], sample_counts[i]);
    }

    return res / denominator;
}

double
egttools::multivariateHypergeometricPDF(size_t m, size_t k, size_t n, const std::vector<size_t> &sample_counts,
                                        const Eigen::Ref<const VectorXui> &population_counts) {

    double res = 1;
    // First we calculate the number of unordered samples of size n chosen from the population
    auto denominator = egttools::binomialCoeff<double, size_t>(m, n);

    // Then we calculate the multiplication of the number of all unordered subsets of a subset of the population
    // with only 1 type of object
    for (signed long i = 0; i < static_cast<signed long>(k); ++i) {
        res *= egttools::binomialCoeff<double, size_t>(population_counts(i), sample_counts[i]);
    }

    return res / denominator;
}

double
egttools::multivariateHypergeometricPDF(size_t m, size_t k, size_t n, const Eigen::Ref<const VectorXui> &sample_counts,
                                        const Eigen::Ref<const VectorXui> &population_counts) {

    double res = 1;
    // First we calculate the number of unordered samples of size n chosen from the population
    auto denominator = egttools::binomialCoeff<double, size_t>(m, n);

    // Then we calculate the multiplication of the number of all unordered subsets of a subset of the population
    // with only 1 type of object
    for (signed long i = 0; i < static_cast<signed long>(k); ++i) {
        res *= egttools::binomialCoeff<double, size_t>(population_counts(i), sample_counts(i));
    }

    return res / denominator;
}
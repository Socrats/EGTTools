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

#if (HAS_BOOST)
#define uint_type_ mp::uint128_t
#define binomial_coeff_ binomial_precision
#else
#define uint_type_ size_t
#define binomial_coeff_ egttools::binomialCoeff<double, size_t>
#endif

template<>
size_t egttools::binomialCoeff<size_t, size_t>(size_t n, size_t k) {
    if ((k > n) || (n == ULONG_MAX))
        return 0;

    // Since C(n, k) = C(n, n-k)
    size_t n_terms = std::min(k, n - k);

    size_t res = 1;
    size_t m = n + 1;

    // Calculate value of [n * (n-1) * ... * (n-k+1)] / [k * (k-1) * ... * 1]
    for (size_t i = 1; i < n_terms + 1; ++i) {
        res *= m - i;
        res /= i;
    }

    return res;
}

#if (HAS_BOOST)
mp::uint128_t egttools::binomial_precision(size_t n, size_t k) {
    if ((k > n) || (n == ULONG_MAX))
        return 0;

    // Since C(n, k) = C(n, n-k)
    size_t n_terms = std::min(k, n - k);

    mp::uint128_t res = 1;
    size_t m = n + 1;

    // Calculate value of [n * (n-1) * ... * (n-k+1)] / [k * (k-1) * ... * 1]
    for (mp::uint128_t i = 1; i < n_terms + 1; ++i) {
        res *= m - i;
        res /= i;
    }

    return res;
}
#endif

double
egttools::multivariateHypergeometricPDF(size_t m, size_t k, size_t n, const std::vector<size_t> &sample_counts,
                                        const std::vector<size_t> &population_counts) {

    uint_type_ res = 1;
    // First we calculate the number of unordered samples of size n chosen from the population
    auto denominator = binomial_coeff_(m, n);

    // Then we calculate the multiplication of the number of all unordered subsets of a subset of the population
    // with only 1 type of object
    for (size_t i = 0; i < k; ++i) {
        res *= binomial_coeff_(population_counts[i], sample_counts[i]);
    }

    return static_cast<double>(res) / static_cast<double>(denominator);
}

double
egttools::multivariateHypergeometricPDF(size_t m, size_t k, size_t n, const std::vector<size_t> &sample_counts,
                                        const Eigen::Ref<const VectorXui> &population_counts) {

    uint_type_ res = 1;
    // First we calculate the number of unordered samples of size n chosen from the population
    auto denominator = binomial_coeff_(m, n);

    // Then we calculate the multiplication of the number of all unordered subsets of a subset of the population
    // with only 1 type of object
    for (signed long i = 0; i < static_cast<signed long>(k); ++i) {
        res *= binomial_coeff_(population_counts(i), sample_counts[i]);
    }

    return static_cast<double>(res) / static_cast<double>(denominator);
}

double
egttools::multivariateHypergeometricPDF(size_t m, size_t k, size_t n, const Eigen::Ref<const VectorXui> &sample_counts,
                                        const Eigen::Ref<const VectorXui> &population_counts) {

    uint_type_ res = 1;
    // First we calculate the number of unordered samples of size n chosen from the population
    auto denominator = binomial_coeff_(m, n);

    // Then we calculate the multiplication of the number of all unordered subsets of a subset of the population
    // with only 1 type of object
    for (signed long i = 0; i < static_cast<signed long>(k); ++i) {
        res *= binomial_coeff_(population_counts(i), sample_counts(i));
    }

    return static_cast<double>(res) / static_cast<double>(denominator);
}

#if (HAS_BOOST)
template<>
mp::uint128_t egttools::starsBars<size_t, mp::uint128_t>(size_t stars, size_t bins) {
    return egttools::binomial_precision(stars + bins - 1, stars);
}
#endif
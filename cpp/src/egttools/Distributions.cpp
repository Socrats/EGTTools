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

#include <limits>
#include <stdexcept>
#include <string>

uint_type_ egttools::binomial_precision(size_t n, size_t k) {
    if (k > n)
        return 0;

    // Since C(n, k) = C(n, n-k)
    const size_t n_terms = std::min(k, n - k);

    // After step i, res = C(n, i + 1), so every division is exact. No n + 1 is formed,
    // so n = SIZE_MAX (e.g. a negative int converted to size_t) cannot wrap around.
    //
    // Fast path: fixed-width 128-bit arithmetic while res < 2^64. Both res and n - i are
    // then below 2^64, so res * (n - i) < 2^128 cannot overflow.
    uint_type_ res = 1;
    const uint_type_ fast_limit = uint_type_(1) << 64;
    size_t i = 0;
    for (; i < n_terms && res < fast_limit; ++i) {
        res *= n - i;
        res /= i + 1;
    }
    if (i == n_terms)
        return res;

    // Results of 2^64 or more: finish in arbitrary precision. C(n, i) increases for
    // i <= n / 2, so once it exceeds the 128-bit maximum the result does too: throw
    // instead of wrapping around. As C(n, i) >= 2^i here, this loop runs at most ~129 times.
    const boost::multiprecision::cpp_int max_value(std::numeric_limits<uint_type_>::max());
    boost::multiprecision::cpp_int big(res);
    for (; i < n_terms; ++i) {
        big *= n - i;
        big /= i + 1;
        if (big > max_value)
            throw std::overflow_error("binomial_precision: C(" + std::to_string(n) + ", " + std::to_string(k) +
                                      ") does not fit in a 128-bit unsigned integer");
    }

    return static_cast<uint_type_>(big);
}

double
egttools::multivariateHypergeometricPDF(const size_t m, const size_t k, const size_t n,
                                                  const std::vector<size_t> &sample_counts,
                                                  const std::vector<size_t> &population_counts) {
    double res = 0;
    // First we calculate the number of unordered samples of size n chosen from the population
    auto denominator = log_binomial_coefficient(m, n);

    // Then we calculate the multiplication of the number of all unordered subsets of a subset of the population
    // with only 1 type of object
    for (size_t i = 0; i < k; ++i) {
        if (sample_counts[i] > population_counts[i])
            return 0;
        res += log_binomial_coefficient(population_counts[i], sample_counts[i]);
    }
    res -= denominator;
    res = exp(res);

    return res;
}

double
egttools::multivariateHypergeometricPDF(size_t m, size_t k, size_t n, const std::vector<size_t> &sample_counts,
                                        const Eigen::Ref<const VectorXui> &population_counts) {
    double res = 0;
    // First we calculate the number of unordered samples of size n chosen from the population
    auto denominator = log_binomial_coefficient(m, n);

    // Then we calculate the multiplication of the number of all unordered subsets of a subset of the population
    // with only 1 type of object
    for (signed long i = 0; i < static_cast<signed long>(k); ++i) {
        if (sample_counts[i] > population_counts[i])
            return 0;
        res += log_binomial_coefficient(population_counts(i), sample_counts[i]);
    }
    res -= denominator;
    res = exp(res);

    return res;
}

double
egttools::multivariateHypergeometricPDF(size_t m, size_t k, size_t n, const Eigen::Ref<const VectorXui> &sample_counts,
                                        const Eigen::Ref<const VectorXui> &population_counts) {
    double res = 0;
    // First we calculate the number of unordered samples of size n chosen from the population
    auto denominator = log_binomial_coefficient(m, n);

    // Then we calculate the multiplication of the number of all unordered subsets of a subset of the population
    // with only 1 type of object
    for (signed long i = 0; i < static_cast<signed long>(k); ++i) {
        if (sample_counts[i] > population_counts[i])
            return 0;
        res += log_binomial_coefficient(population_counts(i), sample_counts(i));
    }
    res -= denominator;
    res = exp(res);

    return res;
}

double
egttools::multivariateHypergeometricPDF(const double log_denominator, const size_t k,
                                        const std::vector<size_t> &sample_counts,
                                        const Eigen::Ref<const VectorXui> &population_counts) {
    // Fast-path: caller has precomputed log_binomial_coefficient(m, n) outside the hot loop,
    // saving 3 ln_factorial lookups per call relative to the standard overloads.
    double res = -log_denominator;
    for (size_t i = 0; i < k; ++i) {
        if (sample_counts[i] > static_cast<size_t>(population_counts(static_cast<signed long>(i))))
            return 0.0;
        res += log_binomial_coefficient(population_counts(static_cast<signed long>(i)), sample_counts[i]);
    }
    return exp(res);
}

double egttools::multinomialPMF(const Eigen::Ref<const VectorXui> &group_configuration, size_t n,
                                const Eigen::Ref<const Vector> &p) {
    if (n == ULONG_MAX)
        return 0;
    // This would be an error, and probably better to throw an exception here!
    if (group_configuration.size() != p.size())
        throw std::invalid_argument("Arguments p and group configuration must have the same length!");

    int_type_ n_factorial = egttools::math::factorial(n);
    float_type_ prob = 1;

    // Iterate through the possible group configurations and
    // calculate the multiplications
    for (signed long i = 0; i < group_configuration.size(); ++i) {
        for (size_t j = 1; j < group_configuration(i) + 1; ++j) {
            prob *= p(i) / static_cast<int>(j);
        }
    }
    return (prob * n_factorial.convert_to<float_type_>()).convert_to<double>();
}

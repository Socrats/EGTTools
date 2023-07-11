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

//
// Adapted from https://github.com/Svalorzen/AI-Toolbox/
//
#pragma once
#ifndef EGTTOOLS_DISTRIBUTIONS_H
#define EGTTOOLS_DISTRIBUTIONS_H

#include <egttools/Types.h>

#include <algorithm>
#include <egttools/math.hpp>
#include <random>
#include <stdexcept>

#if (HAS_BOOST)
#include <boost/multiprecision/cpp_dec_float.hpp>
#include <boost/multiprecision/cpp_int.hpp>

#define int_type_ boost::multiprecision::cpp_int
#define uint_type_ boost::multiprecision::uint128_t
#define float_type_ boost::multiprecision::cpp_dec_float_100
#define binomial_coeff_ binomial_precision
#define CONVERT_TO_(object, type) object.convert_to<type>()
#else
#define int_type_ int64_t
#define uint_type_ size_t
#define float_type_ double
#define binomial_coeff_ egttools::binomialCoeff<double, size_t>
#define CONVERT_TO_(object, type) static_cast<type>(object)
#endif

namespace egttools {
    /**
     * @brief This function samples an index from a probability vector.
     *
     * @tparam T Type of vector container
     * @tparam G Type of random number generator
     * @param d vector size
     * @param in probability vector
     * @param generator random number generator
     * @return An index in range [0, d-1].
     */
    template<typename T, typename G>
    size_t choice(const size_t d, const T &in, G &generator) {
        std::uniform_real_distribution<double> probabilityDistribution(0.0, 1.0);
        double p = probabilityDistribution(generator);
        double container;

        for (size_t i = 0; i < d; ++i) {
            container = in[i];
            if (container > p) return i;
            p -= container;
        }
        return d - 1;
    }

    /**
     * @brief This function samples an index from a probability vector.
     *
     * @tparam T Type of vector container
     * @tparam G Type of random number generator
     * @param d vector size
     * @param in probability vector
     * @param dist uniform distribution [0, 1)
     * @param generator random number generator
     * @return An index in range [0, d-1].
     */
    template<typename T, typename G>
    size_t choice(const size_t d, const T &in, std::uniform_real_distribution<double> &dist, G &generator) {
        double p = dist(generator);
        double container;

        for (size_t i = 0; i < d; ++i) {
            container = in[i];
            if (container > p) return i;
            p -= container;
        }
        return d - 1;
    }

    /**
     * @brief This function samples and index from a sparse probability vector.
     *
     * This function randomly samples an index between 0 and d, given a vector
     * containing the probabilities of sampling each of the indexes.
     *
     * @tparam G
     * @param d
     * @param in
     * @param generator
     * @return
     */
    template<typename G>
    size_t choice(const size_t d, const SparseMatrix2D::ConstRowXpr &in, G &generator) {
        std::uniform_real_distribution<double> probabilityDistribution(0.0, 1.0);
        double p = probabilityDistribution(generator);

        for (SparseMatrix2D::ConstRowXpr::InnerIterator i(in, 0);; ++i) {
            if (i.value() > p) return i.col();
            p -= i.value();
        }
        return d - 1;
    }

    /**
    * @brief Calculates the binomial coefficient C(n, k)
    *
    * @tparam T : Output type
    * @tparam I : Input type
    * @param n size of the fixed set
    * @param k size of the unordered subset
    * @return C(n, k)
    */
    template<typename Output, typename Input>
    Output binomialCoeff(Input n, Input k) {
        if ((k > n) || (n < 0) || (k < 0))
            return 0;

        // Since C(n, k) = C(n, n-k)
        if (k > n - k) k = n - k;

        Output res = 1;

        // Calculate value of [n * (n-1) * ... * (n-k+1)] / [k * (k-1) * ... * 1]
        for (Input i = 0; i < k; ++i) {
            res *= static_cast<Output>(n - i);
            res /= static_cast<Output>(i + 1);
        }

        return res;
    }

#if (HAS_BOOST)
    uint_type_ binomial_precision(size_t n, size_t k);
#endif

    /**
     * @brief Calculates the probability density function of a multivariate hypergeometric distribution.
     *
     * This function returns the probability that a sample of size @param n in a population of @param k
     * objects will have @param sample_counts counts of each object in a sample, given a population D
     * with @param population_counts counts of each object.
     *
     * The sampling is without replacement.
     *
     * @param m size of the population
     * @param k number of objects in the population
     * @param n size of the sample
     * @param sample_counts a vector containing the counts of each objects in the sample
     * @param population_counts a vector containing the counts of each objects in the population
     * @return probability of a sample occurring in the population.
     */
    double
    multivariateHypergeometricPDF(size_t m, size_t k, size_t n, const std::vector<size_t> &sample_counts,
                                  const std::vector<size_t> &population_counts);

    /**
    * @brief Calculates the probability density function of a multivariate hypergeometric distribution.
    *
    * This function returns the probability that a sample of size @param n in a population of @param k
    * objects will have @param sample_counts counts of each object in a sample, given a population D
    * with @param population_counts counts of each object.
    *
    * The sampling is without replacement.
    *
    * @param m size of the population
    * @param k number of objects in the population
    * @param n size of the sample
    * @param sample_counts a vector containing the counts of each objects in the sample
    * @param population_counts a vector containing the counts of each objects in the population
    * @return probability of a sample occurring in the population.
    */
    double
    multivariateHypergeometricPDF(size_t m, size_t k, size_t n, const std::vector<size_t> &sample_counts,
                                  const Eigen::Ref<const VectorXui> &population_counts);

    /**
    * @brief Calculates the probability density function of a multivariate hypergeometric distribution.
    *
    * This function returns the probability that a sample of size @param n in a population of @param k
    * objects will have @param sample_counts counts of each object in a sample, given a population D
    * with @param population_counts counts of each object.
    *
    * The sampling is without replacement.
    *
    * @param m size of the population
    * @param k number of objects in the population
    * @param n size of the sample
    * @param sample_counts a vector containing the counts of each objects in the sample
    * @param population_counts a vector containing the counts of each objects in the population
    * @return probability of a sample occurring in the population.
    */
    double
    multivariateHypergeometricPDF(size_t m, size_t k, size_t n, const Eigen::Ref<const VectorXui> &sample_counts,
                                  const Eigen::Ref<const VectorXui> &population_counts);

    /**
     * Calculates the Probability Mass Function of a multinomial distribution
     *
     * @param group_configuration
     * @param n
     * @param p
     * @throws invalid_argument
     * @return the probability of a specific group configuration occurring.
     */
    double multinomialPMF(const Eigen::Ref<const VectorXui> &group_configuration, size_t n, const Eigen::Ref<const Vector> &p);

    /**
     * @brief Finds the number for elements given possible bins/slots and star types.
     *
     * @tparam T : Type of to use for the computation
     * @param stars : number of elements to fill the bins
     * @param bins : number of bins that can be filled
     * @return the number of possible combinations of stars in the bins.
     */
    template<typename T, typename O = T>
    O starsBars(T stars, T bins) {
        return egttools::binomialCoeff<O, T>(stars + bins - 1, stars);
    }

#if (HAS_BOOST)
    template<>
    inline uint_type_ starsBars<size_t, uint_type_>(size_t stars, size_t bins) {
        return egttools::binomial_precision(stars + bins - 1, stars);
    }
#endif

}// namespace egttools

#endif//EGTTOOLS_DISTRIBUTIONS_H

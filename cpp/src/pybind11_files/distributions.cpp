/** Copyright (c) 2022-2023  Elias Fernandez
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
#include "distributions.hpp"

void init_distributions(py::module_ &mDistributions) {
    mDistributions.attr("__init__") = py::str(
            "The `egttools.numerical.distributions` submodule contains "
            "functions and classes that produce stochastic distributions.");

    py::class_<egttools::utils::TimingUncertainty<>>(mDistributions, "TimingUncertainty")
            .def(py::init<double, int>(),
                 R"pbdoc(
                    Timing uncertainty distribution container.

                    This class provides methods to calculate the final round of the game according to some predifined distribution, which is geometric by default.

                    Parameters
                    ----------
                    p : float
                        Probability that the game will end after the minimum number of rounds.
                    max_rounds : int
                        maximum number of rounds that the game can take (if 0, there is no maximum).
                    )pbdoc",
                 py::arg("p"), py::arg("max_rounds") = 0)
            .def("calculate_end", &egttools::utils::TimingUncertainty<>::calculate_end,
                 "Calculates the final round limiting by max_rounds, i.e., outputs a value between"
                 "[min_rounds, max_rounds].",
                 py::arg("min_rounds"), py::arg("random_generator"))
            .def("calculate_full_end", &egttools::utils::TimingUncertainty<>::calculate_full_end,
                 "Calculates the final round, i.e., outputs a value between"
                 "[min_rounds, Inf].",
                 py::arg("min_rounds"), py::arg("random_generator"))
            .def_property_readonly("p", &egttools::utils::TimingUncertainty<>::probability)
            .def_property("max_rounds", &egttools::utils::TimingUncertainty<>::max_rounds,
                          &egttools::utils::TimingUncertainty<>::set_max_rounds);

    mDistributions.def("multinomial_pmf", &egttools::multinomialPMF,
                       R"pbdoc(
                                Calculates the probability density function of a multivariate hyper-geometric distribution.

                                This function returns the probability that a sample of size
                                :param n with counts of each type indicated by :param x
                                would be drawn from a population with frequencies :param p.

                                Both :param population_counts and :param sample_counts must be of shape
                                (k,), where k is the number of types of `objects` in the population.

                                For the application often used in this library, :param n would be the size of the group,
                                :param k would be the number of strategies, :param x would be group configuration and
                                :param p would be the state of the population (in infinite populations).

                                Parameters
                                ----------
                                x : numpy.ndarray
                                    Vector of containing the counts of each element that should be drawn.
                                    Must sum to n.
                                n : int
                                    Total number of elements to draw
                                p : numpy.ndarray
                                    Vector indicating the total frequency of each element. Must sum to 1.

                                Returns
                                -------
                                float
                                    The probability that a sample of size n with counts x of each type is
                                    draw from a population with total frequencies per type defined by p.

                                See Also
                                --------
                                egttools.distributions.multivariate_hypergeometric_pdf
                                egttools.distributions.binom
                                egttools.distributions.comb
                        )pbdoc", py::arg("x"), py::arg("n"), py::arg("p")
                       );

    mDistributions.def("multivariate_hypergeometric_pdf",
                       static_cast<double (*)(size_t, size_t, size_t, const std::vector<size_t> &,
                                              const Eigen::Ref<const egttools::VectorXui> &)>(&egttools::multivariateHypergeometricPDF),
                       R"pbdoc(
                                Calculates the probability density function of a multivariate hyper-geometric distribution.

                                This function returns the probability that a sample :param sample_counts
                                would be drawn from a population :param population_counts. Assuming that
                                the population is of size :param m, has :param k objects, and the sample
                                has size :param n.

                                Both :param population_counts and :param sample_counts must be of shape
                                (k,). The sum of all entries in :param population_counts,
                                must sum to :param m, and the sum of all entries in :param sample_counts
                                must sum to :param n.

                                For the application often used in this library, :param m would be the size of the population,
                                :param k would be the number of strategies, :param n would be the group size, :param sample_counts
                                would contain the counts of each strategy in the group, and :param population_counts contains the
                                counts of each strategy in the population.

                                Parameters
                                ----------
                                m : int
                                    size of the population
                                k : int
                                    number of objects in the population
                                n : int
                                    size of the sample
                                sample_counts : List[int]
                                    a vector containing the counts of each objects in the sample
                                population_counts : numpy.ndarray
                                    a vector containing the counts of each objects in the population

                                Returns
                                -------
                                float
                                    The probability that a sample of size n in a population of k objects

                                See Also
                                --------
                                egttools.distributions.binom
                                egttools.distributions.comb
                        )pbdoc",
                       py::arg("m"),
                       py::arg("k"),
                       py::arg("n"),
                       py::arg("sample_counts"),
                       py::arg("population_counts"));

    mDistributions.def("multivariate_hypergeometric_pdf",
                       static_cast<double (*)(size_t, size_t, size_t, const Eigen::Ref<const egttools::VectorXui> &,
                                              const Eigen::Ref<const egttools::VectorXui> &)>(&egttools::multivariateHypergeometricPDF),
                       R"pbdoc(
                                Calculates the probability density function of a multivariate hyper-geometric distribution.

                                This function returns the probability that a sample :param sample_counts
                                would be drawn from a population :param population_counts. Assuming that
                                the population is of size :param m, has :param k objects, and the sample
                                has size :param n.

                                Both :param population_counts and :param sample_counts must be of shape
                                (k,). The sum of all entries in :param population_counts,
                                must sum to :param m, and the sum of all entries in :param sample_counts
                                must sum to :param n.

                                For the application often used in this library, :param m would be the size of the population,
                                :param k would be the number of strategies, :param n would be the group size, :param sample_counts
                                would contain the counts of each strategy in the group, and :param population_counts contains the
                                counts of each strategy in the population.

                                Parameters
                                ----------
                                m : int
                                    size of the population
                                k : int
                                    number of objects in the population
                                n : int
                                    size of the sample
                                sample_counts : List[int]
                                    a vector containing the counts of each objects in the sample
                                population_counts : numpy.ndarray
                                    a vector containing the counts of each objects in the population

                                Returns
                                -------
                                float
                                    The probability that a sample of size n in a population of k objects

                                See Also
                                --------
                                egttools.distributions.binom
                                egttools.distributions.comb
                        )pbdoc",
                       py::arg("m"),
                       py::arg("k"),
                       py::arg("n"),
                       py::arg("sample_counts"),
                       py::arg("population_counts"));

    mDistributions.def("binom",
                       &egttools::binomialCoeff<double, int64_t>,
                       R"pbdoc(
                                Calculates the binomial coefficient C(n, k).

                                This method is approximate and will return a float value.
                                The result should be equivalent to the one produced by
                                `scipy.special.binom`.

                                Parameters
                                ----------
                                n : int
                                    size of the fixed set
                                k : int
                                    size of the unordered subset

                                Returns
                                -------
                                float
                                    The binomial coefficient C(n, k).

                                See Also
                                --------
                                egttools.distributions.multivariate_hypergeometric_pdf
                                egttools.distributions.comb
                        )pbdoc",
                       py::arg("n"),
                       py::arg("k"));

#if (HAS_BOOST)
    mDistributions.def(
            "comb", [](size_t n, size_t k) {
                auto result = egttools::binomialCoeff<boost::multiprecision::cpp_int, boost::multiprecision::cpp_int>(n, k);
                return py::cast(result);
            },
            R"pbdoc(
                    Calculates the binomial coefficient C(n, k).

                    The number of combinations of :param n things taken :param k at a time.
                    This is often expressed as "N choose k".

                    This method is exact and should be equivalent `scipy.special.comb`.
                    However, if the outcome or any intermediary product occupies more than
                    an uint128_t, the result will not be correct, since there will be
                    an overflow!

                    Parameters
                    ----------
                    n : int
                        size of the fixed set
                    k : int
                        size of the unordered subset

                    Returns
                    -------
                    int
                        The binomial coefficient C(n, k).

                    See Also
                    --------
                    egttools.distributions.multivariate_hypergeometric_pdf
                    egttools.distributions.binom
            )pbdoc",
            py::arg("n"), py::arg("k"));
#else
    mDistributions.def("comb",
                       &egttools::binomialCoeff<size_t, size_t>,
                       R"pbdoc(
                                Calculates the binomial coefficient C(n, k).

                                The number of combinations of :param n things taken :param k at a time.
                                This is often expressed as "N choose k".

                                This method is exact and should be equivalent `scipy.special.comb`.
                                However, if the outcome or any intermediary product occupies more than
                                an uint64_t, the result will not be correct, since there will be
                                an overflow!

                                Parameters
                                ----------
                                n : int
                                    size of the fixed set
                                k : int
                                    size of the unordered subset

                                Returns
                                -------
                                int
                                    The binomial coefficient C(n, k).

                                See Also
                                --------
                                egttools.distributions.multivariate_hypergeometric_pdf
                                egttools.distributions.binom
                                )pbdoc",
                       py::arg("n"),
                       py::arg("k"));
#endif
}
/** Copyright (c) 2022-2026  Elias Fernandez
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
    mDistributions.doc() =
            "The `egttools.numerical.distributions` submodule contains functions and classes "
            "for probability distributions and timing uncertainty."; {
        py::options options;
        options.disable_function_signatures();

        py::class_<egttools::utils::TimingUncertainty<> >(
                    mDistributions,
                    "TimingUncertainty",
                    R"pbdoc(
                Timing uncertainty distribution container.

                This class provides methods to sample the final round of a game.
                By default, the timing uncertainty follows a geometric distribution.

                Parameters
                ----------
                p : float
                    Probability that the game ends after the minimum number of rounds.
                max_rounds : int, optional
                    Maximum number of rounds allowed. If 0, no maximum is enforced.

                Examples
                --------
                >>> from egttools.numerical.distributions import TimingUncertainty
                >>> tu = TimingUncertainty(0.2, 20)
            )pbdoc"
                )
                .def(
                    py::init<double, int>(),
                    py::arg("p"),
                    py::arg("max_rounds") = 0
                )
                .def(
                    "calculate_end",
                    &egttools::utils::TimingUncertainty<>::calculate_end,
                    R"pbdoc(
                    Sample the final round, truncated by `max_rounds`.

                    The returned round lies in the interval
                    `[min_rounds, max_rounds]` when `max_rounds > 0`.

                    Parameters
                    ----------
                    min_rounds : int
                        Minimum number of rounds.
                    random_generator : object
                        Random number generator used internally.

                    Returns
                    -------
                    int
                        Sampled final round.
                )pbdoc",
                    py::arg("min_rounds"),
                    py::arg("random_generator")
                )
                .def(
                    "calculate_full_end",
                    &egttools::utils::TimingUncertainty<>::calculate_full_end,
                    R"pbdoc(
                    Sample the final round without truncation.

                    The returned round is at least `min_rounds`.

                    Parameters
                    ----------
                    min_rounds : int
                        Minimum number of rounds.
                    random_generator : object
                        Random number generator used internally.

                    Returns
                    -------
                    int
                        Sampled final round.
                )pbdoc",
                    py::arg("min_rounds"),
                    py::arg("random_generator")
                )
                .def_property_readonly(
                    "p",
                    &egttools::utils::TimingUncertainty<>::probability,
                    "Probability that the game ends after the minimum number of rounds."
                )
                .def_property(
                    "max_rounds",
                    &egttools::utils::TimingUncertainty<>::max_rounds,
                    &egttools::utils::TimingUncertainty<>::set_max_rounds,
                    "Maximum allowed number of rounds. A value of 0 means no maximum."
                );
    }

    mDistributions.def(
        "multinomial_pmf",
        &egttools::multinomialPMF,
        R"pbdoc(
            Calculate the probability mass function of a multinomial distribution.

            This function returns the probability of drawing counts `x` in a sample
            of size `n`, given category probabilities `p`.

            Parameters
            ----------
            x : numpy.ndarray
                Counts for each category in the sample. Must sum to `n`.
            n : int
                Total number of draws.
            p : numpy.ndarray
                Category probabilities. Must sum to 1.

            Returns
            -------
            float
                Probability of observing the counts `x`.

            See Also
            --------
            egttools.distributions.multivariate_hypergeometric_pdf
            egttools.distributions.binom
            egttools.distributions.comb

            Examples
            --------
            >>> import numpy as np
            >>> from egttools.numerical.distributions import multinomial_pmf
            >>> multinomial_pmf(np.array([2, 1]), 3, np.array([0.5, 0.5]))
        )pbdoc",
        py::arg("x"),
        py::arg("n"),
        py::arg("p")
    );

    mDistributions.def(
        "multivariate_hypergeometric_pdf",
        static_cast<double (*)(
            size_t,
            size_t,
            size_t,
            const std::vector<size_t> &,
            const Eigen::Ref<const egttools::VectorXui> &
        )>(&egttools::multivariateHypergeometricPDF),
        R"pbdoc(
            Calculate the probability mass function of a multivariate hypergeometric distribution.

            This function returns the probability of observing `sample_counts` when drawing
            a sample of size `n` from a population of size `m`.

            Parameters
            ----------
            m : int
                Population size.
            k : int
                Number of categories in the population.
            n : int
                Sample size.
            sample_counts : list[int]
                Counts for each category in the sample. Must sum to `n`.
            population_counts : numpy.ndarray
                Counts for each category in the population. Must sum to `m`.

            Returns
            -------
            float
                Probability of observing `sample_counts`.

            See Also
            --------
            egttools.distributions.binom
            egttools.distributions.comb
        )pbdoc",
        py::arg("m"),
        py::arg("k"),
        py::arg("n"),
        py::arg("sample_counts"),
        py::arg("population_counts")
    );

    mDistributions.def(
        "multivariate_hypergeometric_pdf",
        static_cast<double (*)(
            size_t,
            size_t,
            size_t,
            const Eigen::Ref<const egttools::VectorXui> &,
            const Eigen::Ref<const egttools::VectorXui> &
        )>(&egttools::multivariateHypergeometricPDF),
        R"pbdoc(
            Calculate the probability mass function of a multivariate hypergeometric distribution.

            This function returns the probability of observing `sample_counts` when drawing
            a sample of size `n` from a population of size `m`.

            Parameters
            ----------
            m : int
                Population size.
            k : int
                Number of categories in the population.
            n : int
                Sample size.
            sample_counts : numpy.ndarray
                Counts for each category in the sample. Must sum to `n`.
            population_counts : numpy.ndarray
                Counts for each category in the population. Must sum to `m`.

            Returns
            -------
            float
                Probability of observing `sample_counts`.

            See Also
            --------
            egttools.distributions.binom
            egttools.distributions.comb
        )pbdoc",
        py::arg("m"),
        py::arg("k"),
        py::arg("n"),
        py::arg("sample_counts"),
        py::arg("population_counts")
    );

    mDistributions.def(
        "binom",
        &egttools::binomialCoeff<double, int64_t>,
        R"pbdoc(
            Calculate the binomial coefficient C(n, k).

            This implementation returns a floating-point approximation and should
            be equivalent to `scipy.special.binom`.

            Parameters
            ----------
            n : int
                Size of the full set.
            k : int
                Size of the subset.

            Returns
            -------
            float
                Binomial coefficient C(n, k).

            See Also
            --------
            egttools.distributions.multivariate_hypergeometric_pdf
            egttools.distributions.comb
        )pbdoc",
        py::arg("n"),
        py::arg("k")
    );

#if (HAS_BOOST)
    mDistributions.def(
        "comb",
        [](const size_t n, const size_t k) {
            auto result = egttools::binomialCoeff<boost::multiprecision::cpp_int, size_t>(n, k);
            return py::cast(result);
        },
        R"pbdoc(
            Calculate the binomial coefficient C(n, k).

            This implementation returns the exact result using multiprecision integers.

            Parameters
            ----------
            n : int
                Size of the full set.
            k : int
                Size of the subset.

            Returns
            -------
            int
                Binomial coefficient C(n, k).

            See Also
            --------
            egttools.distributions.multivariate_hypergeometric_pdf
            egttools.distributions.binom
        )pbdoc",
        py::arg("n"),
        py::arg("k")
    );
#else
    mDistributions.def(
        "comb",
        &egttools::binomialCoeff<size_t, size_t>,
        R"pbdoc(
            Calculate the binomial coefficient C(n, k).

            This implementation is exact only while intermediate results fit in `uint64_t`.

            Parameters
            ----------
            n : int
                Size of the full set.
            k : int
                Size of the subset.

            Returns
            -------
            int
                Binomial coefficient C(n, k).

            See Also
            --------
            egttools.distributions.multivariate_hypergeometric_pdf
            egttools.distributions.binom
        )pbdoc",
        py::arg("n"),
        py::arg("k")
    );
#endif
}

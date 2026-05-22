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

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <egttools/finite_populations/evolvers/MLS.hpp>

namespace py = pybind11;
using namespace egttools::FinitePopulations;

using MLSTraulsenType = MLS<Group>;
using MLSGarciaType = MLS<GarciaGroup>;

void init_evolvers(py::module_ &m) {
    // -------------------------------------------------------------------------
    // MLSTraulsen — Traulsen & Nowak (2006) multi-level selection
    // -------------------------------------------------------------------------
    py::class_<MLSTraulsenType>(m, "MLSTraulsen",
        R"pbdoc(
        Multi-level selection following Traulsen & Nowak (2006).

        A population of m groups each of maximum size n. In each step an
        individual is selected proportional to its fitness and reproduces; if
        the group exceeds n it either splits (probability q) or ejects a random
        member (probability 1-q). Migration probability lambda moves offspring to
        a randomly chosen other group.

        Parameters
        ----------
        nb_generations : int
            Maximum number of Moran steps per run.
        nb_strategies : int
            Number of distinct strategies.
        group_size : int
            Maximum group capacity n (must be >= 4).
        nb_groups : int
            Number of groups m.
        w : float
            Intensity of selection.
        strategies_freq : np.ndarray
            Initial frequency of each strategy (must sum to 1).
        payoff_matrix : np.ndarray
            Square (nb_strategies x nb_strategies) payoff matrix.
        )pbdoc")
        .def(py::init<size_t, size_t, size_t, size_t, double,
                      const Eigen::Ref<const egttools::Vector> &,
                      const Eigen::Ref<const egttools::Matrix2D> &>(),
             py::arg("nb_generations"),
             py::arg("nb_strategies"),
             py::arg("group_size"),
             py::arg("nb_groups"),
             py::arg("w"),
             py::arg("strategies_freq"),
             py::arg("payoff_matrix"))
        .def("evolve",
             py::overload_cast<double, double,
                               const Eigen::Ref<const egttools::VectorXui> &>(
                 &MLSTraulsenType::evolve),
             py::arg("q"), py::arg("w"), py::arg("init_state"),
             py::call_guard<py::gil_scoped_release>(),
             "Run one Moran simulation and return the final strategy frequencies.")
        .def("evolve",
             py::overload_cast<double, double, double,
                               const Eigen::Ref<const egttools::VectorXui> &>(
                 &MLSTraulsenType::evolve),
             py::arg("q"), py::arg("w"), py::arg("lambda"), py::arg("init_state"),
             py::call_guard<py::gil_scoped_release>(),
             "Run one Moran simulation with migration and return the final strategy frequencies.")
        .def("fixation_probability",
             py::overload_cast<size_t, size_t, size_t, double, double>(
                 &MLSTraulsenType::fixationProbability),
             py::arg("invader"), py::arg("resident"), py::arg("nb_runs"),
             py::arg("q"), py::arg("w"),
             py::call_guard<py::gil_scoped_release>(),
             "Estimate fixation probability of invader over resident (no migration).")
        .def("fixation_probability",
             py::overload_cast<size_t, size_t, size_t, double, double, double>(
                 &MLSTraulsenType::fixationProbability),
             py::arg("invader"), py::arg("resident"), py::arg("nb_runs"),
             py::arg("q"), py::arg("lambda"), py::arg("w"),
             py::call_guard<py::gil_scoped_release>(),
             "Estimate fixation probability of invader over resident (with migration).")
        .def("gradient_of_selection",
             py::overload_cast<size_t, size_t, size_t, double, double>(
                 &MLSTraulsenType::gradientOfSelection),
             py::arg("invader"), py::arg("resident"), py::arg("nb_runs"),
             py::arg("w"), py::arg("q") = 0.0,
             py::call_guard<py::gil_scoped_release>(),
             "Estimate gradient of selection (T+ - T-) for each population configuration.")
        .def_property("generations",
                      &MLSTraulsenType::generations,
                      &MLSTraulsenType::set_generations)
        .def_property("group_size",
                      &MLSTraulsenType::group_size,
                      &MLSTraulsenType::set_group_size)
        .def_property("nb_groups",
                      &MLSTraulsenType::nb_groups,
                      &MLSTraulsenType::set_nb_groups)
        .def_property("selection_intensity",
                      &MLSTraulsenType::selection_intensity,
                      &MLSTraulsenType::set_selection_intensity)
        .def_property_readonly("nb_strategies", &MLSTraulsenType::nb_strategies)
        .def_property_readonly("max_pop_size", &MLSTraulsenType::max_pop_size)
        .def_property_readonly("payoff_matrix", &MLSTraulsenType::payoff_matrix)
        .def("__repr__", &MLSTraulsenType::toString);

    // -------------------------------------------------------------------------
    // MLSGarcia — Garcia & van den Bergh (2011) MLS with group conflict
    // -------------------------------------------------------------------------
    py::class_<MLSGarciaType>(m, "MLSGarcia",
        R"pbdoc(
        Multi-level selection following Garcia & van den Bergh (2011).

        Extends the Traulsen & Nowak model with direct group conflict (kappa)
        and separate in-group / out-group payoff matrices (alpha weighting).

        Parameters
        ----------
        nb_generations : int
            Maximum number of Moran steps per run.
        nb_strategies : int
            Number of distinct strategies.
        group_size : int
            Maximum group capacity n (must be >= 4).
        nb_groups : int
            Number of groups m.
        )pbdoc")
        .def(py::init<size_t, size_t, size_t, size_t>(),
             py::arg("nb_generations"),
             py::arg("nb_strategies"),
             py::arg("group_size"),
             py::arg("nb_groups"))
        .def("fixation_probability",
             &MLSGarciaType::fixationProbability,
             py::arg("invader"), py::arg("resident"), py::arg("nb_runs"),
             py::arg("q"), py::arg("lambda"), py::arg("w"),
             py::arg("alpha"), py::arg("kappa"), py::arg("z"),
             py::arg("payoff_matrix_in"), py::arg("payoff_matrix_out"),
             py::call_guard<py::gil_scoped_release>(),
             R"pbdoc(
             Estimate fixation probability of invader over resident.

             Parameters
             ----------
             invader : int
             resident : int
             nb_runs : int
             q : float      splitting probability
             lambda : float  migration probability
             w : float       intensity of selection
             alpha : float   fraction of interactions within the group
             kappa : float   average fraction of groups involved in conflict
             z : float       importance of payoffs in conflict (0 = deterministic)
             payoff_matrix_in : np.ndarray   (nb_strategies x nb_strategies) in-group payoff
             payoff_matrix_out : np.ndarray  (nb_strategies x nb_strategies) out-group payoff
             )pbdoc")
        .def_property("generations",
                      &MLSGarciaType::generations,
                      &MLSGarciaType::set_generations)
        .def_property("group_size",
                      &MLSGarciaType::group_size,
                      &MLSGarciaType::set_group_size)
        .def_property("nb_groups",
                      &MLSGarciaType::nb_groups,
                      &MLSGarciaType::set_nb_groups)
        .def_property_readonly("nb_strategies", &MLSGarciaType::nb_strategies)
        .def_property_readonly("max_pop_size", &MLSGarciaType::max_pop_size)
        .def("__repr__", &MLSGarciaType::toString);
}

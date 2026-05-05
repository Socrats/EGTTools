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

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

#include <egttools/mpi/PairwisePetscOperator.hpp>
#include <egttools/finite_populations/games/AbstractGame.hpp>

namespace py = pybind11;
using egttools::mpi::PairwisePetscOperator;
using egttools::FinitePopulations::AbstractGame;

void init_mpi_operators(py::module_ &m) {

    py::class_<PairwisePetscOperator>(
        m,
        "PairwisePetscOperator",
        R"pbdoc(
MPI-distributed matrix-free transition operator backed by PETSc MatShell + SLEPc EPS.

Computes the stationary distribution of the pairwise comparison process for
population sizes where the full sparse transition matrix would exceed available RAM,
by distributing the computation across MPI ranks.

This class is the MPI counterpart of PairwiseComparisonTransitionOperator.
It exposes y = P^T x through a PETSc MatShell so that SLEPc's Krylov–Schur
eigensolver can find π (eigenvalue = 1) without Python callbacks per Krylov step.

Usage
-----
Construct directly from Python (single-rank or multi-rank via mpiexec)::

    from egttools.numerical.mpi import PairwisePetscOperator
    op = PairwisePetscOperator(population_size=50, game=game, beta=1.0, mu=0.01)
    pi = op.compute_stationary_distribution()

For multi-rank execution::

    mpiexec -n 4 python my_script.py

When to use
-----------
Use this class when ``n = C(Z+k-1, k-1)`` states exceed single-node RAM:
  - k=3 strategies: n > ~1 GB CSR at Z ≈ 4766 (n ≈ 11.4 M)
  - k=4 strategies: n > ~1 GB CSR at Z ≈ 333  (n ≈ 6.3 M)

For smaller state spaces, prefer:
  - ``PairwiseComparisonTransitionOperator.compute_stationary_distribution``  (power iter)
  - ``PairwiseComparisonTransitionOperator.compute_stationary_arpack``         (ARPACK)
  - ``stationary_distribution_from_sparse``                                    (scipy eigs)
        )pbdoc")

        .def(py::init<size_t, AbstractGame &, double, double>(),
             py::arg("population_size"),
             py::arg("game"),
             py::arg("beta"),
             py::arg("mu"),
             py::keep_alive<1, 3>(),  // keep game alive as long as operator is alive
             R"pbdoc(
Construct the PETSc-backed transition operator.

Parameters
----------
population_size : int
    Number of individuals Z (must be >= 2).
game : AbstractGame
    Game defining fitness.  The caller must keep this object alive as long as
    the operator is in use.
beta : float
    Fermi selection intensity (must be >= 0).
mu : float
    Mutation probability per step (must be in [0, 1]).
             )pbdoc")

        .def("compute_stationary_distribution",
             [](PairwisePetscOperator &self, double tol, int max_iter) {
                 py::gil_scoped_release rel;
                 return self.compute_stationary_distribution(tol, max_iter);
             },
             py::arg("tol")      = 1e-12,
             py::arg("max_iter") = 300,
             R"pbdoc(
Compute the stationary distribution via SLEPc Krylov–Schur (pure C++, no Python callbacks).

Parameters
----------
tol : float
    Convergence tolerance for SLEPc EPS (default 1e-12).
max_iter : int
    Maximum number of EPS iterations (default 300).

Returns
-------
numpy.ndarray
    Normalised stationary distribution of length ``size()``.

Raises
------
RuntimeError
    If SLEPc EPS does not converge within *max_iter* iterations.
             )pbdoc")

        .def_property_readonly("size",
             &PairwisePetscOperator::size,
             "Total number of states n = C(Z+k-1, k-1).")

        .def_property_readonly("population_size",
             &PairwisePetscOperator::population_size,
             "Population size Z.")

        .def_property_readonly("nb_strategies",
             &PairwisePetscOperator::nb_strategies,
             "Number of strategies k.")

        .def_property_readonly("beta",
             &PairwisePetscOperator::beta,
             "Fermi selection intensity.")

        .def_property_readonly("mu",
             &PairwisePetscOperator::mu,
             "Mutation probability.");
}

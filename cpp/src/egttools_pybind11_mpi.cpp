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

namespace py = pybind11;

void init_mpi_operators(py::module_ &);

PYBIND11_MODULE(numerical_mpi_, m) {
    m.doc() =
        "MPI-distributed stationary distribution solvers backed by PETSc/SLEPc.\n\n"
        "This module is only present in EGTtools builds compiled with\n"
        "EGTTOOLS_ENABLE_PETSC=ON.  Import it via egttools.numerical.mpi.\n";

    init_mpi_operators(m);
}

"""MPI-distributed stationary distribution solvers backed by PETSc/SLEPc.

This subpackage is only functional when EGTtools was compiled with
``EGTTOOLS_ENABLE_PETSC=ON`` (i.e. the ``numerical_mpi_`` C++ extension module
is present).  Importing it without that module is safe — the subpackage simply
exports nothing.

Typical use
-----------
>>> from egttools.numerical.mpi import PairwisePetscOperator
>>> op = PairwisePetscOperator(population_size=100, game=game, beta=1.0, mu=0.01)
>>> pi = op.compute_stationary_distribution()

Multi-rank via mpiexec
----------------------
.. code-block:: bash

    mpiexec -n 4 python my_script.py

See ``PairwisePetscOperator`` for when to use this class vs the serial solvers.
"""
from __future__ import annotations

try:
    from egttools.numerical.numerical_mpi_ import PairwisePetscOperator
    __all__ = ["PairwisePetscOperator"]
except ImportError:
    # numerical_mpi_ is only present in PETSC-enabled wheel builds.
    __all__: list[str] = []

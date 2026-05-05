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

// Include standard C/C++ headers before PETSc to avoid LDBL_MAX / assert conflicts.
#include <cassert>
#include <cfloat>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include <egttools/mpi/PairwisePetscOperator.hpp>
#include <egttools/Distributions.h>

#include <mpi.h>

#include <stdexcept>
#include <string>
#include <vector>

namespace egttools::mpi {

PairwisePetscOperator::PairwisePetscOperator(
    const size_t                              population_size,
    egttools::FinitePopulations::AbstractGame &game,
    const double                              beta,
    const double                              mu)
    : pop_size_(population_size),
      nb_strategies_(game.nb_strategies()),
      game_(&game),
      beta_(beta),
      mu_(mu) {

    if (population_size < 2)
        throw std::invalid_argument("population_size must be >= 2");
    if (game.nb_strategies() < 2)
        throw std::invalid_argument("game must have at least 2 strategies");
    if (beta < 0.0)
        throw std::invalid_argument("beta must be >= 0");
    if (mu < 0.0 || mu > 1.0)
        throw std::invalid_argument("mu must be in [0, 1]");

    // Initialise SLEPc (which in turn initialises PETSc) if not already done.
    PetscBool initialized = PETSC_FALSE;
    PetscInitialized(&initialized);
    if (!initialized)
        SlepcInitializeNoArguments();

    nb_states_ = static_cast<PetscInt>(
        egttools::starsBars(population_size, game.nb_strategies()));

    inv_Z_              = 1.0 / static_cast<double>(pop_size_);
    inv_Zm1_            = 1.0 / static_cast<double>(pop_size_ - 1);
    one_minus_mu_       = 1.0 - mu_;
    mutation_probability_ = (nb_strategies_ > 2)
                                ? mu_ / static_cast<double>(nb_strategies_ - 1)
                                : mu_;

    // Create n×n MatShell; PETSc distributes rows evenly across ranks.
    PetscCallAbort(PETSC_COMM_WORLD,
        MatCreateShell(PETSC_COMM_WORLD,
                       PETSC_DECIDE, PETSC_DECIDE,
                       nb_states_,   nb_states_,
                       this, &shell_));
    PetscCallAbort(PETSC_COMM_WORLD,
        MatShellSetOperation(shell_, MATOP_MULT,
                             reinterpret_cast<void(*)()>(
                                 PairwisePetscOperator::matvec)));

    // Record this rank's row ownership range.
    PetscCallAbort(PETSC_COMM_WORLD,
        MatGetOwnershipRange(shell_, &lo_, &hi_));
}

PairwisePetscOperator::~PairwisePetscOperator() {
    MatDestroy(&shell_);
}

// ---------------------------------------------------------------------------
// matvec: y = P^T x  (PETSc MatShell MATOP_MULT callback)
// ---------------------------------------------------------------------------

PetscErrorCode PairwisePetscOperator::matvec(Mat A, Vec x, Vec y) {
    PetscFunctionBeginUser;

    PairwisePetscOperator *op;
    PetscCall(MatShellGetContext(A, &op));

    const PetscInt n = op->nb_states_;

    // ------------------------------------------------------------------
    // Step 1: Gather full x on every rank via MPI_Allgatherv.
    // ------------------------------------------------------------------
    PetscInt local_x_size;
    PetscCall(VecGetLocalSize(x, &local_x_size));

    PetscMPIInt comm_size;
    MPI_Comm_size(PETSC_COMM_WORLD, &comm_size);

    std::vector<int> rcounts(comm_size), displs(comm_size);
    MPI_Allgather(
        &local_x_size, 1, MPI_INT,
        rcounts.data(), 1, MPI_INT,
        PETSC_COMM_WORLD);
    displs[0] = 0;
    for (int r = 1; r < comm_size; ++r)
        displs[r] = displs[r - 1] + rcounts[r - 1];

    std::vector<double> x_full(static_cast<size_t>(n));
    const PetscScalar  *x_local;
    PetscCall(VecGetArrayRead(x, &x_local));
    MPI_Allgatherv(
        x_local, static_cast<int>(local_x_size), MPI_DOUBLE,
        x_full.data(), rcounts.data(), displs.data(), MPI_DOUBLE,
        PETSC_COMM_WORLD);
    PetscCall(VecRestoreArrayRead(x, &x_local));

    // ------------------------------------------------------------------
    // Step 2: Enumerate source states owned by this rank; accumulate.
    // ------------------------------------------------------------------
    std::vector<double> y_partial(static_cast<size_t>(n), 0.0);
    egttools::VectorXui current(static_cast<int64_t>(op->nb_strategies_));

    for (PetscInt src = op->lo_; src < op->hi_; ++src) {
        egttools::FinitePopulations::sample_simplex(
            static_cast<size_t>(src), op->pop_size_, op->nb_strategies_, current);

        const double xi           = x_full[static_cast<size_t>(src)];
        const double total_offdiag = op->enumerate_local_(src, current, x_full.data(), y_partial.data());
        y_partial[static_cast<size_t>(src)] += (1.0 - total_offdiag) * xi;
    }

    // ------------------------------------------------------------------
    // Step 3: Sum contributions across all ranks.
    // ------------------------------------------------------------------
    MPI_Allreduce(
        MPI_IN_PLACE, y_partial.data(), static_cast<int>(n),
        MPI_DOUBLE, MPI_SUM, PETSC_COMM_WORLD);

    // ------------------------------------------------------------------
    // Step 4: Write local slice into y.
    // ------------------------------------------------------------------
    PetscScalar *y_arr;
    PetscCall(VecGetArray(y, &y_arr));
    for (PetscInt i = op->lo_; i < op->hi_; ++i)
        y_arr[i - op->lo_] = y_partial[static_cast<size_t>(i)];
    PetscCall(VecRestoreArray(y, &y_arr));

    PetscFunctionReturn(PETSC_SUCCESS);
}

// ---------------------------------------------------------------------------
// enumerate_local_: accumulate P^T contributions from one source state
// ---------------------------------------------------------------------------

double PairwisePetscOperator::enumerate_local_(
    PetscInt             src,
    egttools::VectorXui &current,
    const double        *x_full,
    double              *y_partial) const {

    const int    k  = static_cast<int>(nb_strategies_);
    const size_t Z  = pop_size_;
    const double xi = x_full[static_cast<size_t>(src)];

    std::vector<int>    present;
    std::vector<double> fitness(k, 0.0);
    present.reserve(k);

    egttools::VectorXui temp(current);

    for (int i = 0; i < k; ++i) {
        if (current(i) > 0) {
            present.push_back(i);
            temp(i) -= 1;
            fitness[i] = game_->calculate_fitness(i, Z, temp);
            temp(i) += 1;
        }
    }

    double total_offdiag = 0.0;

    if (present.size() == 1) {
        const int mono = present[0];
        temp(mono) -= 1;
        for (int j = 0; j < k; ++j) {
            if (j == mono) continue;
            temp(j) += 1;
            const auto dest = static_cast<size_t>(
                egttools::FinitePopulations::calculate_state(Z, temp));
            y_partial[dest] += mutation_probability_ * xi;
            total_offdiag   += mutation_probability_;
            temp(j) -= 1;
        }
        temp(mono) += 1;
    } else {
        for (int i = 0; i < k; ++i) {
            temp(i) += 1;

            if (current(i) == 0) {
                for (const int j : present) {
                    if (j == i) continue;
                    temp(j) -= 1;
                    const auto   dest = static_cast<size_t>(
                        egttools::FinitePopulations::calculate_state(Z, temp));
                    const double prob =
                        static_cast<double>(current(j)) * inv_Z_ * mutation_probability_;
                    if (prob > 0.0) {
                        y_partial[dest] += prob * xi;
                        total_offdiag   += prob;
                    }
                    temp(j) += 1;
                }
            } else {
                const double sel_prefactor =
                    one_minus_mu_ * static_cast<double>(current(i)) * inv_Zm1_;

                for (const int j : present) {
                    if (j == i) continue;
                    temp(j) -= 1;
                    const auto   dest = static_cast<size_t>(
                        egttools::FinitePopulations::calculate_state(Z, temp));
                    const double sel_prob =
                        sel_prefactor *
                        egttools::FinitePopulations::fermi(beta_, fitness[j], fitness[i]);
                    const double prob =
                        static_cast<double>(current(j)) * inv_Z_ *
                        (sel_prob + mutation_probability_);
                    if (prob > 0.0) {
                        y_partial[dest] += prob * xi;
                        total_offdiag   += prob;
                    }
                    temp(j) += 1;
                }
            }

            temp(i) -= 1;
        }
    }

    return total_offdiag;
}

// ---------------------------------------------------------------------------
// compute_stationary_distribution: SLEPc Krylov–Schur EPS solve
// ---------------------------------------------------------------------------

egttools::Vector PairwisePetscOperator::compute_stationary_distribution(
    double tol, int max_iter) {

    EPS eps;
    PetscCallAbort(PETSC_COMM_WORLD, EPSCreate(PETSC_COMM_WORLD, &eps));
    PetscCallAbort(PETSC_COMM_WORLD, EPSSetOperators(eps, shell_, NULL));
    PetscCallAbort(PETSC_COMM_WORLD, EPSSetProblemType(eps, EPS_NHEP));
    PetscCallAbort(PETSC_COMM_WORLD, EPSSetWhichEigenpairs(eps, EPS_LARGEST_MAGNITUDE));
    PetscCallAbort(PETSC_COMM_WORLD, EPSSetDimensions(eps, 1, PETSC_DEFAULT, PETSC_DEFAULT));
    PetscCallAbort(PETSC_COMM_WORLD,
        EPSSetTolerances(eps, tol, static_cast<PetscInt>(max_iter)));
    PetscCallAbort(PETSC_COMM_WORLD, EPSSolve(eps));

    PetscInt nconv;
    PetscCallAbort(PETSC_COMM_WORLD, EPSGetConverged(eps, &nconv));
    if (nconv < 1) {
        EPSDestroy(&eps);
        throw std::runtime_error(
            "PairwisePetscOperator::compute_stationary_distribution: "
            "SLEPc EPS did not converge");
    }

    // Retrieve the leading eigenvector (distributed across ranks).
    Vec pi_vec;
    PetscCallAbort(PETSC_COMM_WORLD, MatCreateVecs(shell_, NULL, &pi_vec));
    PetscScalar lambda_r, lambda_i;
    PetscCallAbort(PETSC_COMM_WORLD,
        EPSGetEigenpair(eps, 0, &lambda_r, &lambda_i, pi_vec, NULL));

    // ------------------------------------------------------------------
    // Gather full eigenvector on every rank via MPI_Allgatherv.
    // ------------------------------------------------------------------
    PetscInt local_size;
    PetscCallAbort(PETSC_COMM_WORLD, VecGetLocalSize(pi_vec, &local_size));

    PetscMPIInt comm_size;
    MPI_Comm_size(PETSC_COMM_WORLD, &comm_size);
    std::vector<int> rcounts(comm_size), displs(comm_size);
    MPI_Allgather(
        &local_size, 1, MPI_INT,
        rcounts.data(), 1, MPI_INT,
        PETSC_COMM_WORLD);
    displs[0] = 0;
    for (int r = 1; r < comm_size; ++r)
        displs[r] = displs[r - 1] + rcounts[r - 1];

    std::vector<double> pi_full(static_cast<size_t>(nb_states_));
    const PetscScalar  *pi_local;
    PetscCallAbort(PETSC_COMM_WORLD, VecGetArrayRead(pi_vec, &pi_local));
    MPI_Allgatherv(
        pi_local, static_cast<int>(local_size), MPI_DOUBLE,
        pi_full.data(), rcounts.data(), displs.data(), MPI_DOUBLE,
        PETSC_COMM_WORLD);
    PetscCallAbort(PETSC_COMM_WORLD, VecRestoreArrayRead(pi_vec, &pi_local));

    PetscCallAbort(PETSC_COMM_WORLD, VecDestroy(&pi_vec));
    PetscCallAbort(PETSC_COMM_WORLD, EPSDestroy(&eps));

    egttools::Vector result =
        Eigen::Map<const egttools::Vector>(pi_full.data(), nb_states_);
    result = result.cwiseAbs();
    result /= result.sum();
    return result;
}

} // namespace egttools::mpi

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
#pragma once
#ifndef EGTTOOLS_MPI_PAIRWISEPETSCOPERATOR_HPP
#define EGTTOOLS_MPI_PAIRWISEPETSCOPERATOR_HPP

// Standard C/C++ headers must come before PETSc to avoid conflicts with
// LDBL_MAX, assert, and other identifiers that PETSc headers shadow.
#include <cassert>
#include <cfloat>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include <petscmat.h>
#include <slepceps.h>

#include <egttools/Types.h>
#include <egttools/finite_populations/Utils.hpp>
#include <egttools/finite_populations/games/AbstractGame.hpp>

namespace egttools::mpi {

/**
 * @brief MPI-distributed matrix-free transition operator backed by PETSc MatShell + SLEPc EPS.
 *
 * Exposes y = P^T x through a PETSc MatShell so that SLEPc's Krylov–Schur eigensolver
 * can find the stationary distribution π (leading eigenvector of P^T, eigenvalue = 1)
 * without constructing the full n×n transition matrix.
 *
 * MPI communication pattern (y = P^T x):
 *  1. MPI_Allgatherv  — each rank receives the full x vector.
 *  2. Each rank enumerates source states [lo, hi) and accumulates contributions
 *     to y_partial[0..n-1] for ALL destination indices.
 *  3. MPI_Allreduce   — sum y_partial across ranks.
 *  4. Each rank writes its local slice of y_partial into the distributed y Vec.
 *
 * Communication cost: O(n) per matvec — suitable for up to ~hundreds of MPI ranks.
 */
class PairwisePetscOperator {
public:
    /**
     * @brief Construct the PETSc-backed transition operator.
     *
     * Calls SlepcInitializeNoArguments() if PETSc/SLEPc are not yet initialised.
     * Creates a PETSc MatShell of global size n×n and distributes rows evenly
     * across MPI ranks, where n = C(Z+k-1, k-1).
     *
     * @param population_size  Number of individuals Z (must be >= 2).
     * @param game             Game defining fitness; kept alive by the caller.
     * @param beta             Fermi selection intensity (must be >= 0).
     * @param mu               Mutation probability per step (must be in [0, 1]).
     */
    PairwisePetscOperator(size_t population_size,
                          egttools::FinitePopulations::AbstractGame &game,
                          double beta,
                          double mu);

    ~PairwisePetscOperator();

    // Non-copyable, non-movable (owns PETSc objects with raw handles).
    PairwisePetscOperator(const PairwisePetscOperator &)            = delete;
    PairwisePetscOperator &operator=(const PairwisePetscOperator &) = delete;
    PairwisePetscOperator(PairwisePetscOperator &&)                 = delete;
    PairwisePetscOperator &operator=(PairwisePetscOperator &&)      = delete;

    /**
     * @brief Compute the stationary distribution via SLEPc Krylov–Schur.
     *
     * Solves for the leading eigenvector of P^T (eigenvalue = 1) using SLEPc's
     * EPS solver with Krylov–Schur restart. The solve runs entirely in C++
     * (no Python callbacks per matvec). The result is gathered on ALL ranks.
     *
     * @param tol       Convergence tolerance (default 1e-12).
     * @param max_iter  Maximum EPS iterations (default 300).
     * @return          Normalised stationary distribution of length size().
     * @throws std::runtime_error if SLEPc fails to converge.
     */
    [[nodiscard]] egttools::Vector compute_stationary_distribution(
        double tol      = 1e-12,
        int    max_iter = 300);

    [[nodiscard]] PetscInt size() const { return nb_states_; }
    [[nodiscard]] size_t   population_size() const { return pop_size_; }
    [[nodiscard]] size_t   nb_strategies() const { return nb_strategies_; }
    [[nodiscard]] double   beta() const { return beta_; }
    [[nodiscard]] double   mu() const { return mu_; }

    /**
     * @brief PETSc MatShell callback: y = P^T x using MPI_Allgatherv + MPI_Allreduce.
     *
     * Called by PETSc/SLEPc once per Krylov step — entirely in C++.
     */
    static PetscErrorCode matvec(Mat A, Vec x, Vec y);

private:
    size_t   pop_size_;
    size_t   nb_strategies_;
    PetscInt nb_states_;
    PetscInt lo_, hi_;   // this rank's owned row range [lo, hi)
    egttools::FinitePopulations::AbstractGame *game_;
    double beta_, mu_;
    double inv_Z_, inv_Zm1_, one_minus_mu_, mutation_probability_;

    Mat shell_;   // PETSc MatShell (P^T)

    /**
     * @brief Accumulate P^T contributions from a single source state into y_partial.
     *
     * For source state index `src` (decoded to `current`), computes all off-diagonal
     * transition probabilities and adds prob * x_full[src] to y_partial[dest] for each
     * destination dest. Also returns the total off-diagonal probability so the caller
     * can handle the diagonal term.
     *
     * @param src       Source state global index.
     * @param current   Strategy-count vector for src (decoded externally).
     * @param x_full    Full input vector (all n elements).
     * @param y_partial Accumulation buffer (all n elements).
     * @return          Total off-diagonal probability from src.
     */
    double enumerate_local_(PetscInt                    src,
                            egttools::VectorXui        &current,
                            const double               *x_full,
                            double                     *y_partial) const;
};

} // namespace egttools::mpi

#endif // EGTTOOLS_MPI_PAIRWISEPETSCOPERATOR_HPP

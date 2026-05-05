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
#ifndef EGTTOOLS_FINITEPOPULATIONS_PAIRWISECOMPARISONTRANSITIONOPERATOR_HPP
#define EGTTOOLS_FINITEPOPULATIONS_PAIRWISECOMPARISONTRANSITIONOPERATOR_HPP

#include <egttools/Distributions.h>
#include <egttools/Types.h>
#include <egttools/finite_populations/Utils.hpp>
#include <egttools/finite_populations/games/AbstractGame.hpp>

#include <stdexcept>
#include <vector>

namespace egttools::FinitePopulations {

    /**
     * @brief Matrix-free transition operator for the pairwise comparison process.
     *
     * Computes y = P x, y = P^T x, and y = (I - P^T) x without assembling the
     * transition matrix P. Intended for iterative eigensolvers (scipy ARPACK,
     * petsc4py/slepc4py) and as the foundation for future MPI-distributed computation.
     *
     * The transition probability from state s to s + e_j - e_i (strategy j gains one
     * individual, strategy i loses one) is:
     *
     *   P(s → s+e_j-e_i) = (s[i]/Z) * ((1-μ)*s[j]/(Z-1)*fermi(β, f_i, f_j) + μ/(k-1))
     *
     * where fermi(β, die, birth) = 1/(1+exp(β*(die-birth))), f_i is the fitness of
     * strategy i in state s (computed by removing one i from the population), Z is the
     * population size, and k is the number of strategies.
     *
     * For monomorphic states (s[i] = Z), only mutation is possible:
     *   P(s → s+e_j-e_i) = μ/(k-1)  for all j ≠ i.
     *
     * Self-loop: P(s → s) = 1 - Σ off-diagonal terms.
     *
     * @note The operator is deterministic: all randomness is in the game's fitness
     *       evaluation; once beta, mu, population_size, and game are fixed the operator
     *       is fully defined.
     *
     * @note MPI readiness: apply_transpose uses source enumeration where each source
     *       state writes to y[dest]. For MPI, rewrite as incoming-neighbour enumeration
     *       (each rank owns a contiguous slice of dest indices and ghost-reads x).
     */
    class PairwiseComparisonTransitionOperator {
    public:
        /**
         * @brief Construct the matrix-free transition operator.
         *
         * @param population_size  Number of individuals Z (must be >= 2).
         * @param game             Game object defining fitness; kept alive by the caller.
         * @param beta             Intensity of selection (Fermi parameter, must be >= 0).
         * @param mu               Mutation probability per step (must be in [0, 1]).
         * @throws std::invalid_argument if any parameter is out of range or game has < 2 strategies.
         */
        PairwiseComparisonTransitionOperator(size_t population_size,
                                             AbstractGame &game,
                                             double beta,
                                             double mu);

        /**
         * @brief Total number of states in the simplex: C(Z+k-1, k-1).
         */
        [[nodiscard]] int64_t size() const;

        /**
         * @brief Compute y = P x in-place.
         *
         * Uses source enumeration: for each source state s, reads x[dest] for each
         * neighbouring dest and accumulates into y[s].
         *
         * @param x  Input vector of length size().
         * @param y  Output vector of length size(); zeroed and overwritten.
         * @throws std::invalid_argument if x or y have wrong length.
         */
        void apply(const Eigen::Ref<const Vector> &x,
                   Eigen::Ref<Vector> y) const;

        /**
         * @brief Compute y = P^T x in-place.
         *
         * Uses source enumeration: for each source state s, accumulates
         * p * x[s] into y[dest] for each outgoing transition (s → dest, prob p).
         * The stationary distribution π satisfies P^T π = π.
         *
         * @param x  Input vector of length size().
         * @param y  Output vector of length size(); zeroed and overwritten.
         * @throws std::invalid_argument if x or y have wrong length.
         */
        void apply_transpose(const Eigen::Ref<const Vector> &x,
                             Eigen::Ref<Vector> y) const;

        /**
         * @brief Compute y = (I - P^T) x in-place.
         *
         * Residual operator useful for iterative linear solvers seeking π with
         * (I - P^T) π = 0.
         *
         * @param x  Input vector of length size().
         * @param y  Output vector of length size(); overwritten.
         * @throws std::invalid_argument if x or y have wrong length.
         */
        void apply_residual(const Eigen::Ref<const Vector> &x,
                            Eigen::Ref<Vector> y) const;

        /**
         * @brief Compute the stationary distribution via power iteration (pure C++).
         *
         * Iterates π ← P^T π / ‖P^T π‖₁ until L1 convergence or max_iter is reached.
         * Runs entirely in C++ with no Python callbacks.
         *
         * @param tol       L1 convergence threshold (default 1e-10).
         * @param max_iter  Maximum number of iterations (default 10000).
         * @return          Normalised stationary distribution vector of length size().
         * @throws std::runtime_error if convergence is not reached within max_iter.
         */
        [[nodiscard]] Vector compute_stationary_distribution(
            double tol      = 1e-10,
            size_t max_iter = 10000) const;

#if HAS_ARPACK
        /**
         * @brief Compute the stationary distribution via ARPACK IRAM (pure C++).
         *
         * Uses ARPACK's implicitly restarted Arnoldi method to find the leading
         * eigenvector of P^T without Python callbacks.  Converges much faster than
         * power iteration when the spectral gap is small (small μ or large Z).
         *
         * Only available when EGTtools is compiled with EGTTOOLS_ENABLE_ARPACK=ON.
         *
         * @param tol       ARPACK convergence tolerance (default 0 → machine precision).
         * @param ncv       Krylov subspace size; 0 → auto (max(2*nev+1, 20)).
         * @param max_iter  Maximum Arnoldi iterations (default 300).
         * @return          Normalised stationary distribution vector of length size().
         * @throws std::runtime_error on ARPACK error or non-convergence.
         */
        [[nodiscard]] Vector compute_stationary_arpack(
            double tol      = 0.0,
            int    ncv      = 0,
            int    max_iter = 300) const;
#endif

        // --- Accessors ---
        [[nodiscard]] size_t population_size() const { return pop_size_; }
        [[nodiscard]] size_t nb_strategies() const { return nb_strategies_; }
        [[nodiscard]] double beta() const { return beta_; }
        [[nodiscard]] double mu() const { return mu_; }

    private:
        size_t pop_size_;
        size_t nb_strategies_;
        int64_t nb_states_;
        AbstractGame *game_;
        double beta_;
        double mu_;

        // Precomputed constants
        double inv_Z_;     // 1.0 / pop_size_
        double inv_Zm1_;   // 1.0 / (pop_size_ - 1)
        double one_minus_mu_;
        double mutation_probability_;  // mu / (k-1) for k > 2, else mu

        /**
         * @brief Enumerate all off-diagonal transitions from source_state and call
         *        visit(dest_idx, prob) for each one.
         *
         * @param source_state  Strategy-count vector for the source state (length k).
         * @param visit         Callable with signature void(int64_t dest_idx, double prob).
         * @return              Total off-diagonal probability (sum of all prob values emitted).
         *
         * This is the shared inner loop reused by apply() and apply_transpose().
         * Mirrors the per-row logic of
         * analytical::PairwiseComparison::assemble_transition_matrix_from_fitness
         * but accumulates into a user-supplied callback instead of an Eigen::Triplet list.
         */
        template<typename Visitor>
        double enumerate_transitions_(const VectorXui &source_state, Visitor &&visit) const;
    };

    // -------------------------------------------------------------------------
    // Template implementation of enumerate_transitions_
    // -------------------------------------------------------------------------

    template<typename Visitor>
    double PairwiseComparisonTransitionOperator::enumerate_transitions_(
        const VectorXui &src, Visitor &&visit) const {

        const int k = static_cast<int>(nb_strategies_);
        const size_t Z = pop_size_;

        // Identify which strategies are present and compute their fitnesses.
        std::vector<int> present;
        present.reserve(k);
        std::vector<double> fitness(k, 0.0);

        VectorXui temp(src);  // mutable copy used for fitness computation

        for (int i = 0; i < k; ++i) {
            if (src(i) > 0) {
                present.push_back(i);
                // Fitness of strategy i is computed without the focal player.
                temp(i) -= 1;
                fitness[i] = game_->calculate_fitness(i, Z, temp);
                temp(i) += 1;
            }
        }

        double total_offdiag = 0.0;

        if (present.size() == 1) {
            // Monomorphic state: only mutation is possible.
            const int mono = present[0];
            // Temporarily modify temp to compute destination index.
            temp(mono) -= 1;
            for (int j = 0; j < k; ++j) {
                if (j == mono) continue;
                temp(j) += 1;
                const int64_t dest_idx =
                    static_cast<int64_t>(calculate_state(Z, temp));
                visit(dest_idx, mutation_probability_);
                total_offdiag += mutation_probability_;
                temp(j) -= 1;
            }
            temp(mono) += 1;
        } else {
            // Mixed state: enumerate all (i increases, j decreases) pairs.
            // i is the strategy gaining an individual in the destination.
            // j is the strategy losing an individual in the destination.
            for (int i = 0; i < k; ++i) {
                // Prepare dest by incrementing strategy i.
                temp(i) += 1;

                if (src(i) == 0) {
                    // Strategy i is absent in source: only mutation from present j → i.
                    for (const int j: present) {
                        if (j == i) continue;
                        temp(j) -= 1;
                        const int64_t dest_idx =
                            static_cast<int64_t>(calculate_state(Z, temp));
                        const double prob =
                            static_cast<double>(src(j)) * inv_Z_ * mutation_probability_;
                        if (prob > 0.0) {
                            visit(dest_idx, prob);
                            total_offdiag += prob;
                        }
                        temp(j) += 1;
                    }
                } else {
                    // Strategy i is present: selection (imitation) + mutation.
                    // sel_prefactor = (1-μ) * src[i] / (Z-1)
                    // Player j is the focal (dying) player, player i is imitated.
                    // fermi(β, f_j, f_i) = P(j copies i)
                    const double sel_prefactor =
                        one_minus_mu_ * static_cast<double>(src(i)) * inv_Zm1_;

                    for (const int j: present) {
                        if (j == i) continue;
                        temp(j) -= 1;
                        const int64_t dest_idx =
                            static_cast<int64_t>(calculate_state(Z, temp));

                        const double sel_prob =
                            sel_prefactor * fermi(beta_, fitness[j], fitness[i]);
                        const double prob =
                            static_cast<double>(src(j)) * inv_Z_ *
                            (sel_prob + mutation_probability_);

                        if (prob > 0.0) {
                            visit(dest_idx, prob);
                            total_offdiag += prob;
                        }
                        temp(j) += 1;
                    }
                }

                // Undo the increment of strategy i.
                temp(i) -= 1;
            }
        }

        return total_offdiag;
    }

} // namespace egttools::FinitePopulations

#endif // EGTTOOLS_FINITEPOPULATIONS_PAIRWISECOMPARISONTRANSITIONOPERATOR_HPP

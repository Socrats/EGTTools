//
// Created by Elias Fernandez on 20/11/2022.
//
#pragma once
#ifndef EGTTOOLS_INFINITEPOPULATIONS_REPLICATORDYNAMICS_HPP
#define EGTTOOLS_INFINITEPOPULATIONS_REPLICATORDYNAMICS_HPP

#include <egttools/Distributions.h>
#include <egttools/Types.h>

#include <egttools/finite_populations/Utils.hpp>
#include <egttools/infinite_populations/AbstractReplicatorGame.hpp>
#include <tuple>

#if defined(_OPENMP)
#include <egttools/OpenMPExtensions.hpp>
#endif

namespace egttools::infinite_populations {
    /**
     * Compute the replicator dynamics gradient for a two-player matrix game.
     *
     * Given a population state `frequencies`, this function returns the time derivative
     * of each strategy frequency under the standard replicator equation
     *
     * \f[
     * \dot{x}_i = x_i \left((A x)_i - x^\top A x\right),
     * \f]
     *
     * where `A` is the payoff matrix and `x` is the vector of strategy frequencies.
     *
     * @param frequencies
     *     Vector of strategy frequencies. Expected shape: `(nb_strategies,)`.
     * @param payoff_matrix
     *     Square payoff matrix for a two-player matrix game. Entry `(i, j)` gives the payoff
     *     obtained by strategy `i` when interacting with strategy `j`.
     *
     * @return
     *     Vector containing the replicator gradient.
     */
    Vector replicator_equation(const Vector &frequencies, const Matrix2D &payoff_matrix);

    /**
     * Compute the replicator dynamics gradient for a two-player game object.
     *
     * This overload is intended for games implementing the `AbstractReplicatorGame`
     * interface with group size equal to 2.
     *
     * @param frequencies
     *     Vector of strategy frequencies. Expected shape: `(nb_strategies,)`.
     * @param game
     *     Two-player game object used to compute strategy fitness.
     *
     * @return
     *     Vector containing the replicator gradient.
     *
     * @throws std::invalid_argument
     *     If `frequencies.size() != game.nb_strategies()` or `game.group_size() != 2`.
     */
    Vector replicator_equation(const Vector &frequencies, const AbstractReplicatorGame &game);

    /**
     * Compute the replicator dynamics gradient for an n-player game represented by a payoff table.
     *
     * The expected fitness of each strategy is computed by averaging over all group
     * configurations of size `group_size`, using the population frequencies.
     *
     * @param frequencies
     *     Vector of strategy frequencies. Expected shape: `(nb_strategies,)`.
     * @param payoff_matrix
     *     Matrix whose rows correspond to focal strategies and whose columns correspond to
     *     group configurations.
     * @param group_size
     *     Number of individuals in each interacting group.
     *
     * @return
     *     Vector containing the replicator gradient.
     */
    Vector replicator_equation_n_player(
        const Vector &frequencies,
        const Matrix2D &payoff_matrix,
        size_t group_size
    );

    /**
     * Compute the replicator dynamics gradient for an n-player game object.
     *
     * The expected fitness of each strategy is obtained from `game.calculate_fitness(...)`,
     * and the returned gradient is
     *
     * \f[
     * \dot{x}_i = x_i \left(f_i(x) - \bar f(x)\right),
     * \qquad
     * \bar f(x) = \sum_j x_j f_j(x).
     * \f]
     *
     * @param frequencies
     *     Vector of strategy frequencies. Expected shape: `(nb_strategies,)`.
     * @param game
     *     Game object used to compute strategy fitness.
     *
     * @return
     *     Vector containing the replicator gradient.
     *
     * @throws std::invalid_argument
     *     If `frequencies.size() != game.nb_strategies()`.
     */
    Vector replicator_equation_n_player(
        const Vector &frequencies,
        const AbstractReplicatorGame &game
    );

    /**
     * Evaluate the 3-strategy two-player replicator dynamics on a grid of population states.
     *
     * This function applies `replicator_equation` pointwise to the grids `x1`, `x2`,
     * and `x3`. Only points satisfying `x1 + x2 + x3 \approx 1` are evaluated.
     *
     * @param x1
     *     Grid containing the frequency of strategy 1.
     * @param x2
     *     Grid containing the frequency of strategy 2.
     * @param x3
     *     Grid containing the frequency of strategy 3.
     * @param game
     *     Two-player game object used to compute strategy fitness.
     *
     * @return
     *     Tuple `(dx1, dx2, dx3)` containing the three components of the vector field.
     */
    std::tuple<Matrix2D, Matrix2D, Matrix2D> vectorized_replicator_equation(
        const Matrix2D &x1,
        const Matrix2D &x2,
        const Matrix2D &x3,
        const AbstractReplicatorGame &game
    );

    /**
     * Evaluate the 3-strategy n-player replicator dynamics on a grid of population states.
     *
     * This function applies `replicator_equation_n_player` pointwise to the grids `x1`,
     * `x2`, and `x3`. Only points satisfying `x1 + x2 + x3 \approx 1` are evaluated.
     *
     * @param x1
     *     Grid containing the frequency of strategy 1.
     * @param x2
     *     Grid containing the frequency of strategy 2.
     * @param x3
     *     Grid containing the frequency of strategy 3.
     * @param game
     *     N-player game object used to compute strategy fitness.
     *
     * @return
     *     Tuple `(dx1, dx2, dx3)` containing the three components of the vector field.
     */
    std::tuple<Matrix2D, Matrix2D, Matrix2D> vectorized_replicator_equation_n_player(
        const Matrix2D &x1,
        const Matrix2D &x2,
        const Matrix2D &x3,
        const AbstractReplicatorGame &game
    );

    /**
     * Evaluate the 3-strategy n-player replicator dynamics on a grid of population states,
     * using an explicit payoff table.
     */
    std::tuple<Matrix2D, Matrix2D, Matrix2D> vectorized_replicator_equation_n_player(
        const Matrix2D &x1,
        const Matrix2D &x2,
        const Matrix2D &x3,
        const Matrix2D &payoff_matrix,
        size_t group_size
    );
} // namespace egttools::infinite_populations

#endif//EGTTOOLS_INFINITEPOPULATIONS_REPLICATORDYNAMICS_HPP

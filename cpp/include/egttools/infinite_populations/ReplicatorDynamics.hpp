//
// Created by Elias Fernandez on 20/11/2022.
//
#pragma once
#ifndef EGTTOOLS_REPLICATORDYNAMICS_HPP
#define EGTTOOLS_REPLICATORDYNAMICS_HPP

#include <egttools/Distributions.h>
#include <egttools/Types.h>

#include <egttools/finite_populations/Utils.hpp>
#include <tuple>

#if defined(_OPENMP)
#include <egttools/OpenMPUtils.hpp>
#endif

namespace egttools::infinite_populations {
    /**
     * Returns the gradient of the replicator dynamics given the current population state.
     *
     * The population state is defined by the frequencies of each strategy in the population.
     *
     * @param frequencies : Vector of frequencies of each strategy in the population (it must have
     *                      shape=(nb_strategies,)
     * @param payoff_matrix : Square matrix containing the payoff of each row strategy against
     *                        each column strategy
     * @return a Vector containing the change in frequency of each strategy in the population
     */
    Vector replicator_equation(Vector &frequencies, Matrix2D &payoff_matrix);

    /**
     * Returns the gradient of the replicator dynamics given the current population state.
     *
     * The population state is defined by the frequencies of each strategy in the population.
     *
     * @param frequencies : Vector of frequencies of each strategy in the population (it must have
*                           shape=(nb_strategies,)
     * @param payoff_matrix : A payoff matrix containing the payoff of each row strategy for each
     *                        possible group configuration, indicated by the column index.
     *                        The matrix must have shape (nb_strategies, nb_group_configurations).
     * @param group_size : size of the group
     * @return a Vector containing the change in frequency of each strategy in the population
     */
    Vector replicator_equation_n_player(Vector &frequencies, Matrix2D &payoff_matrix, size_t group_size);


    std::tuple<Matrix2D, Matrix2D, Matrix2D> vectorized_replicator_equation_n_player(Matrix2D &x1, Matrix2D &x2, Matrix2D &x3,
                                                                                     Matrix2D &payoff_matrix, size_t group_size);
}// namespace egttools::infinite_populations

#endif//EGTTOOLS_REPLICATORDYNAMICS_HPP

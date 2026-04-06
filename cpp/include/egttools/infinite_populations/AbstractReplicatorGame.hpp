//
// Created by Elias Fernandez on 04/04/2026.
//
#pragma once
#ifndef EGTTOOLS_INFINITEPOPULATIONS_ABSTRACTREPLICATORGAME_HPP
#define EGTTOOLS_INFINITEPOPULATIONS_ABSTRACTREPLICATORGAME_HPP

#include <egttools/Types.h>
#include <egttools/Utils.h>

#include <egttools/finite_populations/Utils.hpp>
#include <vector>

namespace egttools::infinite_populations {
    using PayoffVector = std::vector<double>;
    using GroupPayoffs = egttools::Matrix2D;

    /**
     * Interface for games that define fitness in the infinite-population limit and
     * can therefore be used with replicator dynamics.
     *
     * Concrete implementations must provide:
     * - the number of strategies in the game,
     * - a way to compute the expected fitness of any focal strategy at a given
     *   population state,
     * - access to the corresponding payoff table when available.
     *
     * The population state is represented by a vector of strategy frequencies.
     * Implementations may assume that this vector has one entry per strategy,
     * that all entries are non-negative, and that they sum to one.
     */
    class AbstractReplicatorGame {
    public:
        virtual ~AbstractReplicatorGame() = default;

        /**
         * Compute or refresh the payoff table of the game.
         *
         * Implementations may use this method to lazily compute and cache the
         * payoff structure associated with the game.
         *
         * @return
         *     Reference to the payoff table of the game.
         */
        virtual const GroupPayoffs &calculate_payoffs() = 0;

        /**
         * Return the expected fitness of all strategies given a population state.
         *
         * The argument `frequencies` represents the population state in the
         * infinite-population limit, with one entry per strategy.
         *
         * @param frequencies
         *     Population state represented as a vector of strategy frequencies.
         *
         * @return
         *     Expected fitness of the focal strategy at the given population state.
         */
        [[nodiscard]] virtual Vector calculate_fitness(
            const Eigen::Ref<const Vector> &frequencies
        ) const = 0;

        /**
         * Return the number of strategies in the game.
         *
         * @return
         *     Number of strategies.
         */
        [[nodiscard]] virtual size_t nb_strategies() const = 0;

        /**
         * Return the group size of the game
         *
         * @return
         *   Group size.
         */
        [[nodiscard]] virtual int group_size() const = 0;

        /**
         * Return a short human-readable description of the game.
         *
         * @return
         *     Description of the game.
         */
        [[nodiscard]] virtual std::string toString() const = 0;

        /**
         * Return the type identifier of the game.
         *
         * @return
         *     Type of game.
         */
        [[nodiscard]] virtual std::string type() const = 0;

        /**
         * Return the currently stored payoff table of the game.
         *
         * If the payoff table is computed lazily, `calculate_payoffs()` should be
         * called first to ensure that the returned reference is initialized and
         * up to date.
         *
         * @return
         *     Reference to the payoff table of the game.
         */
        [[nodiscard]] virtual const GroupPayoffs &payoffs() const = 0;
    };
}

#endif //EGTTOOLS_INFINITEPOPULATIONS_ABSTRACTREPLICATORGAME_HPP

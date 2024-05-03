//
// Created by Elias Fernandez on 11/11/2022.
//
#pragma once
#ifndef EGTTOOLS_FINITEPOPULATIONS_GAMES_MATRIX2PLAYERNETWORKGAMEHOLDER_HPP
#define EGTTOOLS_FINITEPOPULATIONS_GAMES_MATRIX2PLAYERNETWORKGAMEHOLDER_HPP

#include <egttools/Distributions.h>
#include <egttools/Types.h>
#include <egttools/Utils.h>

#include <egttools/finite_populations/Utils.hpp>
#include <egttools/finite_populations/games/AbstractSpatialGame.hpp>
#include <fstream>
#include <stdexcept>
#include <string>


namespace egttools::FinitePopulations::games {
    using PayoffVector = std::vector<double>;

    class Matrix2PlayerNetworkGameHolder final : public egttools::FinitePopulations::games::AbstractSpatialGame {
    public:
        /**
         * @brief Holder class for 2-player games for which the expected
         *        payoff between strategies has already been calculated.
         *
         * This class is useful to store the matrix of expected payoffs between strategies
         * in an 2-player game and keep the methods to calculate the fitness between these strategies.
         *
         * @param nb_strategies : number of strategies in the game
         * @param payoff_matrix : matrix of shape (nb_strategies, nb_strategies) containing the payoffs
         *                        of each strategy against any other strategy.
         */
        Matrix2PlayerNetworkGameHolder(int nb_strategies, const Eigen::Ref<const Matrix2D> &expected_payoffs);

        /**
         * @brief Estimates the fitness for a @param player_type in the population with state @param strategies.
         *
         * This function assumes that the player with strategy @param player_type is not included in
         * the vector of strategy counts @param strategies.
         *
         * @param player_type : index of the strategy used by the player
         * @param pop_size : size of the population
         * @param strategies : current state of the population
         * @return a fitness value
         */
        double calculate_fitness(int strategy_index, VectorXui &state) final;

        // getters
        [[nodiscard]] int nb_strategies() const final;
        [[nodiscard]] std::string toString() const final;
        [[nodiscard]] std::string type() const final;
        [[nodiscard]] const Matrix2D &expected_payoffs() const;

    protected:
        int nb_strategies_;

        Matrix2D expected_payoffs_;
    };
}// namespace egttools::FinitePopulations::games

#endif//EGTTOOLS_FINITEPOPULATIONS_GAMES_MATRIX2PLAYERNETWORKGAMEHOLDER_HPP

//
// Created by Elias Fernandez on 11/11/2022.
//
#pragma once
#ifndef EGTTOOLS_FINITEPOPULATIONS_GAMES_MATRIX2PLAYERGAMEHOLDER_HPP
#define EGTTOOLS_FINITEPOPULATIONS_GAMES_MATRIX2PLAYERGAMEHOLDER_HPP

#include <egttools/Distributions.h>
#include <egttools/Types.h>
#include <egttools/Utils.h>

#include <egttools/finite_populations/Utils.hpp>
#include <egttools/finite_populations/games/AbstractGame.hpp>
#include <fstream>
#include <stdexcept>
#include <string>


namespace egttools::FinitePopulations {
    using PayoffVector = std::vector<double>;

    class Matrix2PlayerGameHolder : public AbstractGame {
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
        Matrix2PlayerGameHolder(int nb_strategies, Matrix2D payoff_matrix);

        void play(const egttools::FinitePopulations::StrategyCounts &group_composition,
                  PayoffVector &game_payoffs) final;

        const GroupPayoffs &calculate_payoffs() final;

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
        double calculate_fitness(const int &player_type, const size_t &pop_size, const Eigen::Ref<const VectorXui> &strategies) final;

        [[nodiscard]] size_t nb_strategies() const final;
        [[nodiscard]] std::string toString() const final;
        [[nodiscard]] std::string type() const final;

        [[nodiscard]] const GroupPayoffs &payoffs() const final;
        [[nodiscard]] double payoff(int strategy, const egttools::FinitePopulations::StrategyCounts &group_composition) const final;

        void update_payoff_matrix(Matrix2D &payoff_matrix);

        void save_payoffs(std::string file_name) const final;

    protected:
        int nb_strategies_;

        Matrix2D expected_payoffs_;
    };
}// namespace egttools::FinitePopulations

#endif//EGTTOOLS_FINITEPOPULATIONS_GAMES_MATRIX2PLAYERGAMEHOLDER_HPP

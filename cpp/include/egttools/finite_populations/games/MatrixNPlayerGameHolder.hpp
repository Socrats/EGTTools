//
// Created by Elias Fernandez on 11/11/2022.
//
#pragma once
#ifndef EGTTOOLS_FINITEPOPULATIONS_GAMES_MATRIXNPLAYERGAMEHOLDER_HPP
#define EGTTOOLS_FINITEPOPULATIONS_GAMES_MATRIXNPLAYERGAMEHOLDER_HPP

#include <egttools/Distributions.h>
#include <egttools/Types.h>
#include <egttools/Utils.h>

#include <cassert>
#include <egttools/finite_populations/Utils.hpp>
#include <egttools/finite_populations/games/AbstractGame.hpp>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>


namespace egttools::FinitePopulations {
    using PayoffVector = std::vector<double>;

    class MatrixNPlayerGameHolder : public AbstractGame {
    public:
        /**
         * @brief Holder class for N-player games for which the expected
         *        payoff between strategies has already been calculated.
         *
         * This class is useful to store the matrix of expected payoffs between strategies
         * in an N-player game and keep the methods to calculate the fitness between these strategies.
         *
         * @param nb_strategies : number of strategies in the game
         * @param group_size : size of the group
         * @param payoff_matrix : matrix of shape (nb_strategies, nb_group_configurations) containing the payoffs
         *                        of each strategy for every possible group configuration.
         */
        MatrixNPlayerGameHolder(int nb_strategies, int group_size, Matrix2D payoff_matrix);

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
        double
        calculate_fitness(const int &player_type, const size_t &pop_size, const Eigen::Ref<const VectorXui> &strategies) final;

        [[nodiscard]] size_t nb_strategies() const final;
        [[nodiscard]] int group_size() const;
        [[nodiscard]] int64_t nb_group_configurations() const;

        [[nodiscard]] std::string toString() const final;
        [[nodiscard]] std::string type() const final;

        [[nodiscard]] const GroupPayoffs &payoffs() const final;
        [[nodiscard]] double payoff(int strategy, const egttools::FinitePopulations::StrategyCounts &group_composition) const final;

        void update_payoff_matrix(Matrix2D &payoff_matrix);

        void save_payoffs(std::string file_name) const final;

    protected:
        int nb_strategies_, group_size_;
        int64_t nb_group_configurations_;

        Matrix2D expected_payoffs_;
    };
}// namespace egttools::FinitePopulations

#endif//EGTTOOLS_FINITEPOPULATIONS_GAMES_MATRIXNPLAYERGAMEHOLDER_HPP

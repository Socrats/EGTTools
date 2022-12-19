//
// Created by Elias Fernandez on 2019-06-13.
//
#pragma once
#ifndef EGTTOOLS_FINITEPOPULATIONS_GAMES_CRDGAMETU_HPP
#define EGTTOOLS_FINITEPOPULATIONS_GAMES_CRDGAMETU_HPP

#include <egttools/Distributions.h>
#include <egttools/Types.h>
#include <egttools/finite_populations/behaviors/CRDStrategies.h>

#include <cassert>
#include <egttools/finite_populations/Utils.hpp>
#include <egttools/finite_populations/behaviors/AbstractCRDStrategy.hpp>
#include <egttools/finite_populations/games/AbstractGame.hpp>
#include <egttools/utils/TimingUncertainty.hpp>
#include <fstream>
#include <stdexcept>
#include <vector>

#if defined(_OPENMP)
#include <egttools/OpenMPUtils.hpp>
#endif

namespace egttools::FinitePopulations::games {
    using PayoffVector = std::vector<double>;
    using AbstractCRDStrategy = egttools::FinitePopulations::behaviors::AbstractCRDStrategy;
    using CRDStrategyVector = std::vector<AbstractCRDStrategy *>;

    class CRDGameTU final : public egttools::FinitePopulations::AbstractGame {
    public:
        /**
         * @brief This class will update the payoffs according to the Collective-risk dilemma
         * with timing uncertianty. (see Fernández Domingos et al. 2020)
         *
         * The number of rounds of the game is uncertain and follows a geometric distribution.
         *
         * @param endowment : initial amount each player receives
         * @param threshold : minimum target to avoid losing all the endowment
         * @param min_rounds : minimum number of rounds
         * @param group_size : number of players in the group
         * @param risk : probability that all players will lose their endowment if the target isn't reached
         * @param tu : class that calculates the total number of rounds through a geometric distribution
         * @param strategies : vector containing pointers to the strategies that will play the game
         */
        CRDGameTU(int endowment, int threshold, int min_rounds, int group_size, double risk,
                  egttools::utils::TimingUncertainty<std::mt19937_64> &tu,
                  const CRDStrategyVector &strategies);

        void play(const egttools::FinitePopulations::StrategyCounts &group_composition,
                  PayoffVector &game_payoffs) override;

        /**
         * @brief updates private payoff matrix and returns it
         *
         * @return payoff matrix of the game
         */
        const GroupPayoffs &calculate_payoffs() override;

        double
        calculate_fitness(const int &player_type, const size_t &pop_size,
                          const Eigen::Ref<const VectorXui> &strategies) override;

        /**
         * @brief Calculates the group achievement for all possible groups
         *
         * If the strategies are deterministic, the output matrix will consist
         * of ones and zeros indicating whether the group reached or not the target.
         * If they are stochastic, it will indicate the probability of success of
         * the group.
         *
         * @return a matrix with the group achievement for each possible group
         */
        const Vector &calculate_success_per_group_composition();

        /**
         * @brief Calculates the probability of success given a population state
         * @param pop_size : size of the population
         * @param population_state : state of the population (number of players of each strategy)
         * @return the group achievement of that population state
         */
        double
        calculate_population_group_achievement(size_t pop_size, const Eigen::Ref<const VectorXui> &population_state);

        /**
         * @brief estimates the group achievement from a stationary distribution
         * @param pop_size : size of the population
         * @param stationary_distribution
         * @return group achievement (probability of group success)
         */
        double calculate_group_achievement(size_t pop_size, const Eigen::Ref<const Vector> &stationary_distribution);

        /**
         * @brief Calculates the fraction of players that invest >, < or = to E/2.
         *
         * Calculates the fraction of players that invest above, below or equal to the fair donation
         * given a population state.
         *
         * @param pop_size : size of the population
         * @param population_state : state of the population
         * @param polarization : container for polarization data
         * @return an array of 3 elements [C < E/2, C = E/2, C > E/2]
         */
        void calculate_population_polarization(size_t pop_size, const Eigen::Ref<const VectorXui> &population_state,
                                               Vector3d &polarization);

        /**
        * @brief Calculates the fraction of players that invest >, < or = to E/2.
        *
        * Calculates the fraction of players that invest above, below or equal to the fair donation
        * given a population state.
        *
        * @param pop_size : size of the population
        * @param population_state : state of the population
        * @param polarization : container for polarization data
        * @return an array of 3 elements [C < E/2, C = E/2, C > E/2]
        */
        void
        calculate_population_polarization_success(size_t pop_size, const Eigen::Ref<const VectorXui> &population_state,
                                                  Vector3d &polarization);

        /**
         * @brief calculates the fraction of players that invest (<, =, >) than E/2 given a stationary distribution.
         * @param pop_size : size of the population
         * @param stationary_distribution
         * @return the polarization vector
         */
        Vector3d calculate_polarization(size_t pop_size, const Eigen::Ref<const Vector> &stationary_distribution);

        /**
         * @brief calculates the fraction of players that invest (<, =, >) than E/2 given a stationary distribution.
         * @param pop_size : size of the population
         * @param stationary_distribution
         * @return the polarization vector
         */
        Vector3d
        calculate_polarization_success(size_t pop_size, const Eigen::Ref<const Vector> &stationary_distribution);

        // getters
        [[nodiscard]] size_t endowment() const;

        [[nodiscard]] size_t target() const;

        [[nodiscard]] size_t min_rounds() const;

        [[nodiscard]] size_t group_size() const;

        [[nodiscard]] size_t nb_strategies() const override;

        [[nodiscard]] size_t nb_states() const;

        [[nodiscard]] double risk() const;

        [[nodiscard]] std::string toString() const override;

        [[nodiscard]] std::string type() const override;

        [[nodiscard]] const GroupPayoffs &payoffs() const override;

        [[nodiscard]] double payoff(int strategy, const egttools::FinitePopulations::StrategyCounts &group_composition) const override;

        void save_payoffs(std::string file_name) const override;

        [[nodiscard]] const Vector &group_achievements() const;

        [[nodiscard]] const Matrix2D &contribution_behaviors() const;

        [[nodiscard]] const CRDStrategyVector &strategies() const;

    protected:
        int endowment_, threshold_, min_rounds_, group_size_, nb_strategies_;
        int64_t nb_states_;
        double risk_;
        GroupPayoffs expected_payoffs_;
        Vector group_achievement_;
        MatrixXui2D c_behaviors_counts_;
        Matrix2D c_behaviors_;
        egttools::utils::TimingUncertainty<std::mt19937_64> tu_;

        CRDStrategyVector strategies_;

        // Random generators
        std::mt19937_64 generator_{egttools::Random::SeedGenerator::getInstance().getSeed()};

        /**
         * @brief Check if game is successful and update state in group_achievement_
         *
         * It updates group_achievement_. It also updates c_behaviors_ with the fraction
         * of players that contributed (<, =, >) than endowment/2.
         *
         * @param state : current state index
         * @param game_payoffs : container for the payoffs
         * @param group_composition : composition of the group
         */
        void _check_success(size_t state, PayoffVector &game_payoffs,
                            const egttools::FinitePopulations::StrategyCounts &group_composition);
    };
}// namespace egttools::FinitePopulations::games

#endif//EGTTOOLS_FINITEPOPULATIONS_GAMES_CRDGAMETU_HPP

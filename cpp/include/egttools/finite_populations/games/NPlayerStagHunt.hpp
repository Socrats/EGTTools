//
// Created by Elias Fernandez on 03/12/2021.
//
#pragma once
#ifndef EGTTOOLS_FINITEPOPULATIONS_GAMES_NPLAYERSTAGHUNT_HPP
#define EGTTOOLS_FINITEPOPULATIONS_GAMES_NPLAYERSTAGHUNT_HPP

#include <egttools/Distributions.h>
#include <egttools/Types.h>

#include <cassert>
#include <egttools/finite_populations/Utils.hpp>
#include <egttools/finite_populations/games/AbstractGame.hpp>
#include <fstream>
#include <stdexcept>
#include <vector>

#if defined(_OPENMP)
#include <egttools/OpenMPUtils.hpp>
#endif

namespace egttools::FinitePopulations {
    using PayoffVector = std::vector<double>;

    class NPlayerStagHunt final : public egttools::FinitePopulations::AbstractGame {
    public:
        /**
         * This class implements a One-Shot Collective Risk Dilemma.
         *
         *  This N-player game was first introduced in "Santos, F. C., & Pacheco, J. M. (2011).
         *  Risk of collective failure provides an escape from the tragedy of the commons.
         *  Proceedings of the National Academy of Sciences of the United States of America, 108(26), 10421–10425.".*
         *
         *  The game consists of a group of size ``group_size`` (N) which can be composed of
         *  Cooperators (Cs) who will contribute a fraction ``cost`` (c) of their
         *  ``endowment`` (b) to the public good. And of Defectors (Ds) who contribute 0.
         *
         *  If the total contribution of the group is equal or surpasses the collective target Mcb,
         *  with M being the ``min_nb_cooperators``, then all participants will receive as payoff
         *  their remaining endowment. Which is, Cs receive b - cb and Ds receive b. Otherwise, all
         *  participants receive 0 endowment with a probability equal to ``risk`` (r), and will
         *  keep their endowment with probability 1-r. This means that each group must have at least
         *  M Cs for the collective target to be achieved.
         *
         * @param group_size : The size of the group (N)
         * @param cooperation_threshold : The minimum number of cooperators (M) required to reach the
         *                                collective target. In other words, the collective target is
         *                                reached if the collective effort is at least Mcb. This value
         *                                must be in the discrete interval [[0, N]].
         * @param enhancement_factor : public good surplus multiplier (the contributions to the public
         *                             good are multiplied by this valued before being divided equally
         *                             among all players).
         * @param cost : The fraction of the endowment that Cooperators contribute to the public good.
         *               This value must be in the interval [0, 1]
         */
        NPlayerStagHunt(int group_size, int cooperation_threshold, double enhancement_factor, double cost);

        /**
         * @brief Plays the One-shop CRD and update the game_payoffs given the group_composition.
         *
         * We always assume that strategy 0 is D and strategy 1 is C.
         *
         * The payoffs of Defectors and Cooperators are described by the following equations:
         *
         * .. math::
         *      \\Pi_{D}(k) = b{\\theta(k-M)+ (1-r)[1 - \\theta(k-M)]}
         *      \\Pi_{C}(k) = \\Pi_{D}(k) - cb
         *      \\text{where } \\theta(x) = 0 \\text{if } x < 0 \\text{ and 1 otherwise.}
         *
         * @param group_composition : vector containing the counts of each strategy in the population.
         * @param game_payoffs : vector which will serve as container for the payoffs of each strategy.
         */
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
        const VectorXi &calculate_success_per_group_composition();

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

        void save_payoffs(std::string file_name) const override;

        // getters
        [[nodiscard]] int group_size() const;
        [[nodiscard]] int cooperation_threshold() const;
        [[nodiscard]] double enhancement_factor() const;
        [[nodiscard]] double cost() const;
        [[nodiscard]] size_t nb_strategies() const override;
        [[nodiscard]] std::vector<std::string> strategies() const;
        [[nodiscard]] int64_t nb_group_configurations() const;
        [[nodiscard]] std::string toString() const override;
        [[nodiscard]] std::string type() const override;
        [[nodiscard]] const GroupPayoffs &payoffs() const override;
        [[nodiscard]] double payoff(int strategy, const egttools::FinitePopulations::StrategyCounts &group_composition) const override;
        [[nodiscard]] const VectorXi &group_achievements() const;


    protected:
        int group_size_, cooperation_threshold_, nb_strategies_;
        int64_t nb_group_configurations_;
        double enhancement_factor_, cost_;
        std::vector<std::string> strategies_;
        GroupPayoffs expected_payoffs_;
        VectorXi group_achievement_;
    };

}// namespace egttools::FinitePopulations


#endif//EGTTOOLS_FINITEPOPULATIONS_GAMES_NPLAYERSTAGHUNT_HPP

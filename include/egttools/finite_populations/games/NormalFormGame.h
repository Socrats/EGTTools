//
// Created by Elias Fernandez on 28/12/2020.
//

#ifndef EGTTOOLS_INCLUDE_DYRWIN_FINITEPOPULATIONS_GAMES_NORMALFORMGAME_H_
#define EGTTOOLS_INCLUDE_DYRWIN_FINITEPOPULATIONS_GAMES_NORMALFORMGAME_H_

#include <egttools/Distributions.h>

#include <cassert>
#include <egttools/OpenMPUtils.hpp>
#include <egttools/finite_populations/behaviors/AbstractNFGStrategy.hpp>
#include <egttools/finite_populations/games/AbstractGame.hpp>
#include <fstream>

namespace egttools::FinitePopulations {
    using PayoffVector = std::vector<double>;
    using AbstractNFGStrategy = egttools::FinitePopulations::behaviors::AbstractNFGStrategy;
    using StrategyVector = std::vector<std::shared_ptr<AbstractNFGStrategy>>;

    class NormalFormGame final : public egttools::FinitePopulations::AbstractGame {
    public:
        /**
        * @brief This class implements a normal form game.
        *
        * The payoffs of the game are defined by a payoff matrix.
        * It is always a 2-player game, but may contain more than 2 possible actions.
        *
        * If @param nb_rounds > 1, than the game is iterated (has more than 1 round).
        *
        * In case the number of rounds is > 1, this class will estimate
        * The expected payoff for each strategy and update it's own internal
        * payoff matrix.
        *
        * The drawback of this method is that is that if the number of strategies is high,
        * it might take a long time to precalculate the payoffs. In cases where the payoffs of each
        * strategy are analytically defined, it might be best input the payoff matrix with the expected
        * payoffs for all strategies and indicate that the game takes only 1 round.
        * In this latter case, the cooperation vector has no meaning
        *
        * @param nb_rounds : number of rounds of the game.
        * @param payoff_matrix : Eigen matrix containing the payoffs.
        */
        NormalFormGame(size_t nb_rounds, const Eigen::Ref<const Matrix2D> &payoff_matrix);

        NormalFormGame(size_t nb_rounds, const Eigen::Ref<const Matrix2D> &payoff_matrix, const StrategyVector& strategies);

//        NormalFormGame(size_t nb_rounds, const Eigen::Ref<const Matrix2D> &payoff_matrix, const std::vector<std::string> strategies);

        void play(const egttools::FinitePopulations::StrategyCounts &group_composition,
                  PayoffVector &game_payoffs) override;

        /**
         * @brief Gets an action from the strategy defined by player type.
         *
         * This method will call one of the behaviors specified in CrdBehaviors.hpp indexed by
         * @param player_type with the parameters @param prev_donation, threshold, current_round.
         *
         * @param player_type : type of strategy (as an unsigned integer).
         * @param       prev_action : previous donation of the group.
         * @param current_round : current round of the game
         * @return action of the strategy
         */
        //  static inline size_t get_action(const size_t &player_type, const size_t &prev_action, const size_t &current_round);

        /**
        * @brief updates private payoff matrix and returns it
        *
        * @return payoff matrix of the game
        */
        const GroupPayoffs &calculate_payoffs() override;

        double
        calculate_fitness(const size_t &player_type, const size_t &pop_size,
                          const Eigen::Ref<const VectorXui> &strategies) override;

        /**
        * @brief Calculates the expected level of cooperation given a population state
        * @param pop_size : size of the population
        *       @param population_state : state of the population (number of players of each strategy)
        * @return the level of cooperation of that population state
        */
        double calculate_cooperation_level(size_t pop_size, const Eigen::Ref<const VectorXui> &population_state);

        // getters
        [[nodiscard]] size_t nb_strategies() const override;
        [[nodiscard]] size_t nb_rounds() const;
        [[nodiscard]] size_t nb_states() const;
        [[nodiscard]] std::string toString() const override;
        [[nodiscard]] std::string type() const override;
        [[nodiscard]] const GroupPayoffs &payoffs() const override;
        [[nodiscard]] const Matrix2D &expected_payoffs() const;
        [[nodiscard]] double payoff(size_t strategy, const egttools::FinitePopulations::StrategyCounts &group_composition) const override;
        [[nodiscard]] const StrategyVector &strategies() const;
        void save_payoffs(std::string file_name) const override;

        // setters

    protected:
        size_t nb_rounds_, nb_strategies_, nb_states_;
        Matrix2D payoffs_, expected_payoffs_, coop_level_;
        StrategyVector strategies_;

        /**
        * @brief updates the expected_payoffs_ and coop_level_ matrices for the strategies indicates
        * @param s1 : strategy 1
        * @param s2 : strategy 2
        */
        void _update_cooperation_and_payoffs(size_t s1, size_t s2);
    };

}// namespace egttools::FinitePopulations

#endif//DYRWIN_INCLUDE_DYRWIN_FINITEPOPULATIONS_GAMES_NORMALFORMGAME_H_

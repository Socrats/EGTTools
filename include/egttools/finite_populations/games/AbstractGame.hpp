//
// Created by Elias Fernandez on 2019-06-13.
//

#ifndef EGTTOOLS_FINITEPOPULATIONS_GAMES_ABSTRACTGAME_HPP
#define EGTTOOLS_FINITEPOPULATIONS_GAMES_ABSTRACTGAME_HPP

#include <egttools/Utils.h>

#include <egttools/finite_populations/Utils.hpp>

namespace egttools::FinitePopulations {
    using PayoffVector = std::vector<double>;
    using RandomDist = std::uniform_real_distribution<double>;

    /**
     * @brief This class defines the interface of a game to be used in an evolutionary process.
     */
    class AbstractGame {
    public:
        virtual ~AbstractGame() = default;

        /**
         * @brief updates the vector of payoffs with the payoffs of each player after playing the game.
         *
         * This method will run the game using the players and player types defined in @param group_composition,
         * and will update the vector @param game_payoffs with the resulting payoff of each player.
         *
         * @param nb_strategies number of strategies in the population
         * @param group_composition number of players of each strategy in the group
         * @param game_payoffs container for the payoffs of each player
         * @param urand distribution for uniform random numbers
         * @param generator random generator
         */
        virtual void play(const egttools::FinitePopulations::StrategyCounts &group_composition,
                          PayoffVector &game_payoffs) = 0;

        /**
         * @brief Estimates the payoff matrix for each strategy.
         *
         * @param urand : uniform random distribution [0, 1).
         * @param generator : random generator
         * @return a payoff matrix
         */
        virtual const GroupPayoffs &calculate_payoffs() = 0;

        /**
         * @brief Estimates the fitness for a @param player_type in the population with state @param strategies.
         *
         * This function assumes that the player with strategy @param player_type is not included in
         * the vector of strategy counts @param strategies.
         *
         * @param player_type : index of the strategy used by the player
         * @param pop_size : size of the population
         * @param strategies : current state of the population
         * @param payoffs : the payoff matrix of the game
         * @return a fitness value
         */
        virtual double
        calculate_fitness(const size_t &player_type, const size_t &pop_size,
                          const Eigen::Ref<const VectorXui> &strategies) = 0;

        [[nodiscard]] virtual size_t nb_strategies() const = 0;

        /**
         * @return Returns a small description of the game.
         */
        [[nodiscard]] virtual std::string toString() const = 0;

        /**
         *
         * @return The type of game
         */
        [[nodiscard]] virtual std::string type() const = 0;

        /**
         *
         * @return payoff matrix of the game
         */
        [[nodiscard]] virtual const GroupPayoffs &payoffs() const = 0;

        /**
         * @brief returns the payoff of a strategy given a group composition
         *
         * If the group composition does not include the strategy, the payoff should be zero
         *
         * @param strategy : index of the strategy
         * @param group_composition : vector with the group composition
         * @return the payoff value
         */
        [[nodiscard]] virtual double payoff(size_t strategy, const egttools::FinitePopulations::StrategyCounts &group_composition) const = 0;

        /**
         * @brief stores the payoff matrix in a txt file
         *
         * @param file_name : name of the file in which the data will be stored
         */
        virtual void save_payoffs(std::string file_name) const = 0;
    };
}// namespace egttools::FinitePopulations

#endif//EGTTOOLS_ABSTRACTGAME_HPP

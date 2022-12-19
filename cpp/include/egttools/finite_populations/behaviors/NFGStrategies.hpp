/** Copyright (c) 2020-2021  Elias Fernandez
  *
  * This file is part of EGTtools.
  *
  * EGTtools is free software: you can redistribute it and/or modify
  * it under the terms of the GNU General Public License as published by
  * the Free Software Foundation, either version 3 of the License, or
  * (at your option) any later version.
  *
  * EGTtools is distributed in the hope that it will be useful,
  * but WITHOUT ANY WARRANTY; without even the implied warranty of
  * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  * GNU General Public License for more details.
  *
  * You should have received a copy of the GNU General Public License
  * along with EGTtools.  If not, see <http://www.gnu.org/licenses/>
*/

#ifndef EGTTOOLS_FINITEPOPULATIONS_BEHAVIORS_NFGSTRATEGIES_HPP
#define EGTTOOLS_FINITEPOPULATIONS_BEHAVIORS_NFGSTRATEGIES_HPP

#include <egttools/SeedGenerator.h>
#include <egttools/Types.h>
#include <egttools/Utils.h>

#include <algorithm>
#include <egttools/finite_populations/behaviors/AbstractNFGStrategy.hpp>
#include <memory>
#include <random>
#include <stdexcept>

// Many of the strategies defined here have been extracted from https://plato.stanford.edu/entries/prisoner-dilemma/strategy-table.html
// TODO: Reformat all namespaces: here change to behaviors::NormalForm::twoActions
namespace egttools::FinitePopulations::behaviors::twoActions {
    /**
     * Always cooperates
     */
    class Cooperator : public AbstractNFGStrategy {
    public:
        int get_action(size_t time_step, int action_prev) final;
        std::string type() final;
        bool is_stochastic() final;
    };
    /**
     * Always defects
     */
    class Defector : public AbstractNFGStrategy {
    public:
        int get_action(size_t time_step, int action_prev) final;
        std::string type() final;
        bool is_stochastic() final;
    };
    /**
     * Chooses an action with uniform random probability
     */
    class RandomPlayer : public AbstractNFGStrategy {
    public:
        RandomPlayer();

        int get_action(size_t time_step, int action_prev) final;
        std::string type() final;
        bool is_stochastic() final;

        std::uniform_int_distribution<int> rand_int_;
    };

    /**
     * Cooperates in the first round and imitates the opponent's move thereafter
     */
    class TitForTat : public AbstractNFGStrategy {
    public:
        int get_action(size_t time_step, int action_prev) final;
        std::string type() final;
        bool is_stochastic() final;
    };
    /**
     * Defects on the first round and imitates its opponent's previous move thereafter.
     */
    class SuspiciousTFT : public AbstractNFGStrategy {
    public:
        int get_action(size_t time_step, int action_prev) final;
        std::string type() final;
        bool is_stochastic() final;
    };
    /**
     * Cooprates on the first round and after its opponent
     * cooperates. Following a defection,it cooperates with probability
     * @f[ p(R,P,T,S) = min{1 - \frac{T-R}{R-S}, \frac{R-P}{T-P}} @f]
     * where R, P, T and S are the reward, punishment, temptation and
     * suckers payoffs.
     */
    class GenerousTFT : public AbstractNFGStrategy {
    public:
        GenerousTFT(double reward, double punishment, double temptation, double sucker);

        int get_action(size_t time_step, int action_prev) final;
        std::string type() final;
        bool is_stochastic() final;

        double p_;
        std::uniform_real_distribution<double> rand_double_;
    };
    /**
     * TFT with two differences:
     * (1) it increases the string of punishing defection responses
     * with each additional defection by its opponent
     * (2) it apologizes for each string of defections
     * by cooperating in the subsequent two rounds.
     */
    class GradualTFT : public AbstractNFGStrategy {
    public:
        int get_action(size_t time_step, int action_prev) final;
        std::string type() final;
        bool is_stochastic() final;

        int defection_string_ = 0;
        int cooperation_string_ = 0;
    };
    /**
     * Imitates opponent as in TFT, but makes mistakes
     * with @param error_probability
     */
    class ImperfectTFT : public AbstractNFGStrategy {
    public:
        explicit ImperfectTFT(double error_probability);

        int get_action(size_t time_step, int action_prev) final;
        std::string type() final;
        bool is_stochastic() final;

        double error_probability_;
        std::uniform_real_distribution<double> rand_double_;
    };
    /**
     * @brief Defects if defected twice
     */
    class TFTT : public AbstractNFGStrategy {
    public:
        int get_action(size_t time_step, int action_prev) final;
        std::string type() final;
        bool is_stochastic() final;

        int action_memory_ = 1;
    };
    /**
     * @brief Defects twice if defected
     */
    class TTFT : public AbstractNFGStrategy {
    public:
        int get_action(size_t time_step, int action_prev) final;
        std::string type() final;
        bool is_stochastic() final;

        int defection_counter_ = 0;// starts with cooperation
    };
    /**
     * @brief Also known as Trigger
     *
     * Cooperates until its opponent has defected once, and then defects for the rest of the game.
     */
    class GRIM : public AbstractNFGStrategy {
    public:
        int get_action(size_t time_step, int action_prev) final;
        std::string type() final;
        bool is_stochastic() final;

        int action_ = 1;
    };
    /**
     * @brief win-stay loose-shift
     * Cooperates if it and its opponent moved alike in
     * previous move and defects if they moved differently.
     */
    class Pavlov : public AbstractNFGStrategy {
    public:
        int get_action(size_t time_step, int action_prev) final;
        std::string type() final;
        bool is_stochastic() final;

        int action_memory_ = 1;
    };

    /**
     * Always repeats the same action, but explores a different action
     * with @param epsilon probability. The initial action is COOPERATE
     * with probability @param p.
     */
    class ActionInertia : public AbstractNFGStrategy {
    public:
        explicit ActionInertia(double epsilon, double p);

        int get_action(size_t time_step, int action_prev) final;
        std::string type() final;
        bool is_stochastic() final;

        double epsilon_, p_;
        int action_ = 1;

        std::uniform_real_distribution<double> rand_double_;
    };

    enum NFActions : int {
        DEFECT,
        COOPERATE
    };

    /**
     *
     * @param strategy_name
     * @return an object with the strategy
     * @throws invalid_argument Exception if the string does not correspond to a valid
     * strategy
     */
    template<typename... Args>
    std::shared_ptr<AbstractNFGStrategy> get_strategy(const std::string& strategy_name, Args... args) {
        std::shared_ptr<egttools::FinitePopulations::behaviors::AbstractNFGStrategy> strategy;

        if (strategy_name == "AllC") {
            strategy = std::make_shared<egttools::FinitePopulations::behaviors::twoActions::Cooperator>();
        } else if (strategy_name == "AllD") {
            strategy = std::make_shared<egttools::FinitePopulations::behaviors::twoActions::Defector>();
        } else if (strategy_name == "Random") {
            strategy = std::make_shared<egttools::FinitePopulations::behaviors::twoActions::RandomPlayer>();
        } else if (strategy_name == "TFT") {
            strategy = std::make_shared<egttools::FinitePopulations::behaviors::twoActions::TitForTat>();
        } else if (strategy_name == "SuspiciousTFT") {
            strategy = std::make_shared<egttools::FinitePopulations::behaviors::twoActions::SuspiciousTFT>();
        } else if (strategy_name == "GenerousTFT") {
            strategy = std::make_shared<egttools::FinitePopulations::behaviors::twoActions::GenerousTFT>(std::forward<Args>(args)...);
        } else if (strategy_name == "GradualTFT") {
            strategy = std::make_shared<egttools::FinitePopulations::behaviors::twoActions::GradualTFT>();
        } else if (strategy_name == "ImperfectTFT") {
            strategy = std::make_shared<egttools::FinitePopulations::behaviors::twoActions::ImperfectTFT>(std::forward<Args>(args)...);
        } else if (strategy_name == "TFTT") {
            strategy = std::make_shared<egttools::FinitePopulations::behaviors::twoActions::TFTT>();
        } else if (strategy_name == "TTFT") {
            strategy = std::make_shared<egttools::FinitePopulations::behaviors::twoActions::TTFT>();
        } else if (strategy_name == "GRIM") {
            strategy = std::make_shared<egttools::FinitePopulations::behaviors::twoActions::GRIM>();
        } else if (strategy_name == "Pavlov") {
            strategy = std::make_shared<egttools::FinitePopulations::behaviors::twoActions::Pavlov>();
        } else if (strategy_name == "ActionInertia") {
            strategy = std::make_shared<egttools::FinitePopulations::behaviors::twoActions::ActionInertia>(std::forward<Args>(args)...);
        } else {
            std::string exception_message;
            exception_message.append(strategy_name);
            exception_message.append(" is not a valid Strategy");
            throw std::invalid_argument(exception_message);
        }

        return strategy;
    };

    constexpr size_t nb_actions = 2;
}// namespace egttools::FinitePopulations::behaviors::twoActions

#endif//EGTTOOLS_FINITEPOPULATIONS_BEHAVIORS_NFGSTRATEGIES_HPP

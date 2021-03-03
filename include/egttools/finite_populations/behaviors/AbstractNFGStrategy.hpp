//
// Created by Elias Fernandez on 18/02/2021.
//

#ifndef EGTTOOLS_FINITEPOPULATIONS_BEHAVIORS_ABSTRACTSTRATEGY_HPP
#define EGTTOOLS_FINITEPOPULATIONS_BEHAVIORS_ABSTRACTSTRATEGY_HPP

namespace egttools::FinitePopulations::behaviors {

    /**
     * @brief defines the interface for strategies that can be used with AbstractGame (and child classes)
     */
    class AbstractNFGStrategy {
    public:
        virtual ~AbstractNFGStrategy() = default;
        /**
         * Function that will return the decision of the strategy
         * given the current @param time_step (or round) and the
         * previous action of the opponent @param action_prev.
         *
         * The strategies may take more information into account,
         * by maintaining state (which maybe the previous actions
         * of several round and their own previous actions)
         *
         * @param time_step : current round
         * @param action_prev : previous action of the opponent
         * @return the action decided by the strategy
         */
        virtual size_t get_action(size_t time_step, size_t action_prev) = 0;
        /**
         *
         * @return a string that indicates the strategy type
         * (e.g. StrategyType::Cooperator)
         */
        virtual std::string type() = 0;
    };
}// namespace egttools::FinitePopulations::behaviors

#endif//EGTTOOLS_FINITEPOPULATIONS_BEHAVIORS_ABSTRACTSTRATEGY_HPP

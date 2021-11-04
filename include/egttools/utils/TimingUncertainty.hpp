//
// Created by Elias Fernandez on 15/09/2021.
//
#pragma once
#ifndef EGTTOOLS_UTILS_TIMINGUNCERTAINTY_HPP
#define EGTTOOLS_UTILS_TIMINGUNCERTAINTY_HPP

#include <random>
#include <stdexcept>

namespace egttools::utils {
    template<typename R = std::mt19937_64, typename G = std::geometric_distribution<int>>
    class TimingUncertainty {
    public:
        /**
         * @brief Timing uncertainty distribution container.
         *
         * This class provides methods to calculate the final round of the game
         * according to some predifined distribution, which is geometric by default.
         *
         * @param p : Probability that the game will end after the minimum number of rounds.
         * @param max_round : maximum number of rounds that the game can take (if 0, there is no maximum).
         */
        explicit TimingUncertainty(double p, int max_rounds = 0);

        /**
         * Calculates the final round limiting by max_rounds
         * @param min_rounds
         * @param generator
         * @return outputs a value between [min_rounds, max_rounds]
         */
        int calculate_end(int min_rounds, R& generator);

        /**
         * Calculates the final round
         * @param min_rounds
         * @param generator
         * @return outputs a value between [min_rounds, Inf]
         */
        int calculate_full_end(int min_rounds, R& generator);

        // Getters
        [[nodiscard]] double probability() const;
        [[nodiscard]] int max_rounds() const;

        // Setters
        void set_probability(double p);

        void set_max_rounds(int max_rounds);

    private:
        double p_;
        int max_rounds_;
        G dist_;
    };

    template<typename R, typename G>
    TimingUncertainty<R, G>::TimingUncertainty(double p, int max_rounds) : p_(p),
                                                                           max_rounds_(max_rounds) {
        dist_ = G(p);
    }

    template<typename R, typename G>
    int
    TimingUncertainty<R, G>::calculate_end(int min_rounds, R& generator) {
        int rounds = min_rounds + dist_(generator);
        return (rounds > max_rounds_) ? max_rounds_ : rounds;
    }

    template<typename R, typename G>
    int
    TimingUncertainty<R, G>::calculate_full_end(int min_rounds, R& generator) {
        return min_rounds + dist_(generator);
    }

    template<typename R, typename G>
    double
    TimingUncertainty<R, G>::probability() const {
        return p_;
    }

    template<typename R, typename G>
    int
    TimingUncertainty<R, G>::max_rounds() const {
        return max_rounds_;
    }

    template<typename R, typename G>
    void
    TimingUncertainty<R, G>::set_probability(double p) {
        if (p <= 0.0 || p > 1.0) throw std::invalid_argument("Probability must be in (0,1)");
        p_ = p;
        dist_.param(G::param_type(p_));
    }

    template<typename R, typename G>
    void
    TimingUncertainty<R, G>::set_max_rounds(int max_rounds) {
        if (max_rounds <= 0) throw std::invalid_argument("Max rounds must be > 0");
        max_rounds_ = max_rounds;
    }


}// namespace egttools::utils

#endif//EGTTOOLS_UTILS_TIMINGUNCERTAINTY_HPP

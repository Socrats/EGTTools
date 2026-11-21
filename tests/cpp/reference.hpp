// Reference implementations for the C++ tests.
//
// Each function is a direct transcription of a published formula or of a definition in
// docs/theory/pairwise_comparison.md, written independently of the library code, so the
// tests compare the library against it. They favour clarity over speed and are only meant
// for the small populations used in the tests.
#ifndef EGTTOOLS_TESTS_REFERENCE_HPP
#define EGTTOOLS_TESTS_REFERENCE_HPP

#include <cmath>
#include <cstddef>
#include <functional>
#include <vector>

#include <egttools/Types.h>

namespace reference {

    // Binomial coefficient as a double, by the multiplicative formula (exact for the small
    // arguments used in the tests).
    inline double binomial(const long n, const long k) {
        if (k < 0 || k > n) return 0.0;
        double result = 1.0;
        for (long i = 1; i <= k; ++i) result = result * static_cast<double>(n - k + i) / static_cast<double>(i);
        return result;
    }

    // Definition 3.1: probability that a focal individual with fitness f_focal imitates a
    // role model with fitness f_model (Fermi rule).
    inline double fermi(const double beta, const double f_focal, const double f_model) {
        return 1.0 / (1.0 + std::exp(-beta * (f_model - f_focal)));
    }

    // Definition 2.1: fitness of strategy l in a two-player game, excluding the focal individual.
    inline double fitness_two_player(const egttools::Matrix2D &A, const std::vector<long> &n, const int l) {
        long Z = 0;
        for (const auto count: n) Z += count;
        double f = 0.0;
        for (int m = 0; m < static_cast<int>(n.size()); ++m)
            f += A(l, m) * static_cast<double>(n[m] - (l == m ? 1 : 0)) / static_cast<double>(Z - 1);
        return f;
    }

    // Definition 2.2 for two strategies: fitness of strategy l in an N-player game, where
    // payoff(l, k) is the payoff of a focal l-strategist whose N - 1 co-players include k
    // individuals of strategy 1. Co-players are drawn without replacement from the other Z - 1.
    inline double fitness_n_player_two_strategies(const std::function<double(int, long)> &payoff,
                                                  const long N, const std::vector<long> &n, const int l) {
        const long Z = n[0] + n[1];
        const long others_1 = n[1] - (l == 1 ? 1 : 0);
        const long others_0 = n[0] - (l == 0 ? 1 : 0);
        double f = 0.0;
        for (long k = 0; k <= N - 1; ++k)
            f += payoff(l, k) * binomial(others_1, k) * binomial(others_0, N - 1 - k) / binomial(Z - 1, N - 1);
        return f;
    }

    // Definition 4.1: probability that one j-strategist becomes an l-strategist (j != l),
    // given the fitness f(n) of each strategy in state n.
    inline double transition(const std::vector<long> &n, const std::vector<double> &f, const int j, const int l,
                             const double beta, const double mu) {
        const auto k = static_cast<double>(n.size());
        long Z = 0;
        for (const auto count: n) Z += count;
        return static_cast<double>(n[j]) / static_cast<double>(Z) *
               ((1 - mu) * static_cast<double>(n[l]) / static_cast<double>(Z - 1) * fermi(beta, f[j], f[l]) +
                mu / (k - 1));
    }

    // Fixation probability of a single invader in a resident population for a birth-death
    // chain with T+(i) / T-(i) = exp(beta (f_inv(i) - f_res(i))) (pairwise comparison with the
    // Fermi rule; Traulsen, Nowak & Pacheco 2006, Phys. Rev. E 74, 011909):
    //   rho = 1 / (1 + sum_{i=1}^{Z-1} prod_{j=1}^{i} T-(j) / T+(j)),  T-(j) / T+(j) = exp(-beta delta_f(j)),
    // where delta_f(j) = f_inv(j) - f_res(j) with j invaders.
    inline double fixation_probability(const std::function<double(long)> &delta_f, const long Z,
                                       const double beta) {
        double sum = 0.0, product = 1.0;
        for (long i = 1; i <= Z - 1; ++i) {
            product *= std::exp(-beta * delta_f(i));
            sum += product;
        }
        return 1.0 / (1.0 + sum);
    }

    // Stationary distribution of a birth-death chain on {0, ..., Z} by detailed balance:
    // pi(i + 1) / pi(i) = T+(i) / T-(i + 1).
    inline std::vector<double> birth_death_stationary(const std::function<double(long)> &t_plus,
                                                      const std::function<double(long)> &t_minus,
                                                      const long Z) {
        std::vector<double> pi(static_cast<size_t>(Z + 1), 0.0);
        pi[0] = 1.0;
        double total = 1.0;
        for (long i = 0; i < Z; ++i) {
            pi[static_cast<size_t>(i + 1)] = pi[static_cast<size_t>(i)] * t_plus(i) / t_minus(i + 1);
            total += pi[static_cast<size_t>(i + 1)];
        }
        for (auto &value: pi) value /= total;
        return pi;
    }

}// namespace reference

#endif// EGTTOOLS_TESTS_REFERENCE_HPP

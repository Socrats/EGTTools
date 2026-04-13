//
// Created by Elias Fernandez on 20/11/2022.
//
#include <egttools/infinite_populations/ReplicatorDynamics.hpp>

#include <stdexcept>

namespace {
    inline void validate_frequency_size(
        const egttools::Vector &frequencies,
        const egttools::infinite_populations::AbstractReplicatorGame &game,
        const char *function_name
    ) {
        if (static_cast<size_t>(frequencies.size()) != game.nb_strategies()) {
            throw std::invalid_argument(
                std::string(function_name) +
                ": frequencies size does not match game.nb_strategies()."
            );
        }
    }

    inline void validate_three_strategy_grid(
        const egttools::Matrix2D &x1,
        const egttools::Matrix2D &x2,
        const egttools::Matrix2D &x3,
        const char *function_name
    ) {
        if (x1.rows() != x2.rows() || x1.rows() != x3.rows() ||
            x1.cols() != x2.cols() || x1.cols() != x3.cols()) {
            throw std::invalid_argument(
                std::string(function_name) +
                ": x1, x2, and x3 must have the same shape."
            );
        }
    }

    inline egttools::Vector replicator_from_fitness(
        const egttools::Vector &frequencies,
        const egttools::Vector &fitness,
        const char *function_name
    ) {
        if (fitness.size() != frequencies.size()) {
            throw std::invalid_argument(
                std::string(function_name) +
                ": fitness vector size does not match frequencies size."
            );
        }

        const double fitness_avg = frequencies.dot(fitness);
        return frequencies.array() * (fitness.array() - fitness_avg);
    }
}

egttools::Vector egttools::infinite_populations::replicator_equation(
    const egttools::Vector &frequencies,
    const egttools::Matrix2D &payoff_matrix
) {
    auto fitness = payoff_matrix * frequencies;
    return replicator_from_fitness(frequencies, fitness, "replicator_equation");
}

egttools::Vector egttools::infinite_populations::replicator_equation(
    const egttools::Vector &frequencies,
    const egttools::infinite_populations::AbstractReplicatorGame &game
) {
    validate_frequency_size(frequencies, game, "replicator_equation");

    if (game.group_size() != 2) {
        throw std::invalid_argument(
            "replicator_equation: this overload requires a two-player game (group_size() == 2)."
        );
    }

    const auto fitness = game.calculate_fitness(frequencies);
    return replicator_from_fitness(frequencies, fitness, "replicator_equation");
}

egttools::Vector egttools::infinite_populations::replicator_equation_n_player(
    const egttools::Vector &frequencies,
    const egttools::Matrix2D &payoff_matrix,
    size_t group_size
) {
    egttools::Vector fitness = egttools::Vector::Zero(frequencies.size());
    egttools::VectorXui group_configuration = egttools::VectorXui::Zero(frequencies.size());
    const auto nb_group_configurations =
            egttools::starsBars<size_t, int64_t>(group_size, frequencies.size());

    for (int64_t i = 0; i < nb_group_configurations; ++i) {
        egttools::FinitePopulations::sample_simplex(
            i, group_size, frequencies.size(), group_configuration
        );

        for (Eigen::Index strategy_index = 0; strategy_index < frequencies.size(); ++strategy_index) {
            if (group_configuration(strategy_index) > 0) {
                group_configuration(strategy_index) -= 1;
                const auto prob = egttools::multinomialPMF(
                    group_configuration, group_size - 1, frequencies
                );
                fitness(strategy_index) += prob * payoff_matrix(strategy_index, i);
                group_configuration(strategy_index) += 1;
            }
        }
    }

    return replicator_from_fitness(frequencies, fitness, "replicator_equation_n_player");
}

egttools::Vector egttools::infinite_populations::replicator_equation_n_player(
    const egttools::Vector &frequencies,
    const egttools::infinite_populations::AbstractReplicatorGame &game
) {
    validate_frequency_size(frequencies, game, "replicator_equation_n_player");

    const auto fitness = game.calculate_fitness(frequencies);
    return replicator_from_fitness(frequencies, fitness, "replicator_equation_n_player");
}

std::tuple<egttools::Matrix2D, egttools::Matrix2D, egttools::Matrix2D>
egttools::infinite_populations::vectorized_replicator_equation(
    const egttools::Matrix2D &x1,
    const egttools::Matrix2D &x2,
    const egttools::Matrix2D &x3,
    const egttools::infinite_populations::AbstractReplicatorGame &game
) {
    validate_three_strategy_grid(x1, x2, x3, "vectorized_replicator_equation");

    if (game.nb_strategies() != 3) {
        throw std::invalid_argument(
            "vectorized_replicator_equation: this overload currently supports only 3 strategies."
        );
    }

    if (game.group_size() != 2) {
        throw std::invalid_argument(
            "vectorized_replicator_equation: this overload requires a two-player game (group_size() == 2)."
        );
    }

    egttools::Matrix2D result1 = egttools::Matrix2D::Zero(x1.rows(), x1.cols());
    egttools::Matrix2D result2 = egttools::Matrix2D::Zero(x2.rows(), x2.cols());
    egttools::Matrix2D result3 = egttools::Matrix2D::Zero(x3.rows(), x3.cols());

#if defined(_OPENMP) && !defined(_MSC_VER)
#pragma omp parallel for default(shared) shared(x1, x2, x3, game, result1, result2, result3)
#endif
    for (int i = 0; i < x1.rows(); ++i) {
        for (int j = 0; j < x1.cols(); ++j) {
            const double f1 = x1(i, j);
            const double f2 = x2(i, j);
            const double f3 = x3(i, j);
            const double sum = f1 + f2 + f3;

            if (sum > 1.0 + 1e-8 || sum < 1.0 - 1e-8) continue;
            if (f1 < -1e-12 || f2 < -1e-12 || f3 < -1e-12) continue;

            egttools::Vector frequencies(3);
            frequencies << f1, f2, f3;

            const auto result = replicator_equation(frequencies, game);

            result1(i, j) = result(0);
            result2(i, j) = result(1);
            result3(i, j) = result(2);
        }
    }

    return {result1, result2, result3};
}

std::tuple<egttools::Matrix2D, egttools::Matrix2D, egttools::Matrix2D>
egttools::infinite_populations::vectorized_replicator_equation_n_player(
    const egttools::Matrix2D &x1,
    const egttools::Matrix2D &x2,
    const egttools::Matrix2D &x3,
    const egttools::infinite_populations::AbstractReplicatorGame &game
) {
    validate_three_strategy_grid(x1, x2, x3, "vectorized_replicator_equation_n_player");

    if (game.nb_strategies() != 3) {
        throw std::invalid_argument(
            "vectorized_replicator_equation_n_player: this overload currently supports only 3 strategies."
        );
    }

    egttools::Matrix2D result1 = egttools::Matrix2D::Zero(x1.rows(), x1.cols());
    egttools::Matrix2D result2 = egttools::Matrix2D::Zero(x2.rows(), x2.cols());
    egttools::Matrix2D result3 = egttools::Matrix2D::Zero(x3.rows(), x3.cols());

#if defined(_OPENMP) && !defined(_MSC_VER)
#pragma omp parallel for default(shared) shared(x1, x2, x3, game, result1, result2, result3)
#endif
    for (int i = 0; i < x1.rows(); ++i) {
        for (int j = 0; j < x1.cols(); ++j) {
            const double f1 = x1(i, j);
            const double f2 = x2(i, j);
            const double f3 = x3(i, j);
            const double sum = f1 + f2 + f3;

            if (sum > 1.0 + 1e-8 || sum < 1.0 - 1e-8) continue;
            if (f1 < -1e-12 || f2 < -1e-12 || f3 < -1e-12) continue;

            egttools::Vector frequencies(3);
            frequencies << f1, f2, f3;

            const auto result = replicator_equation_n_player(frequencies, game);

            result1(i, j) = result(0);
            result2(i, j) = result(1);
            result3(i, j) = result(2);
        }
    }

    return {result1, result2, result3};
}

std::tuple<egttools::Matrix2D, egttools::Matrix2D, egttools::Matrix2D>
egttools::infinite_populations::vectorized_replicator_equation_n_player(
    const egttools::Matrix2D &x1,
    const egttools::Matrix2D &x2,
    const egttools::Matrix2D &x3,
    const egttools::Matrix2D &payoff_matrix,
    size_t group_size
) {
    validate_three_strategy_grid(x1, x2, x3, "vectorized_replicator_equation_n_player");

    egttools::Matrix2D result1 = egttools::Matrix2D::Zero(x1.rows(), x1.cols());
    egttools::Matrix2D result2 = egttools::Matrix2D::Zero(x2.rows(), x2.cols());
    egttools::Matrix2D result3 = egttools::Matrix2D::Zero(x3.rows(), x3.cols());

#if defined(_OPENMP) && !defined(_MSC_VER)
#pragma omp parallel for default(shared) shared(x1, x2, x3, payoff_matrix, group_size, result1, result2, result3)
#endif
    for (int i = 0; i < x1.rows(); ++i) {
        for (int j = 0; j < x1.cols(); ++j) {
            const double f1 = x1(i, j);
            const double f2 = x2(i, j);
            const double f3 = x3(i, j);
            const double sum = f1 + f2 + f3;

            if (sum > 1.0 + 1e-8 || sum < 1.0 - 1e-8) continue;
            if (f1 < -1e-12 || f2 < -1e-12 || f3 < -1e-12) continue;

            egttools::Vector frequencies(3);
            frequencies << f1, f2, f3;

            const auto result = replicator_equation_n_player(frequencies, payoff_matrix, group_size);

            result1(i, j) = result(0);
            result2(i, j) = result(1);
            result3(i, j) = result(2);
        }
    }

    return {result1, result2, result3};
}

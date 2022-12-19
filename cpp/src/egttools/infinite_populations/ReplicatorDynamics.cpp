//
// Created by Elias Fernandez on 20/11/2022.
//
#include <egttools/infinite_populations/ReplicatorDynamics.hpp>


egttools::Vector egttools::infinite_populations::replicator_equation(egttools::Vector &frequencies, egttools::Matrix2D &payoff_matrix) {
    auto ax = payoff_matrix * frequencies;
    return frequencies * (ax - (frequencies * ax));
}

egttools::Vector egttools::infinite_populations::replicator_equation_n_player(egttools::Vector &frequencies, egttools::Matrix2D &payoff_matrix, size_t group_size) {
    egttools::Vector fitness = egttools::Vector::Zero(frequencies.size());
    egttools::VectorXui group_configuration = egttools::VectorXui::Zero(frequencies.size());
    auto nb_group_configurations = egttools::starsBars<size_t, int64_t>(group_size, frequencies.size());

    // This loop calculates the fitness of each strategy
    for (int64_t i = 0; i < nb_group_configurations; ++i) {
        egttools::FinitePopulations::sample_simplex(i, group_size, frequencies.size(), group_configuration);

        for (int strategy_index = 0; strategy_index < frequencies.size(); ++strategy_index) {
            if (group_configuration(strategy_index) > 0) {
                group_configuration(strategy_index) -= 1;
                auto prob = egttools::multinomialPMF(group_configuration, group_size - 1, frequencies);
                fitness(strategy_index) += prob * payoff_matrix(strategy_index, i);
                group_configuration(strategy_index) += 1;
            }
        }
    }

    double fitness_avg = (frequencies.array() * fitness.array()).sum();

    for (int strategy_index = 0; strategy_index < frequencies.size(); ++strategy_index) {
        fitness(strategy_index) = frequencies(strategy_index) * (fitness(strategy_index) - fitness_avg);
    }

    return fitness;
}

std::tuple<egttools::Matrix2D, egttools::Matrix2D, egttools::Matrix2D>
egttools::infinite_populations::vectorized_replicator_equation_n_player(egttools::Matrix2D &x1,
                                                                        egttools::Matrix2D &x2,
                                                                        egttools::Matrix2D &x3,
                                                                        egttools::Matrix2D &payoff_matrix,
                                                                        size_t group_size) {
    egttools::Matrix2D result1 = egttools::Matrix2D::Zero(x1.rows(), x1.cols());
    egttools::Matrix2D result2 = egttools::Matrix2D::Zero(x2.rows(), x2.cols());
    egttools::Matrix2D result3 = egttools::Matrix2D::Zero(x3.rows(), x3.cols());

#pragma omp parallel for default(none) shared(x1, x2, x3, payoff_matrix, group_size, result1, result2, result3)
    for (int i = 0; i < x1.rows(); ++i) {
        for (int j = 0; j < x1.cols(); ++j) {
            // Check if we are in the simplex (frequencies must sum to 1)
            if (x1(i, j) + x2(i, j) + x3(i, j) > 1 + 1e-8 ||
                x1(i, j) + x2(i, j) + x3(i, j) < 1 - 1e-8)
                continue;

            egttools::Vector frequencies = egttools::Vector::Zero(3);
            frequencies(0) = x1(i, j);
            frequencies(1) = x2(i, j);
            frequencies(2) = x3(i, j);

            auto result = replicator_equation_n_player(frequencies, payoff_matrix, group_size);

            result1(i, j) = result(0);
            result2(i, j) = result(1);
            result3(i, j) = result(2);
        }
    }

    return {result1, result2, result3};
}
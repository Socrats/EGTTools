//
// Created by Elias Fernandez on 14/02/2024.
//
#include <egttools/finite_populations/games/Matrix2PlayerNetworkGameHolder.hpp>

egttools::FinitePopulations::games::Matrix2PlayerNetworkGameHolder::Matrix2PlayerNetworkGameHolder(int nb_strategies,
                                                                                                   const Eigen::Ref<const Matrix2D> &expected_payoffs) : nb_strategies_(nb_strategies),
                                                                                                                                                         expected_payoffs_(expected_payoffs) {
    if (expected_payoffs_.rows() != expected_payoffs_.cols()) {
        throw std::invalid_argument(
                "The number of rows must be equal to the number of columns of the payoff matrix. " + std::to_string(nb_strategies_) + " != " + std::to_string(expected_payoffs_.cols()));
    }
    if (nb_strategies_ != expected_payoffs_.rows()) {
        throw std::invalid_argument(
                "The number of rows must be equal to the number of strategies. " + std::to_string(expected_payoffs_.rows()) + " != " + std::to_string(nb_strategies_));
    }
}
double egttools::FinitePopulations::games::Matrix2PlayerNetworkGameHolder::calculate_fitness(int strategy_index, egttools::VectorXui &state) {
    double payoff = 0;

    for (int i = 0; i < nb_strategies_; ++i) {
        payoff += static_cast<int>(state(i)) * expected_payoffs_(strategy_index, i);
    }

    return payoff;
}
int egttools::FinitePopulations::games::Matrix2PlayerNetworkGameHolder::nb_strategies() const {
    return nb_strategies_;
}
std::string egttools::FinitePopulations::games::Matrix2PlayerNetworkGameHolder::toString() const {
    return "2-Player matrix network game holder";
}
std::string egttools::FinitePopulations::games::Matrix2PlayerNetworkGameHolder::type() const {
    return "egttools.games.Matrix2PlayerNetworkGameHolder";
}
const egttools::Matrix2D &egttools::FinitePopulations::games::Matrix2PlayerNetworkGameHolder::expected_payoffs() const {
    return expected_payoffs_;
}

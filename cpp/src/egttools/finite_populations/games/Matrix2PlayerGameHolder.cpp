//
// Created by Elias Fernandez on 11/11/2022.
//
#include <egttools/finite_populations/games/Matrix2PlayerGameHolder.hpp>

egttools::FinitePopulations::Matrix2PlayerGameHolder::Matrix2PlayerGameHolder(int nb_strategies, Matrix2D payoff_matrix) : nb_strategies_(nb_strategies),
                                                                                                                           expected_payoffs_(payoff_matrix) {

    if (payoff_matrix.rows() != payoff_matrix.cols()) {
        throw std::invalid_argument(
                "The number of rows must be equal to the number of columns of the payoff matrix. " + std::to_string(nb_strategies_) + " != " + std::to_string(payoff_matrix.cols()));
    }
    if (nb_strategies_ != payoff_matrix.rows()) {
        throw std::invalid_argument(
                "The number of rows must be equal to the number of strategies. " + std::to_string(payoff_matrix.rows()) + " != " + std::to_string(nb_strategies_));
    }
}

void egttools::FinitePopulations::Matrix2PlayerGameHolder::play(const egttools::FinitePopulations::StrategyCounts &group_composition, egttools::FinitePopulations::PayoffVector &game_payoffs) {
    UNUSED(group_composition);
    UNUSED(game_payoffs);
}

const egttools::FinitePopulations::GroupPayoffs &egttools::FinitePopulations::Matrix2PlayerGameHolder::calculate_payoffs() {
    return expected_payoffs_;
}

double egttools::FinitePopulations::Matrix2PlayerGameHolder::calculate_fitness(const int &player_type,
                                                                               const size_t &pop_size,
                                                                               const Eigen::Ref<const VectorXui> &strategies) {

    // This function assumes that the strategy counts given in @param strategies does not include
    // the player with @param player_type strategy.
    double fitness = 0.0;
    auto divider = static_cast<double>(pop_size - 1);

    // Given that we are only considering 2 player games, the fitness is simply the
    // expected payoff of each strategy * the probability of finding that strategy in the
    // population
    for (int i = 0; i < nb_strategies_; ++i) {
        fitness += expected_payoffs_(player_type, i) * (static_cast<double>(strategies(i)) / divider);
    }

    return fitness;
}

size_t egttools::FinitePopulations::Matrix2PlayerGameHolder::nb_strategies() const {
    return nb_strategies_;
}

std::string egttools::FinitePopulations::Matrix2PlayerGameHolder::toString() const {
    return "Holder game object for matrix of expected payoffs for 2-player games.";
}

std::string egttools::FinitePopulations::Matrix2PlayerGameHolder::type() const {
    return "egttools.games.Matrix2PlayerGameHolder";
}


const egttools::FinitePopulations::GroupPayoffs &egttools::FinitePopulations::Matrix2PlayerGameHolder::payoffs() const {
    return expected_payoffs_;
}

double egttools::FinitePopulations::Matrix2PlayerGameHolder::payoff(int strategy, const egttools::FinitePopulations::StrategyCounts &group_composition) const {
    if (strategy > static_cast<int>(nb_strategies_))
        throw std::invalid_argument(
                "you must specify a valid index for the strategy [0, " + std::to_string(nb_strategies_) +
                ")");
    return expected_payoffs_(static_cast<int>(group_composition[0]), static_cast<int>(group_composition[1]));
}

void egttools::FinitePopulations::Matrix2PlayerGameHolder::update_payoff_matrix(egttools::Matrix2D &payoff_matrix) {
    expected_payoffs_ = payoff_matrix;
}

void egttools::FinitePopulations::Matrix2PlayerGameHolder::save_payoffs(std::string file_name) const {
    // Save payoffs
    std::ofstream file(file_name, std::ios::out | std::ios::trunc);
    if (file.is_open()) {
        file << "Payoffs for strategy against each other:" << std::endl;
        file << "Each entry represents the payoff of the row strategy against the column strategy" << std::endl;
        file << expected_payoffs_ << std::endl;
        file.close();
    }
}

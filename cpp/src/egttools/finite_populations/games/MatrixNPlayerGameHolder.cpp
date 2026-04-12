//
// Created by Elias Fernandez on 11/11/2022.
//
#include <egttools/finite_populations/games/MatrixNPlayerGameHolder.hpp>
#include <egttools/utils/CalculateExpectedIndicators.h>

egttools::FinitePopulations::MatrixNPlayerGameHolder::MatrixNPlayerGameHolder(const int nb_strategies,
                                                                              const int group_size,
                                                                              const Eigen::Ref<const Matrix2D> &payoff_matrix) : nb_strategies_(nb_strategies),
                                                                                                                                           group_size_(group_size),
                                                                                                                                           expected_payoffs_(payoff_matrix) {

    if (group_size <= 2) {
        throw std::invalid_argument(
                "The size of the group must be > 2");
    }

    // number of possible group combinations
    nb_group_configurations_ = egttools::starsBars<int64_t>(group_size_, nb_strategies_);

    if (nb_strategies_ != payoff_matrix.rows()) {
        throw std::invalid_argument(
                "The number of rows must be equal to the number of strategies. " + std::to_string(payoff_matrix.rows()) + " != " + std::to_string(nb_strategies_));
    }

    if (nb_group_configurations_ != payoff_matrix.cols()) {
        throw std::invalid_argument(
                "The number of columns must be equal to the number of possible group configurations! " + std::to_string(nb_group_configurations_) + " != " + std::to_string(payoff_matrix.cols()));
    }
}

void egttools::FinitePopulations::MatrixNPlayerGameHolder::play(const egttools::FinitePopulations::StrategyCounts &group_composition, egttools::FinitePopulations::PayoffVector &game_payoffs) {
    UNUSED(group_composition);
    UNUSED(game_payoffs);
}

const egttools::FinitePopulations::GroupPayoffs &egttools::FinitePopulations::MatrixNPlayerGameHolder::calculate_payoffs() {
    return expected_payoffs_;
}

double egttools::FinitePopulations::MatrixNPlayerGameHolder::calculate_fitness(const int &player_type,
                                                                               const size_t &pop_size,
                                                                               const Eigen::Ref<const VectorXui> &strategies) {
    // This function assumes that the strategy counts given in @param strategies does not include
    // the player with @param player_type strategy.

    const egttools::Vector payoffs_row = expected_payoffs_.row(player_type);
    return egttools::utils::calculate_hypergeometric_fitness(
        player_type, pop_size,
        static_cast<size_t>(group_size_),
        static_cast<size_t>(nb_strategies_),
        strategies, payoffs_row);
}

size_t egttools::FinitePopulations::MatrixNPlayerGameHolder::nb_strategies() const {
    return nb_strategies_;
}

int egttools::FinitePopulations::MatrixNPlayerGameHolder::group_size() const {
    return group_size_;
}

int64_t egttools::FinitePopulations::MatrixNPlayerGameHolder::nb_group_configurations() const {
    return nb_group_configurations_;
}

std::string egttools::FinitePopulations::MatrixNPlayerGameHolder::toString() const {
    return "Holder game object for matrix of expected payoffs for N-player games.";
}

std::string egttools::FinitePopulations::MatrixNPlayerGameHolder::type() const {
    return "egttools.games.MatrixNPlayerGameHolder";
}


const egttools::FinitePopulations::GroupPayoffs &egttools::FinitePopulations::MatrixNPlayerGameHolder::payoffs() const {
    return expected_payoffs_;
}

double
egttools::FinitePopulations::MatrixNPlayerGameHolder::payoff(int strategy, const StrategyCounts &group_composition) const {
    if (strategy > nb_strategies_)
        throw std::invalid_argument(
                "you must specify a valid index for the strategy [0, " + std::to_string(nb_strategies_) +
                ")");
    if (group_composition.size() != static_cast<size_t>(nb_strategies_))
        throw std::invalid_argument("The group composition must be of size " + std::to_string(nb_strategies_));
    return expected_payoffs_(static_cast<int>(strategy), static_cast<int64_t>(calculate_state(group_size_, group_composition)));
}

void egttools::FinitePopulations::MatrixNPlayerGameHolder::update_payoff_matrix(const Matrix2D &payoff_matrix) {
    expected_payoffs_ = payoff_matrix;
}

void egttools::FinitePopulations::MatrixNPlayerGameHolder::save_payoffs(std::string file_name) const {
    // Save payoffs
    std::ofstream file(file_name, std::ios::out | std::ios::trunc);
    if (file.is_open()) {
        file << "Payoffs for each type of player and each possible state:" << std::endl;
        file << "rows: cooperator, defector, altruist, reciprocal, compensator" << std::endl;
        file << "cols: all possible group compositions starting at (0, 0, 0, 0, group_size)" << std::endl;
        file << expected_payoffs_ << std::endl;
        file << "group_size = " << group_size_ << std::endl;
        file << "nb_strategies = " << nb_strategies_ << std::endl;
        file.close();
    }
}

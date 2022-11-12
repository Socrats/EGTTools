//
// Created by Elias Fernandez on 2019-06-27.
//

#include <egttools/finite_populations/games/AbstractNPlayerGame.hpp>

egttools::FinitePopulations::AbstractNPlayerGame::AbstractNPlayerGame(int nb_strategies, int group_size)
    : nb_strategies_(nb_strategies),
      group_size_(group_size)
{

    // number of possible group combinations
    nb_group_configurations_ = egttools::starsBars<int64_t>(group_size_, nb_strategies_);

    expected_payoffs_ = GroupPayoffs::Zero(nb_strategies_, nb_group_configurations_);

    // Initialise payoff matrix
//    egttools::FinitePopulations::AbstractNPlayerGame::calculate_payoffs();
}

double egttools::FinitePopulations::AbstractNPlayerGame::calculate_fitness(const int &player_type, const size_t &pop_size,
                                                                           const Eigen::Ref<const VectorXui> &strategies) {
    // This function assumes that the strategy counts given in @param strategies does not include
    // the player with @param player_type strategy.

    double fitness = 0.0, payoff;
    std::vector<size_t> sample_counts(nb_strategies_, 0);

    // If it isn't, then we must calculate the fitness for every possible group combination
    for (int64_t i = 0; i < nb_group_configurations_; ++i) {
        // Update sample counts based on the current state - get new group composition
        egttools::FinitePopulations::sample_simplex(i, group_size_, nb_strategies_, sample_counts);

        // If the focal player is not in the group, then the payoff should be zero
        if (sample_counts[player_type] > 0) {
            payoff = expected_payoffs_(player_type, i);
            sample_counts[player_type] -= 1;

            // Calculate probability of encountering the current group
            auto prob = egttools::multivariateHypergeometricPDF(pop_size - 1, nb_strategies_, group_size_ - 1,
                                                                sample_counts,
                                                                strategies);
            sample_counts[player_type] += 1;

            fitness += payoff * prob;
        }
    }

    return fitness;
}

std::string egttools::FinitePopulations::AbstractNPlayerGame::toString() const {
    return "AbstractNPlayerGame";
}

std::string egttools::FinitePopulations::AbstractNPlayerGame::type() const {
    return "egttools.games.AbstractNPlayerGame";
}

size_t egttools::FinitePopulations::AbstractNPlayerGame::nb_strategies() const {
    return nb_strategies_;
}

const egttools::FinitePopulations::GroupPayoffs &egttools::FinitePopulations::AbstractNPlayerGame::payoffs() const {
    return expected_payoffs_;
}

double
egttools::FinitePopulations::AbstractNPlayerGame::payoff(int strategy, const egttools::FinitePopulations::StrategyCounts &group_composition) const {
    if (strategy > nb_strategies_)
        throw std::invalid_argument(
                "you must specify a valid index for the strategy [0, " + std::to_string(nb_strategies_) +
                ")");
    if (group_composition.size() != static_cast<size_t>(nb_strategies_))
        throw std::invalid_argument("The group composition must be of size " + std::to_string(nb_strategies_));
    return expected_payoffs_(static_cast<int>(strategy), static_cast<int64_t>(egttools::FinitePopulations::calculate_state(group_size_, group_composition)));
}

void egttools::FinitePopulations::AbstractNPlayerGame::update_payoff(int strategy_index, int group_configuration_index, double value) {
    expected_payoffs_(strategy_index, group_configuration_index) = value;
}


size_t egttools::FinitePopulations::AbstractNPlayerGame::group_size() const {
    return group_size_;
}
int64_t egttools::FinitePopulations::AbstractNPlayerGame::nb_group_configurations() const {
    return nb_group_configurations_;
}

void egttools::FinitePopulations::AbstractNPlayerGame::save_payoffs(std::string file_name) const {
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

//
// Created by Elias Fernandez on 03/12/2021.
//

#include <egttools/finite_populations/games/OneShotCRD.hpp>

egttools::FinitePopulations::OneShotCRD::OneShotCRD(double endowment, double cost,
                                                    double risk, int group_size, int min_nb_cooperators)
    : group_size_(group_size),
      min_nb_cooperators_(min_nb_cooperators),
      endowment_(endowment),
      cost_(cost),
      risk_(risk) {
    // For the moment we only consider Cs and Ds
    nb_strategies_ = 2;

    // Payoffs cooperator and defector
    payoff_defector_success_ = endowment_;
    payoff_defector_failure_ = endowment_ * (1 - risk);
    payoff_coop_success_ = endowment_ * (1 - cost_);
    payoff_coop_failure_ = payoff_defector_failure_ - (endowment_ * cost_);

    // Number of possible group combinations
    nb_group_compositions_ = egttools::starsBars<int64_t>(group_size_, nb_strategies_);

    // Initialize payoff containers
    expected_payoffs_ = GroupPayoffs::Zero(nb_strategies_, nb_group_compositions_);
    group_achievement_ = egttools::VectorXi::Zero(nb_group_compositions_);

    // Initialize payoff matrix
    egttools::FinitePopulations::OneShotCRD::calculate_payoffs();

    // Initialize group achievement vector
    calculate_success_per_group_composition();
}
void egttools::FinitePopulations::OneShotCRD::play(const egttools::FinitePopulations::StrategyCounts &group_composition,
                                                   std::vector<double> &game_payoffs) {
    if (static_cast<int>(group_composition[1]) < min_nb_cooperators_) {
        game_payoffs[0] = payoff_defector_failure_;
        game_payoffs[1] = payoff_coop_failure_;
    } else {
        game_payoffs[0] = payoff_defector_success_;
        game_payoffs[1] = payoff_coop_success_;
    }
}

const egttools::FinitePopulations::GroupPayoffs &egttools::FinitePopulations::OneShotCRD::calculate_payoffs() {
    StrategyCounts group_composition(nb_strategies_, 0);
    std::vector<double> game_payoffs(nb_strategies_, 0);

    // Initialise matrix
    expected_payoffs_.setZero();

    // For every possible group composition run the game and store the payoff of each strategy
    for (int64_t i = 0; i < nb_group_compositions_; ++i) {
        // Update group composition from current state
        egttools::FinitePopulations::sample_simplex(i, group_size_, nb_strategies_, group_composition);

        // play game and update game_payoffs
        play(group_composition, game_payoffs);

        // Fill payoff table
        for (int j = 0; j < nb_strategies_; ++j) expected_payoffs_(j, i) = game_payoffs[j];
    }

    return expected_payoffs_;
}

double egttools::FinitePopulations::OneShotCRD::calculate_fitness(const int &player_type, const size_t &pop_size,
                                                                  const Eigen::Ref<const VectorXui> &strategies) {
    // This function assumes that the strategy counts given in @param strategies does not include
    // the player with @param player_type strategy.

    double fitness = 0.0, payoff;
    std::vector<size_t> sample_counts(nb_strategies_, 0);

    // If it isn't, then we must calculate the fitness for every possible group combination
    for (int64_t i = 0; i < nb_group_compositions_; ++i) {
        // Update sample counts based on the current state
        egttools::FinitePopulations::sample_simplex(i, group_size_, nb_strategies_, sample_counts);

        // If the focal player is not in the group, then the payoff should be zero
        if (sample_counts[player_type] > 0) {
            // First update sample_counts with new group composition
            payoff = expected_payoffs_(static_cast<int>(player_type), i);
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

void egttools::FinitePopulations::OneShotCRD::save_payoffs(std::string file_name) const {
    // Save payoffs
    std::ofstream file(file_name, std::ios::out | std::ios::trunc);
    if (file.is_open()) {
        file << "Payoffs for each type of player and each possible state:" << std::endl;
        file << "rows: cooperator, defector, altruist, reciprocal, compensator" << std::endl;
        file << "cols: all possible group compositions starting at (0, 0, 0, 0, group_size)" << std::endl;
        file << expected_payoffs_ << std::endl;
        file << "group_size = " << group_size_ << std::endl;
        file << "risk = " << risk_ << std::endl;
        file << "cost = " << cost_ << std::endl;
        file << "endowment = " << endowment_ << std::endl;
        file << "min_nb_cooperators = " << min_nb_cooperators_ << std::endl;
        file.close();
    }
}

const egttools::FinitePopulations::GroupPayoffs &egttools::FinitePopulations::OneShotCRD::payoffs() const {
    return expected_payoffs_;
}

double
egttools::FinitePopulations::OneShotCRD::payoff(int strategy, const egttools::FinitePopulations::StrategyCounts &group_composition) const {
    if (strategy > nb_strategies_)
        throw std::invalid_argument(
                "you must specify a valid index for the strategy [0, " + std::to_string(nb_strategies_) +
                ")");
    if (group_composition.size() != static_cast<size_t>(nb_strategies_))
        throw std::invalid_argument("The group composition must be of size " + std::to_string(nb_strategies_));
    return expected_payoffs_(static_cast<int>(strategy), static_cast<int64_t>(egttools::FinitePopulations::calculate_state(group_size_, group_composition)));
}

const egttools::VectorXi &egttools::FinitePopulations::OneShotCRD::calculate_success_per_group_composition() {
    StrategyCounts group_composition(nb_strategies_, 0);
    std::vector<double> game_payoffs(nb_strategies_, 0);

    group_achievement_.setZero();

    // For every possible group composition run the game and store whether the group is successful or not
    for (int64_t i = 0; i < nb_group_compositions_; ++i) {
        // Update group composition from current state
        egttools::FinitePopulations::sample_simplex(i, group_size_, nb_strategies_, group_composition);

        // play game and update group achievement
        if (static_cast<int>(group_composition[1]) >= min_nb_cooperators_) group_achievement_(i) = 1;
    }

    return group_achievement_;
}

double egttools::FinitePopulations::OneShotCRD::calculate_population_group_achievement(size_t pop_size,
                                                                                       const Eigen::Ref<const egttools::VectorXui> &population_state) {

    double group_achievement = 0.0;
    std::vector<size_t> sample_counts(nb_strategies_, 0);

    // If it isn't, then we must calculate the fitness for every possible group combination
    for (int64_t i = 0; i < nb_group_compositions_; ++i) {
        // Update sample counts based on the current state
        egttools::FinitePopulations::sample_simplex(i, group_size_, nb_strategies_, sample_counts);

        if (group_achievement_(i) == 1) {
            group_achievement += egttools::multivariateHypergeometricPDF(pop_size, nb_strategies_, group_size_, sample_counts,
                                                                         population_state);
        }
    }

    return group_achievement;
}

double egttools::FinitePopulations::OneShotCRD::calculate_group_achievement(size_t pop_size,
                                                                            const Eigen::Ref<const egttools::Vector> &stationary_distribution) {
    double group_achievement = 0;

#pragma omp parallel for default(none) shared(pop_size, stationary_distribution, nb_strategies_) reduction(+ \
                                                                                                           : group_achievement)
    for (int64_t i = 0; i < stationary_distribution.size(); ++i) {
        VectorXui strategies = VectorXui::Zero(nb_strategies_);
        egttools::FinitePopulations::sample_simplex(i, pop_size, nb_strategies_, strategies);
        group_achievement += stationary_distribution(i) * calculate_population_group_achievement(pop_size, strategies);
    }
    return group_achievement;
}

std::string egttools::FinitePopulations::OneShotCRD::toString() const {
    std::stringstream ss;
    ss << "Python implementation of a public goods one-shot Collective Risk Dilemma." << std::endl;
    ss << "Game parameters" << std::endl;
    ss << "-------" << std::endl;
    ss << "b = " << endowment_ << std::endl;
    ss << "c = " << cost_ << std::endl;
    ss << "r = " << risk_ << std::endl;
    ss << "N = " << group_size_ << std::endl;
    ss << "M = " << min_nb_cooperators_ << std::endl;
    ss << "Strategies" << std::endl;
    ss << "-------" << std::endl;
    ss << "Currently the only strategies are Cooperators and Defectors." << std::endl;

    return ss.str();
}

std::string egttools::FinitePopulations::OneShotCRD::type() const {
    return "egttools.games.OneShotCRD";
}

size_t egttools::FinitePopulations::OneShotCRD::nb_strategies() const {
    return nb_strategies_;
}

const egttools::VectorXi &egttools::FinitePopulations::OneShotCRD::group_achievements() const {
    return group_achievement_;
}

double egttools::FinitePopulations::OneShotCRD::endowment() const {
    return endowment_;
}

double egttools::FinitePopulations::OneShotCRD::cost() const {
    return cost_;
}

double egttools::FinitePopulations::OneShotCRD::risk() const {
    return risk_;
}

int egttools::FinitePopulations::OneShotCRD::group_size() const {
    return group_size_;
}

int egttools::FinitePopulations::OneShotCRD::min_nb_cooperators() const {
    return min_nb_cooperators_;
}

int64_t egttools::FinitePopulations::OneShotCRD::nb_group_compositions() const {
    return nb_group_compositions_;
}

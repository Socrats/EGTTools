//
// Created by Elias Fernandez on 2019-06-27.
//

#include <egttools/finite_populations/games/CRDGame.hpp>

egttools::FinitePopulations::CRDGame::CRDGame(int endowment, int threshold, int nb_rounds, int group_size,
                                              double risk, double enhancement_factor, const CRDStrategyVector &strategies)
    : endowment_(endowment),
      threshold_(threshold),
      nb_rounds_(nb_rounds),
      group_size_(group_size),
      risk_(risk),
      enhancement_factor_(enhancement_factor),
      strategies_(strategies) {

    // First we check how many strategies will be in the game
    nb_strategies_ = static_cast<int>(strategies.size());

    // number of possible group combinations
    nb_states_ = egttools::starsBars<int64_t>(group_size_, nb_strategies_);

    expected_payoffs_ = GroupPayoffs::Zero(nb_strategies_, nb_states_);
    group_achievement_ = egttools::Vector::Zero(nb_states_);
    c_behaviors_ = egttools::MatrixXui2D::Zero(nb_states_, 3);

    // Initialise payoff matrix
    egttools::FinitePopulations::CRDGame::calculate_payoffs();

    // Initialise group achievement vector
    calculate_success_per_group_composition();
}

// TODO: It might be necessary to change this whole class to use floats
void egttools::FinitePopulations::CRDGame::play(const egttools::FinitePopulations::StrategyCounts &group_composition,
                                                PayoffVector &game_payoffs) {
    int prev_donation = 0, current_donation = 0;
    int public_account = 0;
    VectorXi actions = VectorXi::Zero(nb_strategies_);

    // Initialize payoffs
    for (int j = 0; j < nb_strategies_; ++j) {
        if (group_composition[j] > 0) {
            game_payoffs[j] = endowment_;
        } else {
            game_payoffs[j] = 0;
        }
    }

    for (int i = 0; i < nb_rounds_; ++i) {
        for (int j = 0; j < nb_strategies_; ++j) {
            if (group_composition[j] > 0) {
                actions(j) = strategies_[j]->get_action(i, prev_donation - actions(j));
                if (game_payoffs[j] >= actions(j)) {
                    game_payoffs[j] -= actions(j);
                    current_donation += static_cast<int>(group_composition[j]) * actions(j);
                } else {// If not enough, contribute what is left of the endowment
                    current_donation += static_cast<int>(group_composition[j]) * static_cast<int>(game_payoffs[j]);
                    game_payoffs[j] = 0;
                }
            }
        }
        public_account += current_donation;
        prev_donation = current_donation;
        current_donation = 0;
        if (public_account >= threshold_) {
            for (int j = 0; j < nb_strategies_; ++j) {
                game_payoffs[j] = game_payoffs[j] * enhancement_factor_;
            }
            break;
        }
    }

    // Calculate expected payoffs from risk
    if (public_account < threshold_)
        for (auto &type : game_payoffs) type *= (1.0 - risk_);
}

std::string egttools::FinitePopulations::CRDGame::toString() const {
    return "Collective-risk dilemma game.\n"
           "See Milinski et al. 2008.";
}

std::string egttools::FinitePopulations::CRDGame::type() const {
    return "egttools.games.CRDGame";
}

size_t egttools::FinitePopulations::CRDGame::nb_strategies() const {
    return strategies_.size();
}

const egttools::FinitePopulations::GroupPayoffs &egttools::FinitePopulations::CRDGame::calculate_payoffs() {
    StrategyCounts group_composition(nb_strategies_, 0);
    std::vector<double> game_payoffs(nb_strategies_, 0);

    // Initialise matrix
    expected_payoffs_.setZero();

    // For every possible group composition run the game and store the payoff of each strategy
    for (int64_t i = 0; i < nb_states_; ++i) {
        // Update group composition from current state
        egttools::FinitePopulations::sample_simplex(i, group_size_, nb_strategies_, group_composition);

        // play game and update game_payoffs
        play(group_composition, game_payoffs);

        // Fill payoff table
        for (int j = 0; j < nb_strategies_; ++j) expected_payoffs_(j, i) = game_payoffs[j];
    }

    return expected_payoffs_;
}

double egttools::FinitePopulations::CRDGame::calculate_fitness(const int &player_type, const size_t &pop_size,
                                                               const Eigen::Ref<const VectorXui> &strategies) {
    // This function assumes that the strategy counts given in @param strategies does not include
    // the player with @param player_type strategy.
    double fitness = 0.0, payoff;

    // First detect if we are in an edge
    int strategy_1 = player_type;
    int strategy_2;
    int nb_non_zero_strategies = 0;
    for (int ind = 0; ind < nb_strategies_; ++ind) {
        if (strategies(ind) > 0) {
            nb_non_zero_strategies++;
            if (ind != strategy_1) strategy_2 = ind;
        }
    }
    if (nb_non_zero_strategies == 2) {
        fitness = calculate_fitness_edge(strategy_1, strategy_2, pop_size, strategies);
    } else {
        std::vector<size_t> sample_counts(nb_strategies_, 0);

        // If it isn't, then we must calculate the fitness for every possible group combination
        for (int64_t i = 0; i < nb_states_; ++i) {
            // Update sample counts based on the current state
            egttools::FinitePopulations::sample_simplex(i, group_size_, nb_strategies_, sample_counts);

            // If the focal player is not in the group, then the payoff should be zero
            if (sample_counts[player_type] > 0) {
                // First update sample_counts with new group composition
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
    }

    return fitness;
}

double egttools::FinitePopulations::CRDGame::calculate_fitness_edge(const int &strategy_1, const int &strategy_2,
                                                                    const size_t &pop_size, const Eigen::Ref<const VectorXui> &strategies) {
    // This function assumes that the strategy counts given in @param strategies does not include
    // the player with @param player_type strategy.

    double fitness_strategy_1 = 0.0, payoff_strategy_1;
    std::vector<size_t> sample_counts(nb_strategies_, 0);

    // Instead of iterating over all group compositions, we only iterate over group compositions that contain the
    // two strategies
    for (int64_t i = 1; i <= group_size_; ++i) {
        sample_counts[strategy_1] = i;
        sample_counts[strategy_2] = group_size_ - i;

        // get index
        int64_t index = static_cast<int64_t>(egttools::FinitePopulations::calculate_state(group_size_, sample_counts));

        // calculate fitness
        payoff_strategy_1 = expected_payoffs_(strategy_1, index);

        sample_counts[strategy_1] -= 1;
        // Calculate probability of encountering the current group
        auto prob = egttools::multivariateHypergeometricPDF(pop_size - 1, nb_strategies_, group_size_ - 1,
                                                            sample_counts,
                                                            strategies);
        sample_counts[strategy_1] += 1;

        fitness_strategy_1 += payoff_strategy_1 * prob;
    }

    return fitness_strategy_1;
}

void egttools::FinitePopulations::CRDGame::save_payoffs(std::string file_name) const {
    // Save payoffs
    std::ofstream file(file_name, std::ios::out | std::ios::trunc);
    if (file.is_open()) {
        file << "Payoffs for each type of player and each possible state:" << std::endl;
        file << "rows: cooperator, defector, altruist, reciprocal, compensator" << std::endl;
        file << "cols: all possible group compositions starting at (0, 0, 0, 0, group_size)" << std::endl;
        file << expected_payoffs_ << std::endl;
        file << "group_size = " << group_size_ << std::endl;
        file << "timing_uncertainty = false" << std::endl;
        file << "nb_rounds = " << nb_rounds_ << std::endl;
        file << "risk = " << risk_ << std::endl;
        file << "endowment = " << endowment_ << std::endl;
        file << "threshold = " << threshold_ << std::endl;
        file.close();
    }
}

const egttools::FinitePopulations::GroupPayoffs &egttools::FinitePopulations::CRDGame::payoffs() const {
    return expected_payoffs_;
}

double
egttools::FinitePopulations::CRDGame::payoff(int strategy, const egttools::FinitePopulations::StrategyCounts &group_composition) const {
    if (strategy > nb_strategies_)
        throw std::invalid_argument(
                "you must specify a valid index for the strategy [0, " + std::to_string(nb_strategies_) +
                ")");
    if (group_composition.size() != static_cast<size_t>(nb_strategies_))
        throw std::invalid_argument("The group composition must be of size " + std::to_string(nb_strategies_));
    return expected_payoffs_(static_cast<int>(strategy), static_cast<int64_t>(egttools::FinitePopulations::calculate_state(group_size_, group_composition)));
}

void egttools::FinitePopulations::CRDGame::_check_success(size_t state, PayoffVector &game_payoffs,
                                                          const egttools::FinitePopulations::StrategyCounts &group_composition) {
    int prev_donation = 0, current_donation = 0;
    int public_account = 0;
    double fair_endowment = static_cast<double>(endowment_) / 2;
    VectorXi actions = VectorXi::Zero(nb_strategies_);
    auto tmp_state = static_cast<int64_t>(state);

    // Initialize payoffs
    for (int j = 0; j < nb_strategies_; ++j) {
        if (group_composition[j] > 0) {
            game_payoffs[j] = endowment_;
        } else {
            game_payoffs[j] = 0;
        }
    }

    for (int i = 0; i < nb_rounds_; ++i) {
        for (int j = 0; j < nb_strategies_; ++j) {
            if (group_composition[j] > 0) {
                actions(j) = strategies_[j]->get_action(i, prev_donation - actions(j));
                if (game_payoffs[j] >= actions(j)) {
                    game_payoffs[j] -= actions(j);
                    current_donation += static_cast<int>(group_composition[j]) * actions(j);
                }
            }
        }
        public_account += current_donation;
        prev_donation = current_donation;
        current_donation = 0;
        if (public_account >= threshold_) {
            for (int j = 0; j < nb_strategies_; ++j) {
                if (group_composition[j] > 0) {
                    if (game_payoffs[j] > fair_endowment) c_behaviors_(tmp_state, 0) += group_composition[j];
                    else if (game_payoffs[j] < fair_endowment)
                        c_behaviors_(tmp_state, 2) += group_composition[j];
                    else
                        c_behaviors_(tmp_state, 1) += group_composition[j];
                }
            }
            group_achievement_(tmp_state) = 1.0;
            return;
        }
    }

    if (public_account < threshold_)
        group_achievement_(tmp_state) = 0.0;
    else
        group_achievement_(tmp_state) = 1.0;

    for (int j = 0; j < nb_strategies_; ++j) {
        if (group_composition[j] > 0) {
            if (game_payoffs[j] > fair_endowment) c_behaviors_(tmp_state, 0) += group_composition[j];
            else if (game_payoffs[j] < fair_endowment)
                c_behaviors_(tmp_state, 2) += group_composition[j];
            else
                c_behaviors_(tmp_state, 1) += group_composition[j];
        }
    }
}

const egttools::Vector &egttools::FinitePopulations::CRDGame::calculate_success_per_group_composition() {
    StrategyCounts group_composition(nb_strategies_, 0);
    std::vector<double> game_payoffs(nb_strategies_, 0);

    // For every possible group composition run the game and store the payoff of each strategy
    for (int64_t i = 0; i < nb_states_; ++i) {
        // Update group composition from current state
        egttools::FinitePopulations::sample_simplex(i, group_size_, nb_strategies_, group_composition);

        // play game and update group achievement
        _check_success(i, game_payoffs, group_composition);
    }

    return group_achievement_;
}

double egttools::FinitePopulations::CRDGame::calculate_population_group_achievement(size_t pop_size,
                                                                                    const Eigen::Ref<const egttools::VectorXui> &population_state) {
    // This function assumes that the strategy counts given in @param strategies does not include
    // the player with @param player_type strategy.

    double group_achievement = 0.0, success;
    std::vector<size_t> sample_counts(nb_strategies_, 0);

    // If it isn't, then we must calculate the fitness for every possible group combination
    for (int64_t i = 0; i < nb_states_; ++i) {
        // Update sample counts based on the current state
        egttools::FinitePopulations::sample_simplex(i, group_size_, nb_strategies_, sample_counts);

        // First update sample_counts with new group composition
        success = group_achievement_(i);

        // Calculate probability of encountering the current group
        auto prob = egttools::multivariateHypergeometricPDF(pop_size, nb_strategies_, group_size_, sample_counts,
                                                            population_state);

        group_achievement += success * prob;
    }

    return group_achievement;
}

double egttools::FinitePopulations::CRDGame::calculate_group_achievement(size_t pop_size,
                                                                         const Eigen::Ref<const egttools::Vector> &stationary_distribution) {
    double group_achievement = 0;

#pragma omp parallel for default(none) shared(pop_size, stationary_distribution, nb_strategies_) reduction(+ : group_achievement)
    for (long int i = 0; i < stationary_distribution.size(); ++i) {
        VectorXui strategies = VectorXui::Zero(nb_strategies_);
        egttools::FinitePopulations::sample_simplex(i, pop_size, nb_strategies_, strategies);
        group_achievement += stationary_distribution(i) * calculate_population_group_achievement(pop_size, strategies);
    }
    return group_achievement;
}

void egttools::FinitePopulations::CRDGame::calculate_population_polarization(size_t pop_size,
                                                                             const Eigen::Ref<const egttools::VectorXui> &population_state,
                                                                             egttools::Vector3d &polarization) {
    polarization.setZero();
    std::vector<size_t> sample_counts(nb_strategies_, 0);

    // If it isn't, then we must calculate the fitness for every possible group combination
    for (int64_t i = 0; i < nb_states_; ++i) {
        // Update sample counts based on the current state
        egttools::FinitePopulations::sample_simplex(i, group_size_, nb_strategies_, sample_counts);

        // Calculate probability of encountering the current group
        auto prob = egttools::multivariateHypergeometricPDF(pop_size, nb_strategies_, group_size_, sample_counts,
                                                            population_state);

        polarization += (prob * c_behaviors_.row(i).cast<double>()) / group_size_;
    }
}

void egttools::FinitePopulations::CRDGame::calculate_population_polarization_success(size_t pop_size,
                                                                                     const Eigen::Ref<const egttools::VectorXui> &population_state,
                                                                                     egttools::Vector3d &polarization) {
    polarization.setZero();
    std::vector<size_t> sample_counts(nb_strategies_, 0);

    // If it isn't, then we must calculate the fitness for every possible group combination
    for (int64_t i = 0; i < nb_states_; ++i) {
        // Update sample counts based on the current state
        egttools::FinitePopulations::sample_simplex(i, group_size_, nb_strategies_, sample_counts);

        // Calculate probability of encountering the current group
        auto prob = egttools::multivariateHypergeometricPDF(pop_size, nb_strategies_, group_size_, sample_counts,
                                                            population_state);

        polarization += (prob * group_achievement_(i) * c_behaviors_.row(i).cast<double>()) / group_size_;
    }
    auto sum = polarization.sum();
    if (sum > 0) polarization /= sum;
}

egttools::Vector3d egttools::FinitePopulations::CRDGame::calculate_polarization(size_t pop_size,
                                                                                const Eigen::Ref<const egttools::Vector> &stationary_distribution) {
    egttools::Vector3d polarization = egttools::Vector3d::Zero();

#pragma omp parallel for default(none) shared(pop_size, stationary_distribution, nb_strategies_) reduction(+ : polarization)
    for (long int i = 0; i < stationary_distribution.size(); ++i) {
        egttools::Vector3d container = egttools::Vector3d::Zero();
        VectorXui strategies = VectorXui::Zero(nb_strategies_);

        egttools::FinitePopulations::sample_simplex(i, pop_size, nb_strategies_, strategies);
        calculate_population_polarization(pop_size, strategies, container);
        polarization += stationary_distribution(i) * container;
    }
    return polarization / polarization.sum();
}

egttools::Vector3d egttools::FinitePopulations::CRDGame::calculate_polarization_success(size_t pop_size,
                                                                                        const Eigen::Ref<const egttools::Vector> &stationary_distribution) {
    egttools::Vector3d polarization = egttools::Vector3d::Zero();

#pragma omp parallel for default(none) shared(pop_size, stationary_distribution, nb_strategies_) reduction(+ : polarization)
    for (long int i = 0; i < stationary_distribution.size(); ++i) {
        egttools::Vector3d container = egttools::Vector3d::Zero();
        VectorXui strategies = VectorXui::Zero(nb_strategies_);

        egttools::FinitePopulations::sample_simplex(i, pop_size, nb_strategies_, strategies);
        calculate_population_polarization_success(pop_size, strategies, container);
        polarization += stationary_distribution(i) * container;
    }
    return polarization;
}

const egttools::Vector &egttools::FinitePopulations::CRDGame::group_achievements() const {
    return group_achievement_;
}

const egttools::MatrixXui2D &egttools::FinitePopulations::CRDGame::contribution_behaviors() const {
    return c_behaviors_;
}

int egttools::FinitePopulations::CRDGame::target() const {
    return threshold_;
}

int egttools::FinitePopulations::CRDGame::endowment() const {
    return endowment_;
}

int egttools::FinitePopulations::CRDGame::nb_rounds() const {
    return nb_rounds_;
}

size_t egttools::FinitePopulations::CRDGame::group_size() const {
    return group_size_;
}

double egttools::FinitePopulations::CRDGame::risk() const {
    return risk_;
}

double egttools::FinitePopulations::CRDGame::enhancement_factor() const {
    return enhancement_factor_;
}

size_t egttools::FinitePopulations::CRDGame::nb_states() const {
    return nb_states_;
}
const egttools::FinitePopulations::CRDStrategyVector &egttools::FinitePopulations::CRDGame::strategies() const {
    return strategies_;
}

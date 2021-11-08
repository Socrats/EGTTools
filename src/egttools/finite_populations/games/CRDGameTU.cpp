//
// Created by Elias Fernandez on 2019-06-27.
//

#include <egttools/finite_populations/games/CRDGameTU.hpp>

egttools::FinitePopulations::games::CRDGameTU::CRDGameTU(int endowment, int threshold,
                                                         int min_rounds, int group_size,
                                                         double risk,
                                                         egttools::utils::TimingUncertainty<std::mt19937_64> &tu,
                                                         const CRDStrategyVector &strategies)
    : endowment_(endowment),
      threshold_(threshold),
      min_rounds_(min_rounds),
      group_size_(group_size),
      risk_(risk),
      tu_(tu),
      strategies_(strategies) {

    // First we check how many strategies will be in the game
    nb_strategies_ = static_cast<int>(strategies.size());

    // number of possible group combinations
    nb_states_ = egttools::starsBars<int64_t>(group_size_, nb_strategies_);

    expected_payoffs_ = GroupPayoffs::Zero(nb_strategies_, nb_states_);
    group_achievement_ = egttools::Vector::Zero(nb_states_);
    c_behaviors_counts_ = egttools::MatrixXui2D::Zero(nb_states_, 3);
    c_behaviors_ = egttools::Matrix2D::Zero(nb_states_, 3);

    // Initialise payoff matrix
    calculate_payoffs();

    // Initialise group achievement vector
    calculate_success_per_group_composition();
}

void egttools::FinitePopulations::games::CRDGameTU::play(const egttools::FinitePopulations::StrategyCounts &group_composition,
                                                         PayoffVector &game_payoffs) {
    int prev_donation = 0, current_donation = 0;
    int public_account = 0;
    int game_rounds = tu_.calculate_full_end(min_rounds_, generator_);
    VectorXi actions = VectorXi::Zero(nb_strategies_);

    // Initialize payoffs
    for (int j = 0; j < nb_strategies_; ++j) {
        if (group_composition[j] > 0) {
            game_payoffs[j] = endowment_;
        } else {
            game_payoffs[j] = 0;
        }
    }

    for (int i = 0; i < game_rounds; ++i) {
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
        if (public_account >= threshold_) break;
    }

    // Calculate expected payoffs from risk
    if (public_account < threshold_)
        for (auto &type : game_payoffs) type *= (1.0 - risk_);
}

std::string egttools::FinitePopulations::games::CRDGameTU::toString() const {
    return "Collective-risk dilemma game with Timing uncertainty.\n"
           "See Fernández Domingos et al. 2020.";
}

std::string egttools::FinitePopulations::games::CRDGameTU::type() const {
    return "egttools::FinitePopulations::games::CRDGameTU";
}

size_t egttools::FinitePopulations::games::CRDGameTU::nb_strategies() const {
    return strategies_.size();
}

const egttools::FinitePopulations::GroupPayoffs &egttools::FinitePopulations::games::CRDGameTU::calculate_payoffs() {
    // Initialise matrix
    expected_payoffs_.setZero();

#pragma omp parallel for default(none) shared(nb_strategies_, group_size_, expected_payoffs_)
    // For every possible group composition run the game and store the payoff of each strategy
    for (int64_t i = 0; i < nb_states_; ++i) {
        StrategyCounts group_composition(nb_strategies_, 0);
        std::vector<double> game_payoffs(nb_strategies_, 0);

        // Update group composition from current state
        egttools::FinitePopulations::sample_simplex(i, group_size_, nb_strategies_, group_composition);

        // Since the number of rounds of the game is stochastic
        // we repeat the game 10000 times to obtain a good estimation
        for (size_t z = 0; z < 10000; ++z) {
            // play game and update game_payoffs
            play(group_composition, game_payoffs);

            // Fill payoff table
            for (int j = 0; j < nb_strategies_; ++j) expected_payoffs_(j, i) += game_payoffs[j];
        }
        // avg. the results
        expected_payoffs_.col(i) /= 10000;
    }

    return expected_payoffs_;
}

double egttools::FinitePopulations::games::CRDGameTU::calculate_fitness(const int &player_type, const size_t &pop_size,
                                                                        const Eigen::Ref<const VectorXui> &strategies) {
    // This function assumes that the strategy counts given in @param strategies does not include
    // the player with @param player_type strategy.

    double fitness = 0.0, payoff;
    std::vector<size_t> sample_counts(nb_strategies_, 0);

    // If it isn't, then we must calculate the fitness for every possible group combination
    for (int64_t i = 0; i < nb_states_; ++i) {
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

void egttools::FinitePopulations::games::CRDGameTU::save_payoffs(std::string file_name) const {
    // Save payoffs
    std::ofstream file(file_name, std::ios::out | std::ios::trunc);
    if (file.is_open()) {
        file << "Payoffs for each type of player and each possible state:" << std::endl;
        file << "rows: cooperator, defector, altruist, reciprocal, compensator" << std::endl;
        file << "cols: all possible group compositions starting at (0, 0, 0, 0, group_size)" << std::endl;
        file << expected_payoffs_ << std::endl;
        file << "group_size = " << group_size_ << std::endl;
        file << "timing_uncertainty = false" << std::endl;
        file << "min_rounds = " << min_rounds_ << std::endl;
        file << "p = " << tu_.probability() << std::endl;
        file << "risk = " << risk_ << std::endl;
        file << "endowment = " << endowment_ << std::endl;
        file << "threshold = " << threshold_ << std::endl;
        file.close();
    }
}

const egttools::FinitePopulations::GroupPayoffs &egttools::FinitePopulations::games::CRDGameTU::payoffs() const {
    return expected_payoffs_;
}

double
egttools::FinitePopulations::games::CRDGameTU::payoff(int strategy, const egttools::FinitePopulations::StrategyCounts &group_composition) const {
    if (strategy > nb_strategies_)
        throw std::invalid_argument(
                "you must specify a valid index for the strategy [0, " + std::to_string(nb_strategies_) +
                ")");
    if (group_composition.size() != static_cast<size_t>(nb_strategies_))
        throw std::invalid_argument("The group composition must be of size " + std::to_string(nb_strategies_));
    return expected_payoffs_(static_cast<int>(strategy), static_cast<int64_t>(egttools::FinitePopulations::calculate_state(group_size_, group_composition)));
}

void egttools::FinitePopulations::games::CRDGameTU::_check_success(size_t state, PayoffVector &game_payoffs,
                                                                   const egttools::FinitePopulations::StrategyCounts &group_composition) {
    int prev_donation = 0, current_donation = 0;
    int public_account = 0;
    double fair_endowment = static_cast<double>(endowment_) / 2;
    int game_rounds = tu_.calculate_full_end(min_rounds_, generator_);
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

    // Play the game and find out whether the group is successful
    for (int i = 0; i < game_rounds; ++i) {
        // Let each strategy choose an action (this assumes that the action is not stochastic)
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
            group_achievement_(tmp_state) += 1.0;
            break;
        }
    }

    for (int j = 0; j < nb_strategies_; ++j) {
        if (group_composition[j] > 0) {
            if (game_payoffs[j] > fair_endowment)
                c_behaviors_counts_(tmp_state, 0) += group_composition[j];
            else if (game_payoffs[j] < fair_endowment)
                c_behaviors_counts_(tmp_state, 2) += group_composition[j];
            else
                c_behaviors_counts_(tmp_state, 1) += group_composition[j];
        }
    }
}

const egttools::Vector &egttools::FinitePopulations::games::CRDGameTU::calculate_success_per_group_composition() {
    c_behaviors_counts_.setZero();
    group_achievement_.setZero();

#pragma omp parallel for default(none) shared(nb_strategies_, group_size_, group_achievement_, c_behaviors_counts_, c_behaviors_)
    // For every possible group composition run the game and store the payoff of each strategy
    for (int64_t i = 0; i < nb_states_; ++i) {
        StrategyCounts group_composition(nb_strategies_, 0);
        std::vector<double> game_payoffs(nb_strategies_, 0);

        // Update group composition from current state
        egttools::FinitePopulations::sample_simplex(i, group_size_, nb_strategies_, group_composition);

        // Since the number of rounds of the game is stochastic
        // we repeat the game a 10000 times to obtain a good estimation
        for (size_t z = 0; z < 10000; ++z) {
            // play game and update group achievement
            _check_success(i, game_payoffs, group_composition);
        }
        group_achievement_(i) /= 10000;
        c_behaviors_.row(i) = c_behaviors_counts_.row(i).cast<double>() / (10000 * group_size_);
    }

    return group_achievement_;
}

double egttools::FinitePopulations::games::CRDGameTU::calculate_population_group_achievement(size_t pop_size,
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

double egttools::FinitePopulations::games::CRDGameTU::calculate_group_achievement(size_t pop_size,
                                                                                  const Eigen::Ref<const egttools::Vector> &stationary_distribution) {
    double group_achievement = 0;

#pragma omp parallel for default(none) shared(pop_size, stationary_distribution, nb_strategies_) reduction(+ \
                                                                                                           : group_achievement)
    for (long int i = 0; i < stationary_distribution.size(); ++i) {
        VectorXui strategies = VectorXui::Zero(nb_strategies_);
        egttools::FinitePopulations::sample_simplex(i, pop_size, nb_strategies_, strategies);
        group_achievement += stationary_distribution(i) * calculate_population_group_achievement(pop_size, strategies);
    }
    return group_achievement;
}

void egttools::FinitePopulations::games::CRDGameTU::calculate_population_polarization(size_t pop_size,
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

        polarization += prob * c_behaviors_.row(i);
    }
}

void egttools::FinitePopulations::games::CRDGameTU::calculate_population_polarization_success(size_t pop_size,
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

        polarization += prob * c_behaviors_.row(i) * group_achievement_(i);
    }

    auto sum = polarization.sum();
    if (sum > 0) polarization /= sum;
}

egttools::Vector3d egttools::FinitePopulations::games::CRDGameTU::calculate_polarization(size_t pop_size,
                                                                                         const Eigen::Ref<const egttools::Vector> &stationary_distribution) {
    egttools::Vector3d polarization = egttools::Vector3d::Zero();

#pragma omp parallel for default(none) shared(pop_size, stationary_distribution, nb_strategies_) reduction(+ \
                                                                                                           : polarization)
    for (long int i = 0; i < stationary_distribution.size(); ++i) {
        egttools::Vector3d container = egttools::Vector3d::Zero();
        VectorXui strategies = VectorXui::Zero(nb_strategies_);

        egttools::FinitePopulations::sample_simplex(i, pop_size, nb_strategies_, strategies);
        calculate_population_polarization(pop_size, strategies, container);
        polarization += stationary_distribution(i) * container;
    }
    return polarization;
}

egttools::Vector3d egttools::FinitePopulations::games::CRDGameTU::calculate_polarization_success(size_t pop_size,
                                                                                                 const Eigen::Ref<const egttools::Vector> &stationary_distribution) {
    egttools::Vector3d polarization = egttools::Vector3d::Zero();

#pragma omp parallel for default(none) shared(pop_size, stationary_distribution, nb_strategies_) reduction(+ \
                                                                                                           : polarization)
    for (long int i = 0; i < stationary_distribution.size(); ++i) {
        egttools::Vector3d container = egttools::Vector3d::Zero();
        VectorXui strategies = VectorXui::Zero(nb_strategies_);

        egttools::FinitePopulations::sample_simplex(i, pop_size, nb_strategies_, strategies);
        calculate_population_polarization_success(pop_size, strategies, container);
        polarization += stationary_distribution(i) * container;
    }
    return polarization;
}

const egttools::Vector &egttools::FinitePopulations::games::CRDGameTU::group_achievements() const {
    return group_achievement_;
}

const egttools::Matrix2D &egttools::FinitePopulations::games::CRDGameTU::contribution_behaviors() const {
    return c_behaviors_;
}

size_t egttools::FinitePopulations::games::CRDGameTU::target() const {
    return threshold_;
}

size_t egttools::FinitePopulations::games::CRDGameTU::endowment() const {
    return endowment_;
}

size_t egttools::FinitePopulations::games::CRDGameTU::min_rounds() const {
    return min_rounds_;
}

size_t egttools::FinitePopulations::games::CRDGameTU::group_size() const {
    return group_size_;
}

double egttools::FinitePopulations::games::CRDGameTU::risk() const {
    return risk_;
}

size_t egttools::FinitePopulations::games::CRDGameTU::nb_states() const {
    return nb_states_;
}
const egttools::FinitePopulations::games::CRDStrategyVector &egttools::FinitePopulations::games::CRDGameTU::strategies() const {
    return strategies_;
}

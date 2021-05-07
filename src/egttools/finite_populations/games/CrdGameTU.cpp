//
// Created by Elias Fernandez on 2019-06-27.
//

#include <Dyrwin/SED/games/CrdGameTU.hpp>

EGTTools::SED::CRD::CrdGameTU::CrdGameTU(size_t endowment, size_t threshold, size_t min_rounds, size_t group_size,
                                         double risk,
                                         EGTTools::TimingUncertainty<std::mt19937_64> &tu)
        : endowment_(endowment),
          threshold_(threshold), min_rounds_(min_rounds), group_size_(group_size), risk_(risk), tu_(tu) {
    nb_strategies_ = nb_strategies();
    // number of possible group combinations
    nb_states_ = EGTTools::starsBars(group_size_, nb_strategies_);
    payoffs_ = GroupPayoffs::Zero(nb_strategies_, nb_states_);
    group_achievement_ = EGTTools::Vector::Zero(nb_states_);
    c_behaviors_ = EGTTools::Matrix2D::Zero(nb_states_, 3);

    // initialise random distribution
    real_rand_ = std::uniform_real_distribution<double>(0.0, 1.0);

    // Initialise payoff matrix
    calculate_payoffs();
    // Initialise group achievement vector
    calculate_success_per_group_composition();
}

void
EGTTools::SED::CRD::CrdGameTU::play(const EGTTools::SED::StrategyCounts &group_composition,
                                    PayoffVector &game_payoffs) {
    size_t prev_donation = 0, current_donation = 0;
    size_t public_account = 0;
    size_t player_aspiration = (group_size_ - 1) * 2;
    size_t game_rounds = tu_.calculateFullEnd(min_rounds_, generator_);
    VectorXui actions = VectorXui::Zero(nb_strategies_);

    // Initialize payoffs
    for (size_t j = 0; j < EGTTools::SED::CRD::nb_strategies; ++j) {
        if (group_composition[j] > 0) {
            game_payoffs[j] = endowment_;
        } else {
            game_payoffs[j] = 0;
        }
    }

    for (size_t i = 0; i < game_rounds; ++i) {
        for (size_t j = 0; j < EGTTools::SED::CRD::nb_strategies; ++j) {
            if (group_composition[j] > 0) {
                actions(j) = get_action(j, prev_donation - actions(j), player_aspiration, i);
                if (game_payoffs[j] >= actions(j)) {
                    game_payoffs[j] -= actions(j);
                    current_donation += group_composition[j] * actions(j);
                }
            }
        }
        public_account += current_donation;
        prev_donation = current_donation;
        current_donation = 0;
        if (public_account >= threshold_) break;
    }

    if (public_account < threshold_)
        for (auto &type: game_payoffs) type *= (1.0 - risk_);
}

size_t
EGTTools::SED::CRD::CrdGameTU::get_action(const size_t &player_type, const size_t &prev_donation,
                                          const size_t &threshold,
                                          const size_t &current_round) {
    switch (player_type) {
        case 0:
            return EGTTools::SED::CRD::cooperator(prev_donation, threshold, current_round);
        case 1:
            return EGTTools::SED::CRD::defector(prev_donation, threshold, current_round);
        case 2:
            return EGTTools::SED::CRD::altruist(prev_donation, threshold, current_round);
        case 3:
            return EGTTools::SED::CRD::reciprocal(prev_donation, threshold, current_round);
        case 4:
            return EGTTools::SED::CRD::compensator(prev_donation, threshold, current_round);
        default:
            assert(false);
            throw std::invalid_argument("invalid player type: " + std::to_string(player_type));
    }
}

std::string EGTTools::SED::CRD::CrdGameTU::toString() const {
    return "Collective-risk dilemma game with Timing uncertainty.\n"
           "It only plays the game with the strategies described in EGTtools::SED::CRD::behaviors";
}

std::string EGTTools::SED::CRD::CrdGameTU::type() const {
    return "CrdGameTU";
}

size_t EGTTools::SED::CRD::CrdGameTU::nb_strategies() const {
    return EGTTools::SED::CRD::nb_strategies;
}

const EGTTools::SED::GroupPayoffs &EGTTools::SED::CRD::CrdGameTU::calculate_payoffs() {
    StrategyCounts group_composition(nb_strategies_, 0);
    std::vector<double> game_payoffs(nb_strategies_, 0);
    // Initialise matrix
    payoffs_.setZero();

    // For every possible group composition run the game and store the payoff of each strategy
    for (size_t i = 0; i < nb_states_; ++i) {
        // Update group composition from current state
        EGTTools::SED::sample_simplex(i, group_size_, nb_strategies_, group_composition);

        // Since the number of rounds of the game is stochastic
        // we repeat the game a 10000 times to obtain a good estimation
        for (size_t z = 0; z < 10000; ++z) {
            // play game and update game_payoffs
            play(group_composition, game_payoffs);

            // Fill payoff table
            for (size_t j = 0; j < nb_strategies_; ++j) payoffs_(j, i) += game_payoffs[j];
        }
        payoffs_.col(i) /= 10000;
    }

    return payoffs_;
}

double EGTTools::SED::CRD::CrdGameTU::calculate_fitness(const size_t &player_type, const size_t &pop_size,
                                                        const Eigen::Ref<const VectorXui> &strategies) {
    // This function assumes that the strategy counts given in @param strategies does not include
    // the player with @param player_type strategy.

    double fitness = 0.0, payoff;
    std::vector<size_t> sample_counts(nb_strategies_, 0);

    // If it isn't, then we must calculate the fitness for every possible group combination
    for (size_t i = 0; i < nb_states_; ++i) {
        // Update sample counts based on the current state
        EGTTools::SED::sample_simplex(i, group_size_, nb_strategies_, sample_counts);

        // If the focal player is not in the group, then the payoff should be zero
        if (sample_counts[player_type] > 0) {

            // First update sample_counts with new group composition
            payoff = payoffs_(player_type, i);
            sample_counts[player_type] -= 1;

            // Calculate probability of encountering a the current group
            auto prob = EGTTools::multivariateHypergeometricPDF(pop_size - 1, nb_strategies_, group_size_ - 1,
                                                                sample_counts,
                                                                strategies);
            sample_counts[player_type] += 1;

            fitness += payoff * prob;
        }
    }

    return fitness;
}

void EGTTools::SED::CRD::CrdGameTU::save_payoffs(std::string file_name) const {
    // Save payoffs
    std::ofstream file(file_name, std::ios::out | std::ios::trunc);
    if (file.is_open()) {
        file << "Payoffs for each type of player and each possible state:" << std::endl;
        file << "rows: cooperator, defector, altruist, reciprocal, compensator" << std::endl;
        file << "cols: all possible group compositions starting at (0, 0, 0, 0, group_size)" << std::endl;
        file << payoffs_ << std::endl;
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

const EGTTools::SED::GroupPayoffs &EGTTools::SED::CRD::CrdGameTU::payoffs() const {
    return payoffs_;
}

double
EGTTools::SED::CRD::CrdGameTU::payoff(size_t strategy, const EGTTools::SED::StrategyCounts &group_composition) const {
    return payoffs_(strategy, EGTTools::SED::calculate_state(group_size_, group_composition));
}

double EGTTools::SED::CRD::CrdGameTU::_check_success(PayoffVector &game_payoffs,
                                                     const EGTTools::SED::StrategyCounts &group_composition) {
    size_t prev_donation = 0, current_donation = 0;
    size_t public_account = 0;
    size_t player_aspiration = (group_size_ - 1) * 2;
    size_t game_rounds = tu_.calculateFullEnd(min_rounds_, generator_);
    VectorXui actions = VectorXui::Zero(nb_strategies_);

    // Initialize payoffs
    for (size_t j = 0; j < EGTTools::SED::CRD::nb_strategies; ++j) {
        if (group_composition[j] > 0) {
            game_payoffs[j] = endowment_;
        } else {
            game_payoffs[j] = 0;
        }
    }

    for (size_t i = 0; i < game_rounds; ++i) {
        for (size_t j = 0; j < EGTTools::SED::CRD::nb_strategies; ++j) {
            if (group_composition[j] > 0) {
                actions(j) = get_action(j, prev_donation - actions(j), player_aspiration, i);
                if (game_payoffs[j] >= actions(j)) {
                    game_payoffs[j] -= actions(j);
                    current_donation += group_composition[j] * actions(j);
                }
            }
        }
        public_account += current_donation;
        prev_donation = current_donation;
        current_donation = 0;
        if (public_account >= threshold_) {
            return 1.0;
        }
    }

    if (public_account < threshold_)
        return 0.0;
    else return 1.0;
}

const EGTTools::Vector &EGTTools::SED::CRD::CrdGameTU::calculate_success_per_group_composition() {
    StrategyCounts group_composition(nb_strategies_, 0);
    std::vector<double> game_payoffs(nb_strategies_, 0);
    double fair_endowment = static_cast<double>(endowment_) / 2;

    // For every possible group composition run the game and store the payoff of each strategy
    for (size_t i = 0; i < nb_states_; ++i) {
        // Update group composition from current state
        EGTTools::SED::sample_simplex(i, group_size_, nb_strategies_, group_composition);

        // Since the number of rounds of the game is stochastic
        // we repeat the game a 10000 times to obtain a good estimation
        for (size_t z = 0; z < 10000; ++z) {
            // play game and update group achievement
            group_achievement_(i) += _check_success(game_payoffs, group_composition);
            for (size_t j = 0; j < nb_strategies_; ++j) {
                if (group_composition[j] > 0) {
                    if (game_payoffs[j] > fair_endowment) c_behaviors_(i, 0) += group_composition[j];
                    else if (game_payoffs[j] < fair_endowment) c_behaviors_(i, 2) += group_composition[j];
                    else c_behaviors_(i, 1) += group_composition[j];
                }
            }
        }
        group_achievement_(i) /= 10000;
        c_behaviors_.row(i) /= (10000 * group_size_);
    }

    return group_achievement_;
}

double EGTTools::SED::CRD::CrdGameTU::calculate_population_group_achievement(size_t pop_size,
                                                                             const Eigen::Ref<const EGTTools::VectorXui> &population_state) {
    // This function assumes that the strategy counts given in @param strategies does not include
    // the player with @param player_type strategy.

    double group_achievement = 0.0, success;
    std::vector<size_t> sample_counts(nb_strategies_, 0);

    // If it isn't, then we must calculate the fitness for every possible group combination
    for (size_t i = 0; i < nb_states_; ++i) {
        // Update sample counts based on the current state
        EGTTools::SED::sample_simplex(i, group_size_, nb_strategies_, sample_counts);

        // First update sample_counts with new group composition
        success = group_achievement_(EGTTools::SED::calculate_state(group_size_, sample_counts));

        // Calculate probability of encountering the current group
        auto prob = EGTTools::multivariateHypergeometricPDF(pop_size, nb_strategies_, group_size_, sample_counts,
                                                            population_state);

        group_achievement += success * prob;
    }

    return group_achievement;
}

double EGTTools::SED::CRD::CrdGameTU::calculate_group_achievement(size_t pop_size,
                                                                  const Eigen::Ref<const EGTTools::Vector> &stationary_distribution) {
    double group_achievement = 0;

#pragma omp parallel for default(none) shared(pop_size, stationary_distribution, nb_strategies_) reduction(+:group_achievement)
    for (long int i = 0; i < stationary_distribution.size(); ++i) {
        VectorXui strategies = VectorXui::Zero(nb_strategies_);
        EGTTools::SED::sample_simplex(i, pop_size, nb_strategies_, strategies);
        group_achievement += stationary_distribution(i) * calculate_population_group_achievement(pop_size, strategies);
    }
    return group_achievement;
}

void EGTTools::SED::CRD::CrdGameTU::calculate_population_polarization(size_t pop_size,
                                                                      const Eigen::Ref<const EGTTools::VectorXui> &population_state,
                                                                      EGTTools::Vector3d &polarization) {
    polarization.setZero();
    std::vector<size_t> sample_counts(nb_strategies_, 0);

    // If it isn't, then we must calculate the fitness for every possible group combination
    for (size_t i = 0; i < nb_states_; ++i) {
        // Update sample counts based on the current state
        EGTTools::SED::sample_simplex(i, group_size_, nb_strategies_, sample_counts);

        // Calculate probability of encountering the current group
        auto prob = EGTTools::multivariateHypergeometricPDF(pop_size, nb_strategies_, group_size_, sample_counts,
                                                            population_state);

        polarization += prob * c_behaviors_.row(i);
    }
}

void EGTTools::SED::CRD::CrdGameTU::calculate_population_polarization_success(size_t pop_size,
                                                                              const Eigen::Ref<const EGTTools::VectorXui> &population_state,
                                                                              EGTTools::Vector3d &polarization) {
    polarization.setZero();
    std::vector<size_t> sample_counts(nb_strategies_, 0);

    // If it isn't, then we must calculate the fitness for every possible group combination
    for (size_t i = 0; i < nb_states_; ++i) {
        // Update sample counts based on the current state
        EGTTools::SED::sample_simplex(i, group_size_, nb_strategies_, sample_counts);

        // Calculate probability of encountering the current group
        auto prob = EGTTools::multivariateHypergeometricPDF(pop_size, nb_strategies_, group_size_, sample_counts,
                                                            population_state);

        polarization += prob * c_behaviors_.row(i) * group_achievement_(i);
    }

    auto sum = polarization.sum();
    if (sum > 0) polarization /= sum;
}

EGTTools::Vector3d EGTTools::SED::CRD::CrdGameTU::calculate_polarization(size_t pop_size,
                                                                         const Eigen::Ref<const EGTTools::Vector> &stationary_distribution) {
    EGTTools::Vector3d polarization = EGTTools::Vector3d::Zero();

#pragma omp parallel for default(none) shared(pop_size, stationary_distribution, nb_strategies_) reduction(+:polarization)
    for (long int i = 0; i < stationary_distribution.size(); ++i) {
        EGTTools::Vector3d container = EGTTools::Vector3d::Zero();
        VectorXui strategies = VectorXui::Zero(nb_strategies_);

        EGTTools::SED::sample_simplex(i, pop_size, nb_strategies_, strategies);
        calculate_population_polarization(pop_size, strategies, container);
        polarization += stationary_distribution(i) * container;
    }
    return polarization;
}

EGTTools::Vector3d EGTTools::SED::CRD::CrdGameTU::calculate_polarization_success(size_t pop_size,
                                                                                 const Eigen::Ref<const EGTTools::Vector> &stationary_distribution) {
    EGTTools::Vector3d polarization = EGTTools::Vector3d::Zero();

#pragma omp parallel for default(none) shared(pop_size, stationary_distribution, nb_strategies_) reduction(+:polarization)
    for (long int i = 0; i < stationary_distribution.size(); ++i) {
        EGTTools::Vector3d container = EGTTools::Vector3d::Zero();
        VectorXui strategies = VectorXui::Zero(nb_strategies_);

        EGTTools::SED::sample_simplex(i, pop_size, nb_strategies_, strategies);
        calculate_population_polarization_success(pop_size, strategies, container);
        polarization += stationary_distribution(i) * container;
    }
    return polarization / polarization.sum();
}

const EGTTools::Vector &EGTTools::SED::CRD::CrdGameTU::group_achievements() const {
    return group_achievement_;
}

const EGTTools::Matrix2D &EGTTools::SED::CRD::CrdGameTU::contribution_behaviors() const {
    return c_behaviors_;
}

size_t EGTTools::SED::CRD::CrdGameTU::target() const {
    return threshold_;
}

size_t EGTTools::SED::CRD::CrdGameTU::endowment() const {
    return endowment_;
}

size_t EGTTools::SED::CRD::CrdGameTU::min_rounds() const {
    return min_rounds_;
}

size_t EGTTools::SED::CRD::CrdGameTU::group_size() const {
    return group_size_;
}

double EGTTools::SED::CRD::CrdGameTU::risk() const {
    return risk_;
}

size_t EGTTools::SED::CRD::CrdGameTU::nb_states() const {
    return nb_states_;
}

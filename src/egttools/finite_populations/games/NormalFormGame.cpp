//
// Created by Elias Fernandez on 29/12/2020.
//
#include <egttools/finite_populations/games/NormalFormGame.h>

egttools::FinitePopulations::NormalFormGame::NormalFormGame(size_t nb_rounds,
                                                            const Eigen::Ref<const Matrix2D> &payoff_matrix)
        : nb_rounds_(nb_rounds),
          payoffs_(
                  payoff_matrix) {

    // First we check how many strategies will be in the game
    // We consider only 2 for now (Cooperate and Defect)
    nb_strategies_ = 2;
    // nb_states_ will give the amount of game combinations (between) strategies that can happen
    nb_states_ = egttools::starsBars(2, nb_strategies_);
    // Calculate the number of possible states
    expected_payoffs_ = Matrix2D::Zero(nb_strategies_, nb_strategies_);
    coop_level_ = Matrix2D::Zero(nb_strategies_, nb_strategies_);

    // Initialize payoff matrix
    egttools::FinitePopulations::NormalFormGame::calculate_payoffs();
    // Initialize cooperation level vector
}

void
egttools::FinitePopulations::NormalFormGame::play(const egttools::FinitePopulations::StrategyCounts &group_composition,
                                                  egttools::FinitePopulations::PayoffVector &game_payoffs) {
    // Initialize payoffs
    for (auto &value : game_payoffs) value = 0;

    for (size_t i = 0; i < nb_rounds_; ++i) {
        // For now since we will consider only C and D as strategies, we will pre-calculate all actions since they
        // correspond with the strategies
        game_payoffs[0] += payoffs_(group_composition[0], group_composition[1]);
        game_payoffs[1] += payoffs_(group_composition[1], group_composition[2]);
    }
    game_payoffs[0] /= nb_rounds_;
    game_payoffs[1] /= nb_rounds_;
}

const egttools::FinitePopulations::GroupPayoffs &egttools::FinitePopulations::NormalFormGame::calculate_payoffs() {
    size_t s1, s2;

    // For every possible pair combination, run the game and store the payoff of each strategy
    for (size_t i = 0; i < nb_strategies_; ++i) {
        s1 = i;
        for (size_t j = i; j < nb_strategies_; ++j) {
            // Update group composition from current
            s2 = j;

            // play game and update game_payoffs
            _update_cooperation_and_payoffs(s1, s2);
        }
    }

    return expected_payoffs_;
}

double egttools::FinitePopulations::NormalFormGame::calculate_fitness(const size_t &player_type,
                                                                      const size_t &pop_size,
                                                                      const Eigen::Ref<const VectorXui> &strategies) {
    // This function assumes that the strategy counts given in @param strategies does not include
    // the player with @param player_type strategy.
    double fitness = 0.0;
    size_t divider = pop_size - 1;

    // Given that we are only considering 2 player games, the fitness is simply the
    // expected payoff of each strategy * the probability of finding that strategy in the
    // population
    for (size_t i = 0; i < nb_strategies_; ++i) {
        fitness += expected_payoffs_(player_type, i) * (strategies(i) / static_cast<double>(divider));
    }

    return fitness;
}

double egttools::FinitePopulations::NormalFormGame::calculate_cooperation_level(size_t pop_size,
                                                                                const Eigen::Ref<const VectorXui> &population_state) {
    // The idea here is to count how often a strategy cooperates
    double coop_level = 0.0, prob;
    auto pop_size_double = static_cast<double>(pop_size);
    auto pop_minus_one = static_cast<double>(pop_size - 1);

    // So for each strategy pair we will count the level of cooperation, and then check
    // how likely each game pair is given the population state
    // For every possible pair combination, run the game and store the payoff of each strategy
    for (size_t i = 0; i < nb_strategies_; ++i) {
        for (size_t j = i; j < nb_strategies_; ++j) {
            if (i == j) // prob A1 * prob A2
                prob = (population_state(i) / pop_size_double) *
                       (static_cast<double>(population_state(i) - 1) / pop_minus_one);
            else // (prob A1 * prob B2) + (prob B1 * prob A2)
                prob = ((population_state(i) / pop_size_double) *
                        (population_state(j) / pop_minus_one)) +
                       ((population_state(j) / pop_size_double) *
                        (population_state(i) / pop_minus_one));
            coop_level += prob * (coop_level_(i, j) + coop_level_(j, i)) / 2;
        }
    }

    return coop_level;
}

size_t egttools::FinitePopulations::NormalFormGame::nb_strategies() const {
    return nb_strategies_;
}

size_t egttools::FinitePopulations::NormalFormGame::nb_rounds() const {
    return nb_rounds_;
}

size_t egttools::FinitePopulations::NormalFormGame::nb_states() const {
    return nb_states_;
}

std::string egttools::FinitePopulations::NormalFormGame::toString() const {
    return "Normal-form Game.\n"
           "For now the only available strategies are C and D";
}

std::string egttools::FinitePopulations::NormalFormGame::type() const {
    return "NormalFormGame";
}

const egttools::FinitePopulations::GroupPayoffs &egttools::FinitePopulations::NormalFormGame::payoffs() const {
    return payoffs_;
}

const egttools::Matrix2D &egttools::FinitePopulations::NormalFormGame::expected_payoffs() const {
    return expected_payoffs_;
}

double egttools::FinitePopulations::NormalFormGame::payoff(size_t strategy,
                                                           const egttools::FinitePopulations::StrategyCounts &group_composition) const {
    if (strategy > nb_strategies_)
        throw std::invalid_argument(
                "you must specify a valid index for the strategy [0, " + std::to_string(nb_strategies_) +
                ")");
    return expected_payoffs_(group_composition[0], group_composition[1]);
}

void egttools::FinitePopulations::NormalFormGame::save_payoffs(std::string file_name) const {
    // Save payoffs
    std::ofstream file(file_name, std::ios::out | std::ios::trunc);
    if (file.is_open()) {
        file << "Payoffs for strategy against each other:" << std::endl;
        file << "Each entry represents the payoff of the row strategy against the column strategy" << std::endl;
        file << expected_payoffs_ << std::endl;
        file << "--------------------" << std::endl;
        file << "Original payoff matrix: " << std::endl;
        file << payoffs_ << std::endl;
        file << "--------------------" << std::endl;
        file << "parameters:" << std::endl;
        file << "nb_rounds = " << nb_rounds_ << std::endl;
        file.close();
    }
}

void egttools::FinitePopulations::NormalFormGame::_update_cooperation_and_payoffs(size_t s1, size_t s2) {
    // Initialize payoffs
    size_t coop1 = 0, coop2 = 0;

    // This should be repeated many times if the strategies are stochastic
    for (size_t i = 0; i < nb_rounds_; ++i) {
        // For now since we will consider only C and D as strategies, we will pre-calculate all actions since they
        // correspond with the strategies
        expected_payoffs_(s1, s2) += payoffs_(s1, s2);
        coop1 += s1;
        if (s1 != s2) {
            expected_payoffs_(s2, s1) += payoffs_(s2, s1);
            coop2 += s2;
        }
    }

    if (s1 == s2) {
        expected_payoffs_(s1, s1) /= nb_rounds_;
        coop_level_(s1, s1) /= nb_rounds_;
    } else {
        expected_payoffs_(s1, s2) /= nb_rounds_;
        expected_payoffs_(s2, s1) /= nb_rounds_;
        coop_level_(s1, s2) = static_cast<double>(coop1) / nb_rounds_;
        coop_level_(s2, s1) = static_cast<double>(coop2) / nb_rounds_;
    }
}

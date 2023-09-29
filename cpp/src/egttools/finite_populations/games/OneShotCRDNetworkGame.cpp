//
// Created by Elias Fernandez on 02/05/2023.
//
#include <egttools/finite_populations/games/OneShotCRDNetworkGame.hpp>

egttools::FinitePopulations::games::OneShotCRDNetworkGame::OneShotCRDNetworkGame(double endowment,
                                                                                 double cost,
                                                                                 double risk,
                                                                                 int min_nb_cooperators) : endowment_(endowment),
                                                                                                           cost_(cost),
                                                                                                           risk_(risk),
                                                                                                           min_nb_cooperators_(min_nb_cooperators) {
    nb_strategies_ = 2;

    // Payoffs cooperator and defector
    payoffs_success_[0] = endowment_;
    payoffs_failure_[0] = endowment_ * (1 - risk_);
    payoffs_success_[1] = endowment_ * (1 - cost_);
    payoffs_failure_[1] = endowment_ * (1 - risk_ - cost_);
}

double egttools::FinitePopulations::games::OneShotCRDNetworkGame::calculate_fitness(int strategy_index, egttools::VectorXui &state) {
    // We assume that index 0 is defect and index 1 is cooperate
    double payoff;
    int nb_cooperators = strategy_index + static_cast<int>(state(1));

    if (nb_cooperators < min_nb_cooperators_) {
        payoff = payoffs_failure_[strategy_index];
    } else {
        payoff = payoffs_success_[strategy_index];
    }

    return payoff;
}

int egttools::FinitePopulations::games::OneShotCRDNetworkGame::nb_strategies() const {
    return nb_strategies_;
}
double egttools::FinitePopulations::games::OneShotCRDNetworkGame::endowment() const {
    return endowment_;
}
double egttools::FinitePopulations::games::OneShotCRDNetworkGame::cost() const {
    return cost_;
}
double egttools::FinitePopulations::games::OneShotCRDNetworkGame::risk() const {
    return risk_;
}
int egttools::FinitePopulations::games::OneShotCRDNetworkGame::min_nb_cooperators() const {
    return min_nb_cooperators_;
}
std::string egttools::FinitePopulations::games::OneShotCRDNetworkGame::toString() const {
    std::stringstream ss;
    ss << "Python implementation of a public goods one-shot Collective Risk Dilemma on Networks." << std::endl;
    ss << "Game parameters" << std::endl;
    ss << "-------" << std::endl;
    ss << "b = " << endowment_ << std::endl;
    ss << "c = " << cost_ << std::endl;
    ss << "r = " << risk_ << std::endl;
    ss << "M = " << min_nb_cooperators_ << std::endl;
    ss << "Strategies" << std::endl;
    ss << "-------" << std::endl;
    ss << "Currently the only strategies are Cooperators and Defectors." << std::endl;

    return ss.str();
}
std::string egttools::FinitePopulations::games::OneShotCRDNetworkGame::type() const {
    return "egttools.games.OneShotCRDNetworkGame";
}

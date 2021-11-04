//
// Created by Elias Fernandez on 2019-07-02.
//

#include <Dyrwin/SED/MLS.hpp>

using namespace EGTTools;

SED::MLS<SED::GarciaGroup>::MLS(size_t generations,
                                size_t nb_strategies,
                                size_t group_size, size_t nb_groups) : _generations(generations),
                                                                       _nb_strategies(nb_strategies),
                                                                       _group_size(group_size),
                                                                       _nb_groups(nb_groups) {
    if (group_size < 4)
        throw std::invalid_argument(
                "The maximum group size must be at least 4.");
    _pop_size = _nb_groups * _group_size;

    // Initialize random uniform distribution
    _uint_rand = std::uniform_int_distribution<size_t>(0, _nb_groups - 1);
    _uint_rand_strategy = std::uniform_int_distribution<size_t>(0, _nb_strategies - 1);
    _real_rand = std::uniform_real_distribution<double>(0.0, 1.0);
}

double SED::MLS<SED::GarciaGroup>::fixationProbability(size_t invader, size_t resident, size_t runs,
                                                       double q, double lambda, double w, double alpha,
                                                       double kappa,
                                                       double z,
                                                       const Eigen::Ref<const Matrix2D> &payoff_matrix_in,
                                                       const Eigen::Ref<const Matrix2D> &payoff_matrix_out) {
    if (invader > _nb_strategies || resident > _nb_strategies)
        throw std::invalid_argument(
                "you must specify a valid index for invader and resident [0, " + std::to_string(_nb_strategies) +
                ")");
    if ((_nb_groups == 1) && q != 0.)
        throw std::invalid_argument(
                "The splitting probability must be zero when there is only 1 group in the population");
    if ((_nb_groups == 1) && alpha != 1.)
        throw std::invalid_argument(
                "The probability of ingroup interactions must be 1 when there is only one group in the population.");

    double r2m = 0; // resident to mutant count
    double r2r = 0; // resident to resident count
    VectorXui group_strategies = VectorXui::Zero(_nb_strategies);
    group_strategies(resident) = _group_size;

    Matrix2D payoff_in = payoff_matrix_in;
    Matrix2D payoff_out = payoff_matrix_out;

    // This loop can be done in parallel
#pragma omp parallel for shared(group_strategies) reduction(+:r2m, r2r)
    for (size_t i = 0; i < runs; ++i) {
        // First we initialize a homogeneous population with the resident strategy
        SED::GarciaGroup group(_nb_strategies, _group_size, w, group_strategies, payoff_in,
                               payoff_out);
        std::vector<SED::GarciaGroup> groups(_nb_groups, group);
        VectorXui strategies = VectorXui::Zero(_nb_strategies);
        strategies(resident) = _pop_size;

        // Then we create a mutant of the invading strategy
        _createMutant(invader, resident, groups);
        // Update full population frequencies
        _updateFullPopulationFrequencies(invader, resident, strategies);

        // Then we run the Moran Process
        for (size_t t = 0; t < _generations; ++t) {
            _update(q, lambda, alpha, kappa, z, groups, strategies);

            if (strategies(invader) == 0) {
                r2r += 1.0;
                break;
            } else if (strategies(resident) == 0) {
                r2m += 1.0;
                break;
            }
        } // end Moran process loop
    } // end runs loop

    if ((r2m == 0.0) && (r2r == 0.0)) return 0.0;
    else return r2m / (r2m + r2r);
}

void
SED::MLS<SED::GarciaGroup>::_update(double q, double lambda, double alpha, double kappa, double z,
                                    std::vector<SED::GarciaGroup> &groups,
                                    VectorXui &strategies) {
    _reproduce_garcia(groups, strategies, lambda, alpha);
    _resolve_conflict(kappa, z, groups, strategies);
    for (size_t i = 0; i < _nb_groups; ++i) {
        if (groups[i].isGroupOversize()) {
            if (_real_rand(_mt) < q) { // split group
                _splitGroup(i, groups, strategies);
            } else { // remove individual
                size_t deleted_strategy = groups[i].deleteMember(_mt);
                --strategies(deleted_strategy);
            }
        }
    }
}

void SED::MLS<SED::GarciaGroup>::_createMutant(size_t invader, size_t resident, std::vector<SED::GarciaGroup> &groups) {
    auto mutate_group = _uint_rand(_mt);
    groups[mutate_group].createMutant(invader, resident);
}

void SED::MLS<SED::GarciaGroup>::_createRandomMutant(size_t invader, std::vector<SED::GarciaGroup> &groups,
                                                     EGTTools::VectorXui &strategies) {
    auto mutate_group = _uint_rand(_mt);
    size_t mutating_strategy = groups[mutate_group].deleteMember(_mt);
    groups[mutate_group].addMember(invader);
    --strategies(mutating_strategy);
    ++strategies(invader);
}

void SED::MLS<SED::GarciaGroup>::_updateFullPopulationFrequencies(size_t increase, size_t decrease,
                                                                  EGTTools::VectorXui &strategies) {
    ++strategies(increase);
    --strategies(decrease);
}

void
SED::MLS<SED::GarciaGroup>::_reproduce_garcia(std::vector<SED::GarciaGroup> &groups, VectorXui &strategies,
                                              const double &lambda,
                                              const double &alpha) {
    auto parent_group = _payoffProportionalSelection(alpha, groups, strategies);
    auto[split, new_strategy] = groups[parent_group].createOffspring(_mt);
    ++strategies(new_strategy);
    if (_real_rand(_mt) < lambda) groups[_migrate(parent_group, new_strategy, groups)].totalPayoff(alpha, strategies);
    else groups[parent_group].totalPayoff(alpha, strategies); // update total fitness of the group
}

size_t
SED::MLS<SED::GarciaGroup>::_migrate(const size_t &parent_group, const size_t &migrating_strategy,
                                     std::vector<SED::GarciaGroup> &groups) {
    size_t child_group = _uint_rand(_mt);
    while (child_group == parent_group) child_group = _uint_rand(_mt);
    // First we delete the migrating strategy from the parent group
    groups[parent_group].deleteMember(migrating_strategy);
    // Then add the member to the randomly selected group
    groups[child_group].addMember(migrating_strategy);
    return child_group;
}

void SED::MLS<SED::GarciaGroup>::_mutate(std::vector<SED::GarciaGroup> &groups, VectorXui &strategies) {
    size_t parent_group, mutating_strategy, new_strategy;

    parent_group = _sizeProportionalSelection(groups);
    mutating_strategy = groups[parent_group].deleteMember(_mt);
    new_strategy = _uint_rand_strategy(_mt);
    while (mutating_strategy == new_strategy) new_strategy = _uint_rand_strategy(_mt);
    groups[parent_group].addMember(new_strategy);
    --strategies(mutating_strategy);
    ++strategies(new_strategy);
}

void
SED::MLS<SED::GarciaGroup>::_splitGroup(size_t parent_group, std::vector<SED::GarciaGroup> &groups,
                                        VectorXui &strategies) {
    // First choose a group to die
    size_t child_group = _uint_rand(_mt);
    while (child_group == parent_group) child_group = _uint_rand(_mt);
    // Now we split the group
    VectorXui &strategies_parent = groups[parent_group].strategies();
    VectorXui &strategies_child = groups[child_group].strategies();
    // Parent group size
    auto parent_group_size = groups[parent_group].group_size();

    // update strategies with the eliminated strategies from the child group
    strategies -= strategies_child;
    strategies_child.setZero();
    // vector of binomial distributions for each strategy (this will be used to select the members
    // that go to the child group
    std::binomial_distribution<size_t> binomial(_group_size, 0.5);
    size_t sum = 0;
    while ((sum < 2) || (sum > parent_group_size - 2) || sum > _group_size) {
        sum = 0;
        for (size_t i = 0; i < _nb_strategies; ++i) {
            if (strategies_parent(i) > 0) {
                binomial.param(std::binomial_distribution<size_t>::param_type(strategies_parent(i), 0.5));
                strategies_child(i) = binomial(_mt);
                sum += strategies_child(i);
            }
        }
    }
    // reset group size
    groups[child_group].set_group_size(sum);
    groups[parent_group].set_group_size(groups[parent_group].group_size() - sum);
    // reset parent group strategies
    strategies_parent -= strategies_child;
}

size_t
SED::MLS<SED::GarciaGroup>::_payoffProportionalSelection(const double &alpha, std::vector<SED::GarciaGroup> &groups,
                                                         VectorXui &strategies) {
    double total_fitness = 0.0, tmp = 0.0;
    // Calculate total fitness
    for (auto &group: groups) total_fitness += group.totalPayoff(alpha, strategies);
    total_fitness *= _real_rand(_mt);
    size_t parent_group = 0;
    for (parent_group = 0; parent_group < _nb_groups; ++parent_group) {
        tmp += groups[parent_group].group_fitness();
        if (tmp > total_fitness) return parent_group;
    }

    return 0;
}

size_t SED::MLS<SED::GarciaGroup>::_sizeProportionalSelection(std::vector<SED::GarciaGroup> &groups) {
    size_t pop_size = _current_pop_size(groups), tmp = 0;
    std::uniform_int_distribution<size_t> dist(0, pop_size - 1);
    // Calculate total fitness
    size_t p = dist(_mt);
    size_t parent_group = 0;
    for (parent_group = 0; parent_group < _nb_groups; ++parent_group) {
        tmp += groups[parent_group].group_size();
        if (tmp > p) return parent_group;
    }

    return 0;
}

bool SED::MLS<SED::GarciaGroup>::_pseudoStationary(std::vector<SED::GarciaGroup> &groups) {
    if (_current_pop_size(groups) < _pop_size) return false;
    for (auto &group: groups)
        if (!group.isPopulationMonomorphic())
            return false;

    return true;
}

void SED::MLS<SED::GarciaGroup>::_setFullHomogeneousState(size_t strategy, std::vector<SED::GarciaGroup> &groups) {
    for (auto &group: groups)
        group.setPopulationHomogeneous(strategy);
}

size_t SED::MLS<SED::GarciaGroup>::_current_pop_size(std::vector<SED::GarciaGroup> &groups) {
    size_t size = 0;
    for (auto &group: groups) size += group.group_size();

    return size;
}

void SED::MLS<SED::GarciaGroup>::_setState(std::vector<SED::GarciaGroup> &groups, std::vector<size_t> &container) {
    // Then we shuffle it randomly the contianer
    std::shuffle(container.begin(), container.end(), _mt);

    // Now we randomly initialize the groups with the population configuration from strategies
    for (size_t i = 0; i < _nb_groups; ++i) {
        groups[i].set_group_size(_group_size);
        VectorXui &group_strategies = groups[i].strategies();
        group_strategies.setZero();
        for (size_t j = 0; j < _group_size; ++j) {
            ++group_strategies(container[j + (i * _group_size)]);
        }
    }
}

void
SED::MLS<SED::GarciaGroup>::_resolve_conflict(const double &kappa, const double &z,
                                              std::vector<SED::GarciaGroup> &groups,
                                              VectorXui &strategies) {
    // Pairs of groups are selected for conflict. The value of @param kappa determines
    // the average fraction fo groups involved in conflict, in the following manner:
    //
    // a list of conflicting groups is constructed using a series of Bernoulli trials
    // with success probability kappa. After the nb_groups (m) trials the number of
    // selected groups may be odd. In this case, a random group is added to the list with
    // probability 0.5, or a random group from the list is taken out with probability 0.5.
    //
    // In a pair of conflicting groups, the one having the highest sum of payoffs has
    // a higher chance of winning (depending on z, if z = 0 a higher sum means winning
    // for sure).
    //
    // The winner is duplicated, and replaces the losing group. If the groups
    // selected for conflict have the same sum of payoffs one is chosen randomly to be
    // the winner with probability 0.5.
    double fitness_group1, fitness_group2, prob;

    std::vector<size_t> conflicts, no_conflicts;
    conflicts.reserve(_nb_groups);
    no_conflicts.reserve(_nb_groups);

    // Build conflict list
    for (size_t i = 0; i < _nb_groups; ++i) {
        if (_real_rand(_mt) < kappa) conflicts.push_back(i);
        else no_conflicts.push_back(i);
    }
    // If no conflicts return
    if (conflicts.empty()) return;
    // Update if odd number of groups
    if (conflicts.size() % 2 != 0) {
        if ((_real_rand(_mt) < 0.5) && (!no_conflicts.empty())) {
            std::uniform_int_distribution<size_t> dist(0, no_conflicts.size() - 1);
            conflicts.push_back(no_conflicts[dist(_mt)]);
        } else if (conflicts.size() > 1) {
            std::uniform_int_distribution<size_t> dist(0, conflicts.size() - 1);
            conflicts.erase(conflicts.begin() + dist(_mt));
        } else return;
    }

    // Resolve conflicts
    if (z > 0) {
        for (size_t i = 0; i < conflicts.size() - 1; i += 2) {
            prob = EGTTools::SED::contest_success(z, groups[conflicts[i]].group_fitness(),
                                                  groups[conflicts[i + 1]].group_fitness());

            if (_real_rand(_mt) < prob) {
                strategies.array() -= groups[conflicts[i + 1]].strategies().array();
                strategies.array() += groups[conflicts[i]].strategies().array();
                // Second group is replaced by the first
                groups[conflicts[i + 1]] = groups[conflicts[i]];
            } else {
                strategies.array() -= groups[conflicts[i]].strategies().array();
                strategies.array() += groups[conflicts[i + 1]].strategies().array();
                // Second group is replaced by the first
                groups[conflicts[i]] = groups[conflicts[i + 1]];
            }
        }
    } else {
        for (size_t i = 0; i < conflicts.size() - 1; i += 2) {
            fitness_group1 = groups[conflicts[i]].group_fitness();
            fitness_group2 = groups[conflicts[i + 1]].group_fitness();

            if (fitness_group1 > fitness_group2) {
                strategies.array() -= groups[conflicts[i + 1]].strategies().array();
                strategies.array() += groups[conflicts[i]].strategies().array();
                // Second group is replaced by the first
                groups[conflicts[i + 1]] = groups[conflicts[i]];
            } else if (fitness_group1 < fitness_group2) {
                strategies.array() -= groups[conflicts[i]].strategies().array();
                strategies.array() += groups[conflicts[i + 1]].strategies().array();
                // Second group is replaced by the first
                groups[conflicts[i]] = groups[conflicts[i + 1]];
            } else {
                // A random group wins
                if (_real_rand(_mt) < 0.5) {
                    strategies.array() -= groups[conflicts[i + 1]].strategies().array();
                    strategies.array() += groups[conflicts[i]].strategies().array();
                    // Second group is replaced by the first
                    groups[conflicts[i + 1]] = groups[conflicts[i]];
                } else {
                    strategies.array() -= groups[conflicts[i]].strategies().array();
                    strategies.array() += groups[conflicts[i + 1]].strategies().array();
                    // Second group is replaced by the first
                    groups[conflicts[i]] = groups[conflicts[i + 1]];
                }
            }
        }
    }
}
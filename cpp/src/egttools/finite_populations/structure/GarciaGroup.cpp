//
// Created by Elias Fernandez on 2019-04-25.
//

#include <egttools/finite_populations/structure/GarciaGroup.hpp>

using namespace egttools;

/**
 * @brief adds a mutant of an invading strategy and reduces one member of the resident strategy.
 *
 * @param invader : index of the invading strategy
 * @param resident : index of the resident strategy
 */
void FinitePopulations::GarciaGroup::createMutant(size_t invader, size_t resident) {
  ++_strategies(invader);
  --_strategies(resident);
}

double FinitePopulations::GarciaGroup::totalPayoff(const double &alpha, egttools::VectorXui &strategies) {
  double tmp1, tmp2;
  size_t out_pop_size = strategies.sum() - _group_size;
  assert (out_pop_size > 0);
  if (_group_size == 1) return (1.0 - _w);
  _group_fitness = 0.0;

  for (size_t i = 0; i < _nb_strategies; ++i) {
    if (_strategies(i) == 0) {
      _fitness(i) = 0;
      continue;
    }
    tmp1 = 0.0;
    tmp2 = 0.0;
    for (size_t j = 0; j < _nb_strategies; ++j) {
      if (j == i) {
        tmp1 += _payoff_matrix_in(i, i) * static_cast<double>(_strategies(i) - 1);
        tmp2 += _payoff_matrix_out(i, i) * (strategies(i) - _strategies(i));
      } else {
        tmp1 += _payoff_matrix_in(i, j) * _strategies(j);
        tmp2 += _payoff_matrix_out(i, j) * (strategies(j) - _strategies(j));
      }
    }
    tmp1 /= (_group_size - 1);
    tmp2 /= out_pop_size;
    _fitness(i) = alpha * tmp1 + (1.0 - alpha) * tmp2;
    _fitness(i) = ((1.0 - _w) + _w * _fitness(i)) * _strategies(i);
    assert (_fitness(i) >= 0);
    _group_fitness += _fitness(i);
  }

  return _group_fitness;
}

bool FinitePopulations::GarciaGroup::addMember(size_t new_strategy) {
  ++_strategies(new_strategy);
  return ++_group_size <= _max_group_size;
}

bool FinitePopulations::GarciaGroup::deleteMember(const size_t &member_strategy) {
  if (_strategies(member_strategy) <= 0) return false;
  --_strategies(member_strategy);
  --_group_size;
  return true;
}

/**
 * @brief Checks whether the population inside the group is monomorphic (only one strategy)
 *
 * @return true if monomorphic, otherwise false
 */
bool FinitePopulations::GarciaGroup::isPopulationMonomorphic() {
  for (size_t i = 0; i < _nb_strategies; ++i)
    if (_strategies(i) > 0 && _strategies(i) < _group_size)
      return false;
  return true;
}

/**
 * @brief makes the population in the group homonegous
 * @param strategy
 */
void FinitePopulations::GarciaGroup::setPopulationHomogeneous(size_t strategy) {
  _group_size = _max_group_size;
  _strategies.setZero();
  _strategies(strategy) = _max_group_size;
}

FinitePopulations::GarciaGroup::GarciaGroup(const FinitePopulations::GarciaGroup &grp)
    : _nb_strategies(grp.nb_strategies()),
      _max_group_size(grp.max_group_size()),
      _w(grp.selection_intensity()),
      _strategies(grp.strategies()),
      _payoff_matrix_in(grp.payoff_matrix_in()),
      _payoff_matrix_out(grp.payoff_matrix_out()) {
  _group_size = grp.group_size();
  _group_fitness = grp.group_fitness();
  _fitness = Vector::Zero(_nb_strategies);

  _urand = std::uniform_real_distribution<double>(0.0, 1.0);
  // the number of individuals in the group must be smaller or equal to the maximum capacity
  assert(_group_size <= _max_group_size);
}

FinitePopulations::GarciaGroup &FinitePopulations::GarciaGroup::operator=(const FinitePopulations::GarciaGroup &grp) {
  if (this == &grp) return *this;

  _nb_strategies = grp.nb_strategies();
  _max_group_size = grp.max_group_size();
  _group_size = grp.group_size();
  _group_fitness = grp.group_fitness();
  _w = grp.selection_intensity();
  _strategies.array() = grp.strategies();

  return *this;
}

bool FinitePopulations::GarciaGroup::isGroupOversize() {
  return _group_size > _max_group_size;
}

//
// Created by Elias Fernandez on 2019-04-25.
//

#ifndef DYRWIN_SED_STRUCTURE_GARCIAGROUP_HPP
#define DYRWIN_SED_STRUCTURE_GARCIAGROUP_HPP

#include <cmath>
#include <random>
#include <algorithm>
#include <Dyrwin/Types.h>
#include <iostream>

namespace EGTTools::SED {
class GarciaGroup {
 public:
  /**
   * @brief creates a new group that can undergo stochastic dynamics
   *
   * This class creates a group structure. Each of the groups is subject to an internal selection process.
   * Using this population structure, there is a selection process at an individual level, but also another
   * at group level. Certain groups may invade others. At each time step a player is randomly selected
   * proportional to its payoff. In this implementation, first we select the group with a higher aggregated
   * payoff, and then we select a player inside that group proportional to its payoff.
   *
   * @param nb_strategies : number of possible strategies in the population
   * @param max_group_size : maximum capacity (size) of the group
   * @param w : intensity of selection
   * @param init_strategies : number of individuals of each strategy in the group
   * @param payoff_matrix : reference to the payoff matrix
   */
  GarciaGroup(size_t nb_strategies, size_t max_group_size, double w, const VectorXui &init_strategies,
              const Matrix2D &payoff_matrix_in, const Matrix2D &payoff_matrix_out) : _nb_strategies(nb_strategies),
                                                                                     _max_group_size(max_group_size),
                                                                                     _w(w),
                                                                                     _strategies(init_strategies),
                                                                                     _payoff_matrix_in(payoff_matrix_in),
                                                                                     _payoff_matrix_out(
                                                                                         payoff_matrix_out) {
    if (payoff_matrix_in.rows() != payoff_matrix_in.cols())
      throw std::invalid_argument("Payoff matrix must be a square Matrix (n,n)");
    if (static_cast<size_t>(payoff_matrix_in.rows()) != nb_strategies)
      throw std::invalid_argument(
          "Payoff matrix must have the same number of rows and columns as strategies");
    if (static_cast<size_t>(init_strategies.size()) != nb_strategies)
      throw std::invalid_argument("size of init strategies must be equal to the number of strategies");

    // Initialize the number of individuals of each strategy
    // we take ownership of the init_strategies vector
//            _strategies = VectorXui(_nb_strategies);
//            _strategies << init_strategies;
    _group_size = _strategies.sum();
    _fitness = Vector::Zero(_nb_strategies);
    _group_fitness = 0.0;
    _urand = std::uniform_real_distribution<double>(0.0, 1.0);
    // the number of individuals in the group must be smaller or equal to the maximum capacity
    assert(_group_size <= _max_group_size);
  }

  /**
   * Overloads copy constructor
   * @param grp
   * @return
   */
  GarciaGroup(const GarciaGroup &grp);

  /**
   * @brief We overload the assignment operator so that we can copy a group into another.
   *
   * @param grp group to be copied
   * @return a group reference
   */
  GarciaGroup &operator=(const GarciaGroup &grp);

  // Destructor and move are the default ones
  GarciaGroup(GarciaGroup &&grp) = delete;
  GarciaGroup &operator=(GarciaGroup &&grp) = delete;
  ~GarciaGroup() = default;

  template<typename G = std::mt19937_64>
  std::pair<bool, size_t> createOffspring(G &generator);

  void createMutant(size_t invader, size_t resident);

  /**
   * @brief calculates the total fitness of the group and updates the fitness of each individual
   *
   * @param strategies : frequency of the strategies of the external population
   * @return
   */
  double totalPayoff(const double &alpha, VectorXui &strategies);

  bool addMember(size_t new_strategy); // adds a new member to the group

  template<typename G = std::mt19937_64>
  size_t deleteMember(G &generator);    // delete one randomly chosen member

  /**
   * @brief Deletes a member with strategy @param member_strategy
   * @param member_strategy : strategy to be deleted
   * @return a boolean indicating wether the member was deleted
   */
  bool deleteMember(const size_t &member_strategy);    // delete one randomly chosen member

  template<typename G = std::mt19937_64>
  inline size_t payoffProportionalSelection(G &generator);

  bool isPopulationMonomorphic();

  void setPopulationHomogeneous(size_t strategy);

  /**
   * @brief checks if the size of the group is above the maximum allowed size
   * @return true if oversize else false
   */
  bool isGroupOversize();

  // Getters
  [[nodiscard]] size_t nb_strategies() const { return _nb_strategies; }

  [[nodiscard]] size_t max_group_size() const { return _max_group_size; }

  [[nodiscard]] size_t group_size() const { return _group_size; }

  [[nodiscard]] double group_fitness() const { return _group_fitness; }

  [[nodiscard]] double selection_intensity() const { return _w; }

  VectorXui &strategies() { return _strategies; }

  [[nodiscard]] const VectorXui &strategies() const { return _strategies; }

  [[nodiscard]] const Matrix2D &payoff_matrix_in() const { return _payoff_matrix_in; }
  [[nodiscard]] const Matrix2D &payoff_matrix_out() const { return _payoff_matrix_out; }

  // Setters
  void set_group_size(size_t group_size) { _group_size = group_size; }

  void set_max_group_size(size_t max_group_size) { _max_group_size = max_group_size; }

  void set_selection_intensity(double w) { _w = w; }

  void set_strategy_count(const Eigen::Ref<const VectorXui> &strategies) {
    if (strategies.sum() <= _max_group_size)
      throw std::invalid_argument("The sum of all individuals must not be bigger than the maximum group size!");
    _strategies.array() = strategies;
  }

 private:
  // maximum group size (n) and current group size
  size_t _nb_strategies, _max_group_size, _group_size;
  double _group_fitness;                           // group fitness
  double _w;                                      // intensity of selection
  VectorXui _strategies;                         // vector containing the number of individuals of each strategy
  Vector _fitness;                               // container for the fitness of each strategy
  const Matrix2D &_payoff_matrix_in, &_payoff_matrix_out;  // reference to a payoff matrix
  std::uniform_real_distribution<double> _urand; // uniform random distribution
};

/**
 * @brief Adds a new member of a given strategy to the group (proportional to the fitness).
 *
 * @tparam G : random generator container class
 * @param generator : random generator
 * @return true if group_size <= max_group_size, else false
 */
template<typename G>
std::pair<bool, size_t> GarciaGroup::createOffspring(G &generator) {
  auto new_strategy = payoffProportionalSelection<G>(generator);
  ++_strategies(new_strategy);
  return std::make_pair(++_group_size > _max_group_size, new_strategy);
}

/**
 * @brief deletes a random member from the group
 *
 * @tparam G : random generator container class
 * @param generator : random generator
 * @return index to the deleted member
 */
template<typename G>
size_t GarciaGroup::deleteMember(G &generator) {
  size_t selected_strategy, sum = 0;
  // choose random member for deletion
  std::uniform_int_distribution<size_t> dist(0, _group_size - 1);
  size_t die = dist(generator);
  for (selected_strategy = 0; selected_strategy < _nb_strategies; ++selected_strategy) {
    if (_strategies(selected_strategy) == 0) continue;
    sum += _strategies(selected_strategy);
    if (die < sum) break;
  }

  --_strategies(selected_strategy);
  --_group_size;
  return selected_strategy;
}

/**
 * @brief selects an individual from a strategy proportionally to the payoff
 *
 * @tparam G : random generator container class
 * @param generator : random generator
 * @return : index of the strategy selected
 */
template<typename G>
size_t GarciaGroup::payoffProportionalSelection(G &generator) {
  double sum = 0.0;
  auto p = _urand(generator) * _group_fitness;
  for (size_t i = 0; i < _nb_strategies; ++i) {
    sum += _fitness(i);
    if (p < sum) return i;
  }
  // It should never get here
  assert(p < sum);
  return 0;
}
}

#endif //DYRWIN_SED_STRUCTURE_GROUP_HPP

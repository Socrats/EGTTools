//
// Created by Elias Fernandez on 2019-04-25.
//

#ifndef DYRWIN_SED_MLS_HPP
#define DYRWIN_SED_MLS_HPP

#include <Dyrwin/SeedGenerator.h>
#include <Dyrwin/Types.h>

#include <Dyrwin/OpenMPUtils.hpp>
#include <Dyrwin/SED/Utils.hpp>
#include <Dyrwin/SED/structure/GarciaGroup.hpp>
#include <Dyrwin/SED/structure/Group.hpp>
#include <algorithm>
#include <cmath>
#include <iterator>
#include <random>
#include <stdexcept>
#include <unordered_set>

namespace EGTTools::SED {
template<typename S = Group>
class MLS {
 public:
  /**
   * @brief This class implements the Multi Level selection process introduced in Arne et al.
   *
   * This class implements selection on the level of groups.
   *
   * A population with m groups, which all have a maximum size n. Therefore, the maximum population
   * size N = nm. Each group must contain at least one individual. The minimum population size is m
   * (each group must have at least one individual). In each time step, an individual is chosen from
   * a population with a probability proportional to its fitness. The individual produces an
   * identical offspring that is added to the same group. If the group size is greater than n after
   * this step, then either a randomly chosen individual from the group is eliminated (with probability 1-q)
   * or the group splits into two groups (with probability q). Each individual of the splitting
   * group has probability 1/2 to end up in each of the daughter groups. One daughter group remains
   * empty with probability 2^(1-n). In this case, the repeating process is repeated to avoid empty
   * groups. In order to keep the number of groups constant, a randomly chosen group is eliminated
   * whenever a group splits in two.
   *
   * @param generations : maximum number of generations
   * @param nb_strategies : number of strategies in the population
   * @param group_size : group size (n)
   * @param nb_groups : number of groups (m)
   * @param w : intensity of selection
   * @param strategy_freq : frequency of each strategy in the population
   * @param payoff_matrix : payoff matrix
   */
  MLS(size_t generations, size_t nb_strategies, size_t group_size, size_t nb_groups, double w,
      const Eigen::Ref<const Vector> &strategies_freq, const Eigen::Ref<const Matrix2D> &payoff_matrix);

  /**
  * @brief Runs one complete simulation with multi-level selection.
  *
  * Runs the moran process with multi-level selection for a given number of generations
  * or until it reaches a monomorphic state.
  *
  * @tparam S : container for a group
  * @param q : splitting probability
  * @param w : intensity of selection
  * @param init_state : vector with the initial state of the population
  * @return a vector with the final state of the population
  */
  Vector evolve(double q, double w, const Eigen::Ref<const VectorXui> &init_state);

  /**
   * @brief Runs one complete simulation with multi-level selection
   *
   * This method includes the possibility that players migrate between groups.
   *
   * @param q : splitting probability
   * @param w : intensity of selection
   * @param lambda : migration probability
   * @param init_state : intitial state of the population
   * @return final state of the population
   */
  Vector evolve(double q, double w, double lambda, const Eigen::Ref<const VectorXui> &init_state);

  /**
   * @brief Runs one complete simulation with multi-level selection following (Garcia et al. 2011)
   *
   * This Methods follows the multi-level selection with direct group interactions presented
   * in (Garcia et al. 2011). @param kappa defines the fraction of groups that will interact
   * against each other.
   *
   * @param q : splitting probability
   * @param w : intensity of selection
   * @param lambda : migration probability
   * @param kappa : average fraction of groups involved in a conflict
   * @param z: intesity of selection between groups in conflict
   * @param init_state : initial state of the population
   * @return the final state of the population
   */
  Vector evolve(double q, double w, double lambda, double kappa, double z,
                const Eigen::Ref<const VectorXui> &init_state);

//
//        Vector evolve(size_t runs, double w);

  /**
   * @brief estimates the fixation probability of the invading strategy over the resident strategy.
   *
   * This function will estimate numerically (by running simulations) the fixation probability of
   * a certain strategy in the population of 1 resident strategy.
   *
   * This implementation specializes on the EGTTools::SED::Group class
   *
   * @tparam S : container for the structure of the population
   * @param invader : index of the invading strategy
   * @param resident : index of the resident strategy
   * @param runs : number of runs (used to average the number of times the invading strategy has fixated)
   * @param q : splitting probability
   * @param w : intensity of selection
   * @return a real number (double) indicating the fixation probability
   */
  double fixationProbability(size_t invader, size_t resident, size_t runs,
                             double q, double w);

  /**
  * @brief estimates the fixation probability of the invading strategy over the resident strategy.
  *
  * This function will estimate numerically (by running simulations) the fixation probability of
  * a certain strategy in the population of 1 resident strategy.
  *
  * This implementation specializes on the EGTTools::SED::Group class
  *
  * @tparam S : container for the structure of the population
  * @param invader : index of the invading strategy
  * @param resident : index of the resident strategy
  * @param runs : number of runs (used to average the number of times the invading strategy has fixated)
  * @param q : splitting probability
  * @param lambda : migration probability
  * @param w : intensity of selection
  * @return a real number (double) indicating the fixation probability
  */
  double fixationProbability(size_t invader, size_t resident, size_t runs,
                             double q, double lambda, double w);

  /**
  * @brief estimates the fixation probability of the invading strategy over the resident strategy.
  *
  * This function will estimate numerically (by running simulations) the fixation probability of
  * a certain strategy in the population of 1 resident strategy.
  *
  * The evolutionary process uses (Garcia et al.) multi-level selection with direct conflict
  * between groups.
  *
  * This implementation specializes on the EGTTools::SED::Group class
  *
  * @tparam S : container for the structure of the population
  * @param invader : index of the invading strategy
  * @param resident : index of the resident strategy
  * @param runs : number of runs (used to average the number of times the invading strategy has fixated)
  * @param q : splitting probability
  * @param lambda : migration probability
  * @param w : intensity of selection
  * @param kappa : fraction of groups that enter in conflict
  * @param z : importance of payoffs during conflict
  * @return a real number (double) indicating the fixation probability
  */
  double fixationProbability(size_t invader, size_t resident, size_t runs,
                             double q, double lambda, double w, double kappa, double z);

  /**
   * @brief estimates the fixation probability of the invading strategy over the resident strategy.
   *
   * This function will estimate numerically (by running simulations) the fixation probability of
   * a certain strategy in the population of 1 resident strategy.
   *
   * This implementation specializes on the EGTTools::SED::Group class
   * @tparam S : container for the structure of the population (group)
   * @param invader : index of the invading strategy
   * @param init_state : vector containing the initial state of the population (number of individuals of each strategy)
   * @param runs : number of runs (used to average the number of times the invading strategy has fixated)
   * @param q : splitting probability
   * @param w : intensity of selection
   * @return a vector of doubles indicating the probability that each strategy fixates from the initial state
   */
  Vector fixationProbability(size_t invader, const Eigen::Ref<const VectorXui> &init_state, size_t runs,
                             double q, double w);

  /**
  * @brief calculates the gradient of selection between 2 strategies.
  *
  * Will return the difference between T+ and T- for each possible population configuration
  * when the is conformed only by the resident and the invading strategy.
  *
  * To estimate T+ - T- (the probability that the number of invaders increase/decrease in the population)
  * we run the simulation for population with k invaders and Z - k residents for @param run
  * times and average how many times did the number of invadors increase and decrease.
  *
  * @tparam S : group container
  * @param invader : index of the invading strategy
  * @param resident : index of the resident strategy
  * @param runs : number of runs (to average the results)
  * @param w : intensity of selection
  * @param q : splitting probability
  * @return : an Eigen vector with the gradient of selection for each k/Z where k is the number of invaders.
  */
  Vector
  gradientOfSelection(size_t invader, size_t resident, size_t runs, double w, double q = 0.0);

  /**
  * @brief calculates the gradient of selection for an invading strategy and any initial state.
  *
  * Will return the difference between T+ and T- for each possible population configuration
  * when the is conformed only by the resident and the invading strategy.
  *
  * To estimate T+ - T- (the probability that the number of invaders increase/decrease in the population)
  * we run the simulation for population with k invaders and Z - k residents for @param run
  * times and average how many times did the number of invadors increase and decrease.
  *
  * @tparam S : group container
  * @param invader : index of the invading strategy
  * @param resident : index of the resident strategy
  * @param init_state : vector indicating the initial state of the population (how many individuals of each strategy)
  * @param runs : number of runs (to average the results)
  * @param w : intensity of selection
  * @param q : splitting probability
  * @return : an Eigen vector with the gradient of selection for each k/Z where k is the number of invaders.
  */
  Vector
  gradientOfSelection(size_t invader, size_t resident, const Eigen::Ref<const VectorXui> &init_state, size_t runs,
                      double w, double q = 0.0);

  // To avoid memory explosion, we limit the call to this function for a maximum of 3 strategies
//        SparseMatrix2D transitionMatrix(size_t runs, size_t t0, double q, double lambda, double w);
//
//        SparseMatrix2D
//        transitionMatrix(size_t invader, size_t resident, size_t runs, size_t t0, double q, double lambda, double w);


  // Getters
  [[nodiscard]] size_t generations() const { return _generations; }

  [[nodiscard]] size_t nb_strategies() const { return _nb_strategies; }

  [[nodiscard]] size_t max_pop_size() const { return _pop_size; }

  [[nodiscard]] size_t group_size() const { return _group_size; }

  [[nodiscard]] size_t nb_groups() const { return _nb_groups; }

  [[nodiscard]] [[maybe_unused]] double selection_intensity() const { return _w; }

  [[nodiscard]] const Vector init_strategy_freq() const { return _strategies.cast<double>() / _pop_size; }

  [[nodiscard]] const Vector &strategy_freq() const { return _strategy_freq; }

  [[nodiscard]] const VectorXui &init_strategy_count() const { return _strategies; }

  [[nodiscard]] const Matrix2D &payoff_matrix() const { return _payoff_matrix; }

  // Setters
  void set_generations(size_t generations) { _generations = generations; }

  void set_pop_size(size_t pop_size) { _pop_size = pop_size; }

  void set_group_size(size_t group_size) {
    if (group_size < 4)
      throw std::invalid_argument(
          "The maximum group size must be at least 4.");
    _group_size = group_size;
    _pop_size = _nb_groups * _group_size;
  }

  void set_nb_groups(size_t nb_groups) {
    if (nb_groups == 0)
      throw std::invalid_argument(
          "There must be at least 1 group in the population");
    _nb_groups = nb_groups;
    _uint_rand.param(std::uniform_int_distribution<size_t>::param_type(0, _nb_groups - 1));
    _pop_size = _nb_groups * _group_size;
  }

  void set_selection_intensity(double w) { _w = w; }

  void set_strategy_freq(const Eigen::Ref<const Vector> &strategy_freq) {
    if (strategy_freq.sum() != 1.0) throw std::invalid_argument("Frequencies must sum to 1");
    _strategy_freq.array() = strategy_freq;
    // Recompute strategies
    size_t tmp = 0;
    for (size_t i = 0; i < (_nb_strategies - 1); ++i) {
      _strategies(i) = (size_t) floor(_strategy_freq(i) * _pop_size);
      tmp += _strategies(i);
    }
    _strategies(_nb_strategies - 1) = _pop_size - tmp;
  }

  void set_strategy_count(const Eigen::Ref<const VectorXui> &strategies) {
    if (strategies.sum() != _pop_size)
      throw std::invalid_argument("The sum of all individuals must be equal to the population size!");
    _strategies.array() = strategies;
    // Recompute strategy frequencies
    _strategy_freq.array() = _strategies.cast<double>() / _pop_size;
  }

  void set_payoff_matrix(const Eigen::Ref<const Matrix2D> &payoff_matrix) {
    if (payoff_matrix.rows() != payoff_matrix.cols())
      throw std::invalid_argument("Payoff matrix must be a square Matrix (n,n)");
    _nb_strategies = payoff_matrix.rows();
    _uint_rand_strategy.param(std::uniform_int_distribution<size_t>::param_type(0, _nb_strategies - 1));
    _payoff_matrix.array() = payoff_matrix;
  }

  [[nodiscard]] std::string toString() const {
    Eigen::IOFormat CleanFmt(4, 0, ", ", "\n", "[", "]");
    std::stringstream ss;
    ss << _payoff_matrix.format(CleanFmt);
    return "Z = " + std::to_string(_pop_size) +
        "\nm = " + std::to_string(_nb_groups) +
        "\nn = " + std::to_string(_group_size) +
        "\nnb_strategies = " + std::to_string(_nb_strategies) +
        "\npayoff_matrix = " + ss.str();
  }

  friend std::ostream &operator<<(std::ostream &o, MLS &r) { return o << r.toString(); }

 private:
  size_t _generations, _nb_strategies, _group_size, _nb_groups, _pop_size;
  double _w;

  Vector _strategy_freq; // frequency of each strategy in the population
  VectorXui _strategies; //nb of players of each strategy
  Matrix2D _payoff_matrix; // stores the payoff matrix of the game

  // Uniform random distribution
  std::uniform_int_distribution<size_t> _uint_rand;
  std::uniform_int_distribution<size_t> _uint_rand_strategy;
  std::uniform_real_distribution<double> _real_rand; // uniform random distribution

  // Random generators
  std::mt19937_64 _mt{EGTTools::Random::SeedGenerator::getInstance().getSeed()};

  inline void _update(double q, std::vector<S> &groups, VectorXui &strategies);

  inline void _update(double q, double lambda, std::vector<S> &groups, VectorXui &strategies);

  inline void _update(double q, double lambda, double mu, std::vector<S> &groups, VectorXui &strategies);

  /**
   * @brief Updates the population one step with migration, splitting and group conflict
   * @param q
   * @param lambda
   * @param kappa
   * @param z
   * @param groups
   * @param strategies
   */
  inline void
  _update(double q, double lambda, double kappa, double z, std::vector<S> &groups, VectorXui &strategies);

  /**
   * @brief Updates the population one step with migration, splitting, group conflict and inter-group interactions
   * @param q
   * @param lambda
   * @param alpha
   * @param kappa
   * @param z
   * @param groups
   * @param strategies
   */
  inline void
  _update(double q, double lambda, double alpha, double kappa, double z, std::vector<S> &groups,
          VectorXui &strategies);

  inline void _speedUpdate(double q, std::vector<S> &groups, VectorXui &strategies);

  inline void _speedUpdate(double q, double lambda, std::vector<S> &groups, VectorXui &strategies);

  inline void _speedUpdate(double q, double lambda, double mu, std::vector<S> &groups, VectorXui &strategies);

  inline void _createMutant(size_t invader, size_t resident, std::vector<S> &groups);

  /**
   * @brief Adds a mutant of strategy invader to the population
   *
   * Eliminates a randmo strategy from the population and adds a mutant of strategy invader.
   *
   * @tparam S : container for population structur
   * @param invader : index of the invader strategy
   * @param groups : vector of groups
   * @param strategies : vector of strategies
   */
  inline void _createRandomMutant(size_t invader, std::vector<S> &groups, VectorXui &strategies);

  inline void _updateFullPopulationFrequencies(size_t increase, size_t decrease, VectorXui &strategies);

  /**
  * @brief internal reproduction function.
  *
  * This function always splits the group
  *
  * @tparam S : group container
  * @param groups : vector of groups
  * @param strategies : vector of the current proportions of each strategy in the population
  */
  void _reproduce(std::vector<S> &groups, VectorXui &strategies);

  /**
  * @brief internal reproduction function.
  *
  * This functions will split depending a splitting probability q.
  *
  * @tparam S : group container
  * @param groups : vector of groups
  * @param strategies : vector of the current proportions of each strategy in the population
  * @param q : split probability
  */
  void _reproduce(std::vector<S> &groups, VectorXui &strategies, double q);

  /**
  * @brief internal reproduction function.
  *
  * This functions will split depending a splitting probability q.
  *
  * @tparam S : group container
  * @param groups : vector of groups
  * @param strategies : vector of the current proportions of each strategy in the population
  * @param lambda : migration probability
  */
  inline void _reproduce_garcia(std::vector<S> &groups, VectorXui &strategies, const double &lambda);

  /**
  * @brief internal reproduction function.
  *
  * This functions will split depending a splitting probability q.
  *
  * @tparam S : group container
  * @param groups : vector of groups
  * @param strategies : vector of the current proportions of each strategy in the population
  * @param lambda : migration probability
  * @brief alpha : probability of interacting with members of the same group
  */
  inline void
  _reproduce_garcia(std::vector<S> &groups, VectorXui &strategies, const double &lambda, const double &alpha);

  /**
  * @brief Migrates an individual from a group to another
  *
  * @tparam S : group container
  * @param q : splitting probability
  * @param groups : reference to a vector of groups
  */
  void _migrate(double q, std::vector<S> &groups, VectorXui &strategies);

  /**
   * @brief Migrates an individual from a group to another
   *
   * @param parent_group : group of the invididual that will migrate
   * @param individual : strategy of the migrating individual
   * @param q : splitting probability
   * @param groups : vector of groups
   */
  inline void
  _migrate(const size_t &parent_group, const size_t &migrating_strategy, std::vector<S> &groups);

  /**
  * @brief Mutates an individual from the population
  *
  * @tparam S : group container
  * @param mu
  * @param groups : reference to a vector of groups
  */
  void _mutate(std::vector<S> &groups, VectorXui &strategies);

  /**
   * @brief splits a group in two
   *
   * This method creates a new group. There is a 0.5 probability that each
   * member of the former group will be part of the new group. Also, since
   * the number of groups is kept constant, a random group is chosen to die.
   *
   * @tparam S : group container
   * @param parent_group : index to the group to split
   * @param groups : reference to a vector of groups
   */
  void _splitGroup(size_t parent_group, std::vector<S> &groups, VectorXui &strategies);

  /**
  * @brief selects a group proportional to its total payoff.
  *
  * @tparam S : group container
  * @param groups : reference to the population groups
  * @return : index of the parent group
  */
  size_t _payoffProportionalSelection(std::vector<S> &groups);

  /**
   * @brief selects a group proportional to its total payoff with within group interactions.
   *
   * @param alpha : probability of interacting with members of the same group
   * @param groups : vector of groups
   * @param strategies : vector of strategy counts
   * @return : index of the selected individual
   */
  size_t _payoffProportionalSelection(const double &alpha, std::vector<S> &groups, VectorXui &strategies);

  /**
  * @brief selects a group proportional to its size.
  *
  * @tparam S : group container
  * @param groups : reference to the population groups
  * @return : index of the parent group
  */
  size_t _sizeProportionalSelection(std::vector<S> &groups);

  /**
  * @brief Checks whether a pseudo stationary state has been reached.
  *
  * @tparam S : group container
  * @param groups : reference to a vector of groups
  * @return true if reached a pseudo stationary state, otherwise false
  */
  bool _pseudoStationary(std::vector<S> &groups);

  /**
   * @brief sets the all individuals of one strategy
   *
   * Sets all individuals in the population of one strategy and sets all groups at maximum capacity.
   *
   * @tparam S : group container
   * @param strategy : resident strategy
   * @param groups : reference to a vector of groups
   */
  void _setFullHomogeneousState(size_t strategy, std::vector<S> &groups);

  /**
   * @brief Sets randomly the state of the population given a vector which contains the strategies in the population.
   *
   * This method shuffles a vector containing the population of strategies and then assigns each _group_size
   * of strategies to a group.
   *
   * @tparam S : container for the groups
   * @param groups : vector of groups
   * @param container : vector of strategies
   */
  inline void _setState(std::vector<S> &groups, std::vector<size_t> &container);

  /**
   * @brief returns the total population size
   * @tparam S : group container
   * @param groups : reference to a vector of groups
   * @return the sum of the sizes of all the groups
   */
  inline size_t _current_pop_size(std::vector<S> &groups);

  /**
   * @brief Performs the conflict resultion step of Garcia et al.
   *
   * @param kappa : average fraction of groups involved in conflict
   * @param z : importance of payoffs
   * @param groups : vector of groups
   * @param strategies : vector of strategies
   */
  inline void
  _resolve_conflict(const double &kappa, const double &z, std::vector<S> &groups, VectorXui &strategies);

};

template<>
class MLS<GarciaGroup> {
 public:
  /**
   * @brief This class implements the Multi Level selection process introduced in Arne et al.
   *
   * This class implements selection on the level of groups.
   *
   * A population with m groups, which all have a maximum size n. Therefore, the maximum population
   * size N = nm. Each group must contain at least one individual. The minimum population size is m
   * (each group must have at least one individual). In each time step, an individual is chosen from
   * a population with a probability proportional to its fitness. The individual produces an
   * identical offspring that is added to the same group. If the group size is greater than n after
   * this step, then either a randomly chosen individual from the group is eliminated (with probability 1-q)
   * or the group splits into two groups (with probability q). Each individual of the splitting
   * group has probability 1/2 to end up in each of the daughter groups. One daughter group remains
   * empty with probability 2^(1-n). In this case, the repeating process is repeated to avoid empty
   * groups. In order to keep the number of groups constant, a randomly chosen group is eliminated
   * whenever a group splits in two.
   *
   * @param generations : maximum number of generations
   * @param nb_strategies : number of strategies in the population
   * @param group_size : group size (n)
   * @param nb_groups : number of groups (m)
   */
  MLS(size_t generations, size_t nb_strategies, size_t group_size, size_t nb_groups);

  /**
  * @brief estimates the fixation probability of the invading strategy over the resident strategy.
  *
  * This function will estimate numerically (by running simulations) the fixation probability of
  * a certain strategy in the population of 1 resident strategy.
  *
  * The evolutionary process uses (Garcia et al.) multi-level selection with direct conflict
  * between groups and inter-group interactions.
  *
  * This function should only be called with GarciaGroup
  *
  * This implementation specializes on the EGTTools::SED::Group class
  *
  * @tparam S : container for the structure of the population
  * @param invader : index of the invading strategy
  * @param resident : index of the resident strategy
  * @param runs : number of runs (used to average the number of times the invading strategy has fixated)
  * @param q : splitting probability
  * @param lambda : migration probability
  * @param w : intensity of selection
  * @param alpha : probability of interacter with members of the same group
  * @param kappa : fraction of groups that enter in conflict
  * @param z : importance of payoffs during conflict
  * @param payoff_matrix_in : ingroup payoff matrix
  * @param payoff_matrix_out : outgroup payoff matrix
  * @return a real number (double) indicating the fixation probability
  */
  double fixationProbability(size_t invader, size_t resident, size_t runs,
                             double q, double lambda, double w, double alpha, double kappa, double z,
                             const Eigen::Ref<const Matrix2D> &payoff_matrix_in,
                             const Eigen::Ref<const Matrix2D> &payoff_matrix_out);

  // Getters
  [[nodiscard]] size_t generations() const { return _generations; }

  [[nodiscard]] size_t nb_strategies() const { return _nb_strategies; }

  [[nodiscard]] size_t max_pop_size() const { return _pop_size; }

  [[nodiscard]] size_t group_size() const { return _group_size; }

  [[nodiscard]] size_t nb_groups() const { return _nb_groups; }

  // Setters
  void set_generations(size_t generations) { _generations = generations; }

  void set_pop_size(size_t pop_size) { _pop_size = pop_size; }

  void set_group_size(size_t group_size) {
    if (group_size < 4)
      throw std::invalid_argument(
          "The maximum group size must be at least 4.");
    _group_size = group_size;
    _pop_size = _nb_groups * _group_size;
  }

  void set_nb_groups(size_t nb_groups) {
    if (nb_groups == 0)
      throw std::invalid_argument(
          "There must be at least 1 group in the population");
    _nb_groups = nb_groups;
    _uint_rand.param(std::uniform_int_distribution<size_t>::param_type(0, _nb_groups - 1));
    _pop_size = _nb_groups * _group_size;
  }

  [[nodiscard]] std::string toString() const {
    Eigen::IOFormat CleanFmt(4, 0, ", ", "\n", "[", "]");
    std::stringstream ss;
    return "Z = " + std::to_string(_pop_size) +
        "\nm = " + std::to_string(_nb_groups) +
        "\nn = " + std::to_string(_group_size) +
        "\nnb_strategies = " + std::to_string(_nb_strategies) +
        "\npayoff_matrix = " + ss.str();
  }

  friend std::ostream &operator<<(std::ostream &o, MLS &r) { return o << r.toString(); }

 private:
  size_t _generations, _nb_strategies, _group_size, _nb_groups, _pop_size;

  // Uniform random distribution
  std::uniform_int_distribution<size_t> _uint_rand;
  std::uniform_int_distribution<size_t> _uint_rand_strategy;
  std::uniform_real_distribution<double> _real_rand; // uniform random distribution

  // Random generators
  std::mt19937_64 _mt{EGTTools::Random::SeedGenerator::getInstance().getSeed()};

  /**
   * @brief Updates the population one step with migration, splitting, group conflict and inter-group interactions
   * @param q
   * @param lambda
   * @param alpha
   * @param kappa
   * @param z
   * @param groups
   * @param strategies
   */
  inline void
  _update(double q, double lambda, double alpha, double kappa, double z, std::vector<GarciaGroup> &groups,
          VectorXui &strategies);

  inline void _createMutant(size_t invader, size_t resident, std::vector<GarciaGroup> &groups);

  /**
   * @brief Adds a mutant of strategy invader to the population
   *
   * Eliminates a randmo strategy from the population and adds a mutant of strategy invader.
   *
   * @tparam S : container for population structur
   * @param invader : index of the invader strategy
   * @param groups : vector of groups
   * @param strategies : vector of strategies
   */
  inline void _createRandomMutant(size_t invader, std::vector<GarciaGroup> &groups, VectorXui &strategies);

  inline void _updateFullPopulationFrequencies(size_t increase, size_t decrease, VectorXui &strategies);

  /**
  * @brief internal reproduction function.
  *
  * This functions will split depending a splitting probability q.
  *
  * @tparam S : group container
  * @param groups : vector of groups
  * @param strategies : vector of the current proportions of each strategy in the population
  * @param lambda : migration probability
  * @brief alpha : probability of interacting with members of the same group
  */
  inline void
  _reproduce_garcia(std::vector<GarciaGroup> &groups, VectorXui &strategies, const double &lambda,
                    const double &alpha);

  /**
   * @brief Migrates an individual from a group to another
   *
   * @param parent_group : group of the individual that will migrate
   * @param individual : strategy of the migrating individual
   * @param q : splitting probability
   * @param groups : vector of groups
   * @return index of the child group
   */
  inline size_t
  _migrate(const size_t &parent_group, const size_t &migrating_strategy, std::vector<GarciaGroup> &groups);

  /**
  * @brief Mutates an individual from the population
  *
  * @tparam S : group container
  * @param mu
  * @param groups : reference to a vector of groups
  */
  void _mutate(std::vector<GarciaGroup> &groups, VectorXui &strategies);

  /**
   * @brief splits a group in two
   *
   * This method creates a new group. There is a 0.5 probability that each
   * member of the former group will be part of the new group. Also, since
   * the number of groups is kept constant, a random group is chosen to die.
   *
   * @tparam S : group container
   * @param parent_group : index to the group to split
   * @param groups : reference to a vector of groups
   */
  void _splitGroup(size_t parent_group, std::vector<GarciaGroup> &groups, VectorXui &strategies);

  /**
   * @brief selects a group proportional to its total payoff with within group interactions.
   *
   * @param alpha : probability of interacting with members of the same group
   * @param groups : vector of groups
   * @param strategies : vector of strategy counts
   * @return : index of the selected individual
   */
  size_t
  _payoffProportionalSelection(const double &alpha, std::vector<GarciaGroup> &groups, VectorXui &strategies);

  /**
  * @brief selects a group proportional to its size.
  *
  * @tparam S : group container
  * @param groups : reference to the population groups
  * @return : index of the parent group
  */
  size_t _sizeProportionalSelection(std::vector<GarciaGroup> &groups);

  /**
  * @brief Checks whether a pseudo stationary state has been reached.
  *
  * @tparam S : group container
  * @param groups : reference to a vector of groups
  * @return true if reached a pseudo stationary state, otherwise false
  */
  bool _pseudoStationary(std::vector<GarciaGroup> &groups);

  /**
   * @brief sets the all individuals of one strategy
   *
   * Sets all individuals in the population of one strategy and sets all groups at maximum capacity.
   *
   * @tparam S : group container
   * @param strategy : resident strategy
   * @param groups : reference to a vector of groups
   */
  void _setFullHomogeneousState(size_t strategy, std::vector<GarciaGroup> &groups);

  /**
   * @brief Sets randomly the state of the population given a vector which contains the strategies in the population.
   *
   * This method shuffles a vector containing the population of strategies and then assigns each _group_size
   * of strategies to a group.
   *
   * @tparam S : container for the groups
   * @param groups : vector of groups
   * @param container : vector of strategies
   */
  inline void _setState(std::vector<GarciaGroup> &groups, std::vector<size_t> &container);

  /**
   * @brief returns the total population size
   * @tparam S : group container
   * @param groups : reference to a vector of groups
   * @return the sum of the sizes of all the groups
   */
  inline size_t _current_pop_size(std::vector<GarciaGroup> &groups);

  /**
   * @brief Performs the conflict resultion step of Garcia et al.
   *
   * @param kappa : average fraction of groups involved in conflict
   * @param z : importance of payoffs
   * @param groups : vector of groups
   * @param strategies : vector of strategies
   */
  inline void
  _resolve_conflict(const double &kappa, const double &z, std::vector<GarciaGroup> &groups,
                    VectorXui &strategies);

};

template<typename S>
MLS<S>::MLS(size_t generations, size_t nb_strategies,
            size_t group_size, size_t nb_groups, double w,
            const Eigen::Ref<const EGTTools::Vector> &strategies_freq,
            const Eigen::Ref<const EGTTools::Matrix2D> &payoff_matrix) : _generations(generations),
                                                                         _nb_strategies(nb_strategies),
                                                                         _group_size(group_size),
                                                                         _nb_groups(nb_groups),
                                                                         _w(w),
                                                                         _strategy_freq(strategies_freq),
                                                                         _payoff_matrix(payoff_matrix) {
  if (static_cast<size_t>(_payoff_matrix.rows() * _payoff_matrix.cols()) != (_nb_strategies * _nb_strategies))
    throw std::invalid_argument(
        "Payoff matrix has wrong dimensions it must have shape (nb_strategies, nb_strategies)");
  if (group_size < 4)
    throw std::invalid_argument(
        "The maximum group size must be at least 4.");
  _pop_size = _nb_groups * _group_size;
  // calculate the frequencies of each strategy in the population
  _strategies = VectorXui::Zero(_nb_strategies);
  // Calculate the number of individuals belonging to each strategy from the initial frequencies
  size_t tmp = 0;
  for (size_t i = 0; i < (_nb_strategies - 1); ++i) {
    _strategies(i) = (size_t) floor(_strategy_freq(i) * _pop_size);
    tmp += _strategies(i);
  }
  _strategies(_nb_strategies - 1) = (size_t) _pop_size - tmp;

  // Initialize random uniform distribution
  _uint_rand = std::uniform_int_distribution<size_t>(0, _nb_groups - 1);
  _uint_rand_strategy = std::uniform_int_distribution<size_t>(0, _nb_strategies - 1);
  _real_rand = std::uniform_real_distribution<double>(0.0, 1.0);
}

template<typename S>
Vector MLS<S>::evolve(double q, double w, const Eigen::Ref<const VectorXui> &init_state) {
  if ((_nb_groups == 1) && q != 0.)
    throw std::invalid_argument(
        "The splitting probability must be zero when there is only 1 group in the population");
  if (static_cast<size_t>(init_state.size()) != _nb_strategies)
    throw std::invalid_argument(
        "you must specify the number of individuals of each " + std::to_string(_nb_strategies) +
            " strategies");
  if (init_state.sum() != _pop_size)
    throw std::invalid_argument(
        "the sum of individuals in the initial state must be equal to " + std::to_string(_pop_size));

  _strategies.array() = init_state;
  // Initialize population with initial state
  VectorXui group_strategies = VectorXui::Zero(_nb_strategies);
  Group group(_nb_strategies, _group_size, w, group_strategies, _payoff_matrix);
  group.set_group_size(_group_size);
  std::vector<size_t> pop_container(_pop_size);
  // initialize container
  size_t z = 0, sum = 1;
  for (size_t i = 0; i < _nb_strategies; ++i) {
    for (size_t j = 0; j < init_state(i); ++j) {
      pop_container[z++] = i;
    }
  }
  std::vector<Group> groups(_nb_groups, group);
  _setState(groups, pop_container);


  // Then we run the Moran Process
  for (size_t t = 0; t < _generations; ++t) {
    _update(q, groups, _strategies);
    sum = _strategies.sum();
    if ((_strategies.array() == sum).any()) break;
  } // end Moran process loop
  return _strategies.cast<double>() / static_cast<double>(sum);

}

template<typename S>
Vector MLS<S>::evolve(double q, double w, double lambda, const Eigen::Ref<const VectorXui> &init_state) {
  if ((_nb_groups == 1) && q != 0.)
    throw std::invalid_argument(
        "The splitting probability must be zero when there is only 1 group in the population");
  if ((q < 0.) || (q > 1.))
    throw std::invalid_argument(
        "q must be in the range [0, 1]");
  if ((w < 0.) || (w > 1.))
    throw std::invalid_argument(
        "w must be in the range [0, 1]");
  if ((lambda < 0.) || (lambda > 1.))
    throw std::invalid_argument(
        "lambda must be in the range [0, 1]");
  if (static_cast<size_t>(init_state.size()) != _nb_strategies)
    throw std::invalid_argument(
        "you must specify the number of individuals of each " + std::to_string(_nb_strategies) +
            " strategies");
  if (init_state.sum() != _pop_size)
    throw std::invalid_argument(
        "the sum of individuals in the initial state must be equal to " + std::to_string(_pop_size));

  _strategies.array() = init_state;
  // Initialize population with initial state
  VectorXui group_strategies = VectorXui::Zero(_nb_strategies);
  Group group(_nb_strategies, _group_size, w, group_strategies, _payoff_matrix);
  group.set_group_size(_group_size);
  std::vector<size_t> pop_container(_pop_size);
  // initialize container
  size_t z = 0, sum = 1;
  for (size_t i = 0; i < _nb_strategies; ++i) {
    for (size_t j = 0; j < init_state(i); ++j) {
      pop_container[z++] = i;
    }
  }
  std::vector<Group> groups(_nb_groups, group);
  _setState(groups, pop_container);


  // Then we run the Moran Process
  for (size_t t = 0; t < _generations; ++t) {
    _update(q, lambda, groups, _strategies);
    sum = _strategies.sum();
    if ((_strategies.array() == sum).any()) break;
  } // end Moran process loop
  return _strategies.cast<double>() / static_cast<double>(sum);

}

template<typename S>
Vector MLS<S>::evolve(double q, double w, double lambda, double kappa, double z,
                      const Eigen::Ref<const VectorXui> &init_state) {
  if ((_nb_groups == 1) && q != 0.)
    throw std::invalid_argument(
        "The splitting probability must be zero when there is only 1 group in the population");
  if ((q < 0.) || (q > 1.))
    throw std::invalid_argument(
        "q must be in the range [0, 1]");
  if ((w < 0.) || (w > 1.))
    throw std::invalid_argument(
        "w must be in the range [0, 1]");
  if ((lambda < 0.) || (lambda > 1.))
    throw std::invalid_argument(
        "lambda must be in the range [0, 1]");
  if ((kappa < 0.) || (kappa > 1.))
    throw std::invalid_argument(
        "kappa must be in the range [0, 1]");
  if (z < 0.)
    throw std::invalid_argument(
        "z must be in the range [0, +Inf)");
  if (static_cast<size_t>(init_state.size()) != _nb_strategies)
    throw std::invalid_argument(
        "you must specify the number of individuals of each " + std::to_string(_nb_strategies) +
            " strategies");
  if (init_state.sum() != _pop_size)
    throw std::invalid_argument(
        "the sum of individuals in the initial state must be equal to " + std::to_string(_pop_size));

  _strategies.array() = init_state;
  // Initialize population with initial state
  VectorXui group_strategies = VectorXui::Zero(_nb_strategies);
  Group group(_nb_strategies, _group_size, w, group_strategies, _payoff_matrix);
  group.set_group_size(_group_size);
  std::vector<size_t> pop_container(_pop_size);
  // initialize container
  size_t it = 0, sum = 1;
  for (size_t i = 0; i < _nb_strategies; ++i) {
    for (size_t j = 0; j < init_state(i); ++j) {
      pop_container[it++] = i;
    }
  }
  std::vector<Group> groups(_nb_groups, group);
  _setState(groups, pop_container);


  // Then we run the Moran Process
  for (size_t t = 0; t < _generations; ++t) {
    _update(q, lambda, kappa, z, groups, _strategies);
    sum = _strategies.sum();
    if ((_strategies.array() == sum).any()) break;
  } // end Moran process loop
  return _strategies.cast<double>() / static_cast<double>(sum);

}

template<typename S>
double
MLS<S>::fixationProbability(size_t invader, size_t resident, size_t runs, double q, double w) {
  if (invader > _nb_strategies || resident > _nb_strategies)
    throw std::invalid_argument(
        "you must specify a valid index for invader and resident [0, " + std::to_string(_nb_strategies) +
            ")");
  if ((_nb_groups == 1) && q != 0.)
    throw std::invalid_argument(
        "The splitting probability must be zero when there is only 1 group in the population");

  double r2m = 0; // resident to mutant count
  double r2r = 0; // resident to resident count
  VectorXui group_strategies = VectorXui::Zero(_nb_strategies);
  group_strategies(resident) = _group_size;

  // This loop can be done in parallel
#pragma omp parallel for default(none) shared(group_strategies, invader, resident, runs, q, w, \
_nb_strategies, _group_size, \
_payoff_matrix, _nb_groups, _pop_size, _generations) reduction(+:r2m, r2r)
  for (size_t i = 0; i < runs; ++i) {
    // First we initialize a homogeneous population with the resident strategy
    SED::Group group(_nb_strategies, _group_size, w, group_strategies, _payoff_matrix);
    std::vector<SED::Group> groups(_nb_groups, group);
    VectorXui strategies = VectorXui::Zero(_nb_strategies);
    strategies(resident) = _pop_size;

    // Then we create a mutant of the invading strategy
    _createMutant(invader, resident, groups);
    // Update full population frequencies
    _updateFullPopulationFrequencies(invader, resident, strategies);

    // Then we run the Moran Process
    for (size_t t = 0; t < _generations; ++t) {
      _update(q, groups, strategies);

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

template<typename S>
double
MLS<S>::fixationProbability(size_t invader, size_t resident, size_t runs,
                            double q, double lambda, double w) {
  if (invader > _nb_strategies || resident > _nb_strategies)
    throw std::invalid_argument(
        "you must specify a valid index for invader and resident [0, " + std::to_string(_nb_strategies) +
            ")");
  if ((_nb_groups == 1) && q != 0.)
    throw std::invalid_argument(
        "The splitting probability must be zero when there is only 1 group in the population");

  double r2m = 0; // resident to mutant count
  double r2r = 0; // resident to resident count
  VectorXui group_strategies = VectorXui::Zero(_nb_strategies);
  group_strategies(resident) = _group_size;

  // This loop can be done in parallel
#pragma omp parallel for default(none) shared(group_strategies, invader, resident, runs, q, w, lambda, \
_nb_strategies, _group_size, \
_payoff_matrix, _nb_groups, _pop_size, _generations) reduction(+:r2m, r2r)
  for (size_t i = 0; i < runs; ++i) {
    // First we initialize a homogeneous population with the resident strategy
    SED::Group group(_nb_strategies, _group_size, w, group_strategies, _payoff_matrix);
    std::vector<SED::Group> groups(_nb_groups, group);
    VectorXui strategies = VectorXui::Zero(_nb_strategies);
    strategies(resident) = _pop_size;

    // Then we create a mutant of the invading strategy
    _createMutant(invader, resident, groups);
    // Update full population frequencies
    _updateFullPopulationFrequencies(invader, resident, strategies);

    // Then we run the Moran Process
    for (size_t t = 0; t < _generations; ++t) {
      _update(q, lambda, groups, strategies);

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

template<typename S>
double MLS<S>::fixationProbability(size_t invader, size_t resident, size_t runs,
                                   double q, double lambda, double w, double kappa, double z) {
  if (invader > _nb_strategies || resident > _nb_strategies)
    throw std::invalid_argument(
        "you must specify a valid index for invader and resident [0, " + std::to_string(_nb_strategies) +
            ")");
  if ((_nb_groups == 1) && q != 0.)
    throw std::invalid_argument(
        "The splitting probability must be zero when there is only 1 group in the population");

  double r2m = 0; // resident to mutant count
  double r2r = 0; // resident to resident count
  VectorXui group_strategies = VectorXui::Zero(_nb_strategies);
  group_strategies(resident) = _group_size;

  // This loop can be done in parallel
#pragma omp parallel for default(none) shared(group_strategies, invader, resident, runs, q, w, lambda, kappa, z, \
_nb_strategies, _group_size, \
_payoff_matrix, _nb_groups, _pop_size, _generations) reduction(+:r2m, r2r)
  for (size_t i = 0; i < runs; ++i) {
    // First we initialize a homogeneous population with the resident strategy
    SED::Group group(_nb_strategies, _group_size, w, group_strategies, _payoff_matrix);
    std::vector<SED::Group> groups(_nb_groups, group);
    VectorXui strategies = VectorXui::Zero(_nb_strategies);
    strategies(resident) = _pop_size;

    // Then we create a mutant of the invading strategy
    _createMutant(invader, resident, groups);
    // Update full population frequencies
    _updateFullPopulationFrequencies(invader, resident, strategies);

    // Then we run the Moran Process
    for (size_t t = 0; t < _generations; ++t) {
      _update(q, lambda, kappa, z, groups, strategies);

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

template<typename S>
Vector MLS<S>::fixationProbability(size_t invader, const Eigen::Ref<const VectorXui> &init_state, size_t runs,
                                   double q, double w) {
  if (invader > _nb_strategies)
    throw std::invalid_argument(
        "you must specify a valid index for invader and resident [0, " + std::to_string(_nb_strategies) +
            ")");
  if ((_nb_groups == 1) && q != 0.)
    throw std::invalid_argument(
        "The splitting probability must be zero when there is only 1 group in the population");
  if (static_cast<size_t>(init_state.size()) != _nb_strategies)
    throw std::invalid_argument(
        "you must specify the number of individuals of each " + std::to_string(_nb_strategies) +
            " strategies");
  if (init_state.sum() != _pop_size)
    throw std::invalid_argument(
        "the sum of individuals in the initial state must be equal to " + std::to_string(_pop_size));

  Vector fixations = Vector::Zero(_nb_strategies);

  // Initialize population with initial state
  VectorXui group_strategies = VectorXui::Zero(_nb_strategies);
  Group group(_nb_strategies, _group_size, w, group_strategies, _payoff_matrix);
  group.set_group_size(_group_size);
  std::vector<size_t> pop_container(_pop_size);
  // initialize container
  size_t z = 0;
  for (size_t i = 0; i < _nb_strategies; ++i) {
    for (size_t j = 0; j < init_state(i); ++j) {
      pop_container[z++] = i;
    }
  }

  // This loop can be done in parallel
#pragma omp parallel for default(none) shared(group, pop_container, init_state, invader, runs, q, w, \
group, _nb_groups, _generations, \
_nb_strategies) reduction(+:fixations)
  for (size_t i = 0; i < runs; ++i) {
    // First we initialize a homogeneous population with the resident strategy
    bool fixated = false;
    std::vector<Group> groups(_nb_groups, group);
    VectorXui strategies = init_state;
    std::vector<size_t> container(pop_container);
    _setState(groups, container);

    // Then we create a mutant of the invading strategy
    _createRandomMutant(invader, groups, strategies);

    // Then we run the Moran Process
    for (size_t t = 0; t < _generations; ++t) {
      _update(q, groups, strategies);
      size_t sum = strategies.sum();
      for (size_t s = 0; s < _nb_strategies; ++s)
        if (strategies(s) == sum) {
          fixations(s) += 1.0;
          fixated = true;
          break;
        }
      if (fixated) break;
    } // end Moran process loop
  } // end runs loop

  double tmp = fixations.sum();

  if (tmp > 0.0)
    return fixations.array() / tmp;

  return fixations.array();
}

template<typename S>
Vector
MLS<S>::gradientOfSelection(size_t invader, size_t resident, size_t runs, double w, double q) {
  if (invader > _nb_strategies || resident > _nb_strategies)
    throw std::invalid_argument(
        "you must specify a valid index for invader and resident [0, " + std::to_string(_nb_strategies) +
            ")");
  if ((_nb_groups == 1) && q != 0.)
    throw std::invalid_argument(
        "The splitting probability must be zero when there is only 1 group in the population");

  Vector gradient = Vector::Zero(_pop_size + 1);

  // This loop can be done in parallel
#pragma omp parallel for default(none) shared(gradient, invader, resident, runs, w, q, \
_pop_size, _nb_strategies, _group_size, _payoff_matrix, _nb_groups)
  for (size_t k = 1; k < _pop_size; ++k) { // Loops over all population configurations
    VectorXui strategies = VectorXui::Zero(_nb_strategies);
    Group group(_nb_strategies, _group_size, w, strategies, _payoff_matrix);
    group.set_group_size(_group_size);
    std::vector<Group> groups(_nb_groups, group);
    std::vector<size_t> pop_container(_pop_size);
    size_t t_plus = 0; // resident to mutant count
    size_t t_minus = 0; // resident to resident count
    strategies(resident) = _pop_size - k;
    strategies(invader) = k;
    // initialize container
    for (size_t i = 0; i < k; ++i) pop_container[i] = invader;
    for (size_t i = k; i < _pop_size; ++i) pop_container[i] = resident;

    // Calculate T+ and T-
    for (size_t i = 0; i < runs; ++i) {
      // First we initialize a homogeneous population with the resident strategy
      _setState(groups, pop_container);
      _update(q, groups, strategies);
      auto sum = static_cast<double>(strategies(invader) + strategies(resident));
      if (static_cast<double>(strategies(invader)) / sum >
          static_cast<double>(k) / static_cast<double>(_pop_size)) {
        ++t_plus;
      } else if (static_cast<double>(strategies(invader)) / sum <
          static_cast<double>(k) / static_cast<double>(_pop_size)) {
        ++t_minus;
      }
      strategies(resident) = _pop_size - k;
      strategies(invader) = k;
    }
    // Calculate gradient
    gradient(k) = (static_cast<double>(t_plus) - static_cast<double>(t_minus)) / static_cast<double>(runs);
  }

  return gradient;
}

template<typename S>
Vector
MLS<S>::gradientOfSelection(size_t invader, size_t resident, const Eigen::Ref<const VectorXui> &init_state,
                            size_t runs, double w, double q) {
  if (invader > _nb_strategies)
    throw std::invalid_argument(
        "you must specify a valid index for invader and resident [0, " + std::to_string(_nb_strategies) +
            ")");
  if (_nb_groups == 1 && q > 0.0)
    throw std::invalid_argument(
        "The splitting probability must be zero when there is only 1 group in the population");

  if (init_state.sum() != _pop_size)
    throw std::invalid_argument(
        "the sum of individuals in the initial state must be equal to " + std::to_string(_pop_size));

  Vector gradient = Vector::Zero(init_state(resident) + 1);

  // This loop can be done in parallel
#pragma omp parallel for default(none) shared(gradient, invader, resident, runs, w, q, init_state, \
_pop_size, _nb_strategies, _group_size, _payoff_matrix, _nb_groups, _pop_size)
  for (size_t k = 0; k <= init_state(resident); ++k) { // Loops over all population configurations
    VectorXui strategies = VectorXui::Zero(_nb_strategies);
    Group group(_nb_strategies, _group_size, w, strategies, _payoff_matrix);
    group.set_group_size(_group_size);
    std::vector<Group> groups(_nb_groups, group);
    std::vector<size_t> pop_container(_pop_size);
    size_t t_plus = 0; // resident to mutant count
    size_t t_minus = 0; // resident to resident count
    // initialize container
    size_t z = 0;
    for (size_t i = 0; i < _nb_strategies; ++i) {
      if (i == invader) strategies(i) = k;
      else if (i == resident) strategies(i) = init_state(i) - k;
      else strategies(i) = init_state(i);
      for (size_t j = 0; j < strategies(i); ++j) {
        pop_container[z++] = i;
      }
    }

    // Calculate T+ and T-
    for (size_t i = 0; i < runs; ++i) {
      // First we initialize a homogeneous population with the resident strategy
      _setState(groups, pop_container);
      _update(q, groups, strategies);
      auto sum = static_cast<double>(strategies.sum());
      if (strategies(invader) / sum > k / static_cast<double>(_pop_size)) {
        ++t_plus;
      } else if (strategies(invader) / sum < k / static_cast<double>(_pop_size)) {
        ++t_minus;
      }
      strategies.array() = init_state;
      strategies(invader) = k;
      strategies(resident) -= k;
    }
    // Calculate gradient
    gradient(k) = (static_cast<double>(t_plus) - static_cast<double>(t_minus)) / static_cast<double>(runs);
  }

  return gradient;
}

template<typename S>
void MLS<S>::_update(double q, std::vector<S> &groups, VectorXui &strategies) {
  _reproduce(groups, strategies, q);
}

template<typename S>
void MLS<S>::_update(double q, double lambda, std::vector<S> &groups, VectorXui &strategies) {
  _reproduce(groups, strategies, q);
  if (_real_rand(_mt) < lambda) _migrate(q, groups, strategies);
}

template<typename S>
void
MLS<S>::_update(double q, double lambda, double mu, std::vector<S> &groups, VectorXui &strategies) {
  _reproduce(groups, strategies, q);
  if (_real_rand(_mt) < lambda) _migrate(q, groups, strategies);
  if (_real_rand(_mt) < mu) _mutate(groups, strategies);
}

template<typename S>
void
MLS<S>::_update(double q, double lambda, double kappa, double z, std::vector<S> &groups, VectorXui &strategies) {
  _reproduce_garcia(groups, strategies, lambda);
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

template<typename S>
void
MLS<S>::_update(double q, double lambda, double alpha, double kappa, double z,
                std::vector<S> &groups,
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

template<typename S>
void MLS<S>::_speedUpdate(double q, std::vector<S> &groups, VectorXui &strategies) {
  if (!_pseudoStationary(groups)) {
    _reproduce(groups, strategies, q);
  } else { // If the groups have reached maximum size and the population is monomorphic
    if (_real_rand(_mt) < q) _reproduce(groups, strategies);
  }
}

template<typename S>
void MLS<S>::_speedUpdate(double q, double lambda, std::vector<S> &groups, VectorXui &strategies) {
  if (!_pseudoStationary(groups)) {
    _reproduce(groups, strategies, q);
    if (_real_rand(_mt) < lambda) _migrate(q, groups, strategies);
  } else { // If the groups have reached maximum size and the population is monomorphic
    if ((_real_rand(_mt) * (q + lambda)) < q) _reproduce(groups, strategies);
    else _migrate(q, groups, strategies);
  }
}

template<typename S>
void
MLS<S>::_speedUpdate(double q, double lambda, double mu, std::vector<S> &groups,
                     VectorXui &strategies) {
  if (!_pseudoStationary(groups)) {
    _reproduce(groups, strategies, q);
    if (_real_rand(_mt) < lambda) _migrate(q, groups, strategies);
    if (_real_rand(_mt) < mu) _mutate(groups, strategies);
  } else { // If the groups have reached maximum size and the population is monomorphic
    double p = _real_rand(_mt) * (q + lambda + mu);
    if (p <= q) _reproduce(groups, strategies);
    else if (p <= (q + lambda)) _migrate(q, groups, strategies);
    else _mutate(groups, strategies);
  }
}

template<typename S>
void MLS<S>::_createMutant(size_t invader, size_t resident, std::vector<S> &groups) {
  auto mutate_group = _uint_rand(_mt);
  groups[mutate_group].createMutant(invader, resident);
}

template<typename S>
void MLS<S>::_createRandomMutant(size_t invader, std::vector<S> &groups, EGTTools::VectorXui &strategies) {
  auto mutate_group = _uint_rand(_mt);
  size_t mutating_strategy = groups[mutate_group].deleteMember(_mt);
  groups[mutate_group].addMember(invader);
  --strategies(mutating_strategy);
  ++strategies(invader);
}

template<typename S>
void MLS<S>::_updateFullPopulationFrequencies(size_t increase, size_t decrease,
                                              EGTTools::VectorXui &strategies) {
  ++strategies(increase);
  --strategies(decrease);
}

template<typename S>
void MLS<S>::_reproduce(std::vector<S> &groups, VectorXui &strategies) {
  auto parent_group = _payoffProportionalSelection(groups);
  auto[split, new_strategy] = groups[parent_group].createOffspring(_mt);
  ++strategies(new_strategy);
  if (split) _splitGroup(parent_group, groups, strategies);
}

template<typename S>
void MLS<S>::_reproduce(std::vector<S> &groups, VectorXui &strategies, double q) {
  auto parent_group = _payoffProportionalSelection(groups);
  auto[split, new_strategy] = groups[parent_group].createOffspring(_mt);
  ++strategies(new_strategy);
  if (split) {
    if (_real_rand(_mt) < q) { // split group
      _splitGroup(parent_group, groups, strategies);
    } else { // remove individual
      size_t deleted_strategy = groups[parent_group].deleteMember(_mt);
      --strategies(deleted_strategy);
    }
  }
}

template<typename S>
void MLS<S>::_reproduce_garcia(std::vector<S> &groups, VectorXui &strategies, const double &lambda) {
  auto parent_group = _payoffProportionalSelection(groups);
  auto[split, new_strategy] = groups[parent_group].createOffspring(_mt);
  ++strategies(new_strategy);
  if (_real_rand(_mt) < lambda) _migrate(parent_group, new_strategy, groups);
}

template<typename S>
void
MLS<S>::_reproduce_garcia(std::vector<S> &groups, VectorXui &strategies, const double &lambda,
                          const double &alpha) {
  auto parent_group = _payoffProportionalSelection(alpha, groups, strategies);
  auto[split, new_strategy] = groups[parent_group].createOffspring(_mt);
  ++strategies(new_strategy);
  if (_real_rand(_mt) < lambda) _migrate(parent_group, new_strategy, groups);
  else groups[parent_group].totalPayoff();
}

template<typename S>
void MLS<S>::_migrate(double q, std::vector<S> &groups, VectorXui &strategies) {
  size_t parent_group, child_group, migrating_strategy;

  parent_group = _sizeProportionalSelection(groups);
  while (groups[parent_group].group_size() < 2) parent_group = _uint_rand(_mt);
  child_group = _uint_rand(_mt);
  // Makes sure that parent group and child group are different
  while (child_group == parent_group) child_group = _uint_rand(_mt);
  // First we delete a random member from the parent group
  migrating_strategy = groups[parent_group].deleteMember(_mt);
  // Then add the member to the child group
  if (groups[child_group].addMember(migrating_strategy)) {
    if (_real_rand(_mt) < q) _splitGroup(child_group, groups, strategies);
    else { // in case we delete a random member, that strategy will diminish in the population
      migrating_strategy = groups[child_group].deleteMember(_mt);
      --strategies(migrating_strategy);
    }
  }
}

template<typename S>
void
MLS<S>::_migrate(const size_t &parent_group, const size_t &migrating_strategy,
                 std::vector<S> &groups) {
  size_t child_group = _uint_rand(_mt);
  while (child_group == parent_group) child_group = _uint_rand(_mt);
  // First we delete the migrating strategy from the parent group
  groups[parent_group].deleteMember(migrating_strategy);
  // Then add the member to the randomly selected group
  groups[child_group].addMember(migrating_strategy);
  // Update group payoffs
  groups[child_group].totalPayoff();
}

template<typename S>
void MLS<S>::_mutate(std::vector<S> &groups, VectorXui &strategies) {
  size_t parent_group, mutating_strategy, new_strategy;

  parent_group = _sizeProportionalSelection(groups);
  mutating_strategy = groups[parent_group].deleteMember(_mt);
  new_strategy = _uint_rand_strategy(_mt);
  while (mutating_strategy == new_strategy) new_strategy = _uint_rand_strategy(_mt);
  groups[parent_group].addMember(new_strategy);
  --strategies(mutating_strategy);
  ++strategies(new_strategy);
}

template<typename S>
void MLS<S>::_splitGroup(size_t parent_group, std::vector<S> &groups, VectorXui &strategies) {
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

template<typename S>
size_t MLS<S>::_payoffProportionalSelection(std::vector<S> &groups) {
  double total_fitness = 0.0, tmp = 0.0;
  // Calculate total fitness
  for (auto &group: groups) total_fitness += group.totalPayoff();
  total_fitness *= _real_rand(_mt);
  size_t parent_group = 0;
  for (parent_group = 0; parent_group < _nb_groups; ++parent_group) {
    tmp += groups[parent_group].group_fitness();
    if (tmp > total_fitness) return parent_group;
  }

  return 0;
}

template<typename S>
size_t MLS<S>::_payoffProportionalSelection(const double &alpha, std::vector<S> &groups,
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

template<typename S>
size_t MLS<S>::_sizeProportionalSelection(std::vector<S> &groups) {
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

template<typename S>
bool MLS<S>::_pseudoStationary(std::vector<S> &groups) {
  if (_current_pop_size(groups) < _pop_size) return false;
  for (auto &group: groups)
    if (!group.isPopulationMonomorphic())
      return false;

  return true;
}

template<typename S>
void MLS<S>::_setFullHomogeneousState(size_t strategy, std::vector<S> &groups) {
  for (auto &group: groups)
    group.setPopulationHomogeneous(strategy);
}

template<typename S>
size_t MLS<S>::_current_pop_size(std::vector<S> &groups) {
  size_t size = 0;
  for (auto &group: groups) size += group.group_size();

  return size;
}

template<typename S>
void MLS<S>::_setState(std::vector<S> &groups, std::vector<size_t> &container) {
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

template<typename S>
void
MLS<S>::_resolve_conflict(const double &kappa, const double &z, std::vector<S> &groups,
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
}

#endif //DYRWIN_SED_MLS_HPP

# Copyright (c) 2019-2021  Elias Fernandez
#
# This file is part of EGTtools.
#
# EGTtools is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# EGTtools is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with EGTtools.  If not, see <http://www.gnu.org/licenses/>

"""
This python module contains the necessary functions
to calculate analytically the evolutionary dynamics in Infinite and Finite
populations on 2-player games.
"""

import numpy as np
import numpy.typing as npt
from scipy.sparse import lil_matrix, csr_matrix
from scipy.linalg import eig
from scipy.stats import hypergeom, multivariate_hypergeom, multinomial
from itertools import permutations
from typing import Tuple, Optional
from warnings import warn
from .. import sample_simplex, calculate_nb_states, calculate_state


def replicator_equation(x: np.ndarray, payoffs: np.ndarray) -> np.ndarray:
    """
    Produces the discrete time derivative of the replicator dynamics

    This only works for 2-player games.

    Parameters
    ----------
    x : numpy.ndarray[numpy.float64[m,1]]
        array containing the frequency of each strategy in the population.
    payoffs : numpy.ndarray[numpy.float64[m,m]]
        payoff matrix
    Returns
    -------
    numpy.ndarray
        time derivative of x

    See Also
    --------
    egttools.analytical.StochDynamics
    egttools.numerical.PairwiseComparisonNumerical
    """
    ax = np.dot(payoffs, x)
    return x * (ax - np.dot(x, ax))


def replicator_equation_n_player(x: np.ndarray, payoffs: np.ndarray, group_size: int) -> np.ndarray:
    """
    Replicator dynamics in N-player games

    The replicator equation is of the form

    .. math::
        g(x) \\equiv \\dot{x_{i}} = x_{i}(f_{i}(x) - \\sum_{j=1}^{N}{x_{j}f_{j}(x))

    Which can also be represented using a pairwise comparison rule as:

    .. math::
        \\dot{x_{i}} = x_{i}\\sum_{j}(f_{ij}(x) - f_{ji}(x))x_{j}

    For N-player games, to calculate the fitness of a strategy given a population state, we
    need to calculate the probability of each possible group configuration. This can be obtained
    by summing for each possible group configuration the payoff of strategy i times the probability
    of the group configurations occurring.

    Parameters
    ----------
    x : numpy.ndarray
        A vector of shape (1, nb_strategies), which contains the current frequency of each strategy in the population.
    payoffs : numpy.ndarray
        Payoff matrix. Each row represents a strategy and each column a possible group configuration.
        Each entry in the matrix should give the expected payoff for each row strategy for a given column group
        configuration.
    group_size : int
        Size of the group.

    Returns
    -------
    numpy.ndarray
        A vector of shape (1, nb_strategies), which contains the change in frequency of each strategy in the population
        (so the gradient).

    """
    fitness = np.zeros(shape=(len(x),))
    fitness_avg = 0.
    nb_group_configurations = calculate_nb_states(group_size, len(x))
    for strategy_index in range(len(x)):
        for i in range(nb_group_configurations):
            group_configuration = sample_simplex(i, group_size, len(x))
            if group_configuration[strategy_index] > 0:
                group_configuration[strategy_index] -= 1
                prob = multinomial.pmf(group_configuration, group_size - 1, x)
                fitness[strategy_index] += prob * payoffs[strategy_index, i]
        fitness_avg += x[strategy_index] * fitness[strategy_index]

    return x * (fitness - fitness_avg)


class StochDynamics:
    """A class containing methods to calculate the stochastic evolutionary dynamics of a population.

    Defines a class that contains methods to compute the stationary distribution for
    the limit of small mutation (only the monomorphic states) and the full transition matrix.

    Parameters
    ----------
    nb_strategies : int
                number of strategies in the population
    payoffs : numpy.ndarray[numpy.float64[m,m]]
            Payoff matrix indicating the payoff of each strategy (rows) against each other (columns).
            When analyzing an N-player game (group_size > 2) the structure of the matrix is a bit more involved,
            and we can have 2 options for structuring the payoff matrix:

            1) If we consider a simplified version of the system with a reduced Markov Chain which only contains
            the states at the edges of the simplex (the Small Mutation Limit - SML), then, we can assume that, at most,
            there will be 2 strategies in a group at any given moment. In this case, StochDynamics expects
            a square matrix of size nb_strategies x nb_strategies, in which each entry is a function that takes
            2 positional arguments k and group_size, and an optional *args argument, and will return the expected payoff
            of the row strategy A in a group with k A strategists and group_size - k
            B strategists (the column strategy). For all the elements in the diagonal, only 1 strategy should be present
            in the group, thus, this function should always return the same value, i.e., the payoff of a row strategy
            when all individuals in the group adopt the same strategy. See below for an example.

            2) If we want to consider the full Markov Chain composed of all possible states in the simplex, then
            the payoff matrix should be of the shape nb_strategies x nb_group_configurations, where the number
            of group configurations can be calculated using `egttools.calculate_nb_states(group_size, nb_strategies)`.
            Moreover, the mapping between group configurations and integer indexes must be done using
            `egttools.sample_simplex(index, group_size, nb_strategies)`. See below for an example
    pop_size : int
            population size
    group_size : int
            group size
    mu : float
        mutation probability

    See Also
    --------
    egttools.numerical.PairwiseComparisonNumerical
    egttools.analytical.replicator_equation
    egttools.analytical.PairwiseComparison

    Notes
    -----
    We recommend that instead of`StochDynamics`, you use `PairwiseComparison` because the latter
    is implemented in C++, runs faster and supports more precise types.

    Examples
    --------
    Example of the payoff matrix for case 1) mu = 0:
        >>> def get_payoff_a_vs_b(k, group_size, *args):
        ...     pre_computed_payoffs = [4, 5, 2, ..., 4] # the size of this list should be group_size + 1
        ...     return pre_computed_payoffs[k]
        >>> def get_payoff_b_vs_a(k, group_size, *args):
        ...     pre_computed_payoffs = [0, 2, 1, ..., 0] # the size of this list should be group_size + 1
        ...     return pre_computed_payoffs[k]
        >>> def get_payoff_a_vs_a(k, group_size, *args):
        ...     pre_computed_payoffs = [1, 1, 1, ..., 1] # the size of this list should be group_size + 1
        ...     return pre_computed_payoffs[k]
        >>> def get_payoff_b_vs_b(k, group_size, *args):
        ...     pre_computed_payoffs = [0, 0, 0, ..., 0] # the size of this list should be group_size + 1
        ...     return pre_computed_payoffs[k]
        >>> payoff_matrix = np.array([
        ...     [get_payoff_A_vs_A, get_payoff_A_vs_B],
        ...     [get_payoff_B_vs_A, get_payoff_B_vs_B]
        ...     ])

    Example of payoff matrix for case 2) full markov chain (mu > 0):
        >>> import egttools
        >>> nb_group_combinations = egttools.calculate_nb_states(group_size, nb_strategies)
        >>> payoff_matrix = np.zeros(shape=(nb_strategies, nb_group_combinations))
        >>> for group_configuration_index in range(nb_group_combinations):
        ...     for strategy in range(nb_strategies):
        ...         group_configuration = egttools.sample_simplex(group_configuration_index, group_size, nb_strategies)
        ...         payoff_matrix[strategy, group_configuration_index] = get_payoff(strategy, group_configuration)
    """

    def __init__(self, nb_strategies: int, payoffs: np.ndarray, pop_size: int, group_size=2, mu=0) -> None:
        self.nb_strategies = nb_strategies
        self.payoffs = payoffs
        self.pop_size = pop_size
        self.group_size = group_size
        self.mu = mu
        self.nb_states_population = calculate_nb_states(pop_size, nb_strategies)
        self.nb_group_combinations = calculate_nb_states(group_size, nb_strategies)
        if group_size > 2:  # pairwise game
            self.fitness = self.fitness_group
            self.full_fitness = self.full_fitness_difference_group
        else:  # group game
            self.fitness = self.fitness_pair
            self.full_fitness = self.full_fitness_difference_pairwise

    def update_population_size(self, pop_size: int):
        """
        Updates the size of the population and the number of possible population states.

        Parameters
        ----------
        pop_size: New population size
        """
        self.pop_size = pop_size
        self.nb_states_population = calculate_nb_states(pop_size, self.nb_strategies)

    def update_group_size(self, group_size: int):
        """
        Updates the groups size of the game (and the methods used to compute the fitness)

        Parameters
        ----------
        group_size: new group size
        """
        self.group_size = group_size
        self.nb_group_combinations = calculate_nb_states(group_size, self.nb_strategies)
        if group_size > 2:  # pairwise game
            self.fitness = self.fitness_group
            self.full_fitness = self.full_fitness_difference_group
        else:  # group game
            self.fitness = self.fitness_pair
            self.full_fitness = self.full_fitness_difference_pairwise

    def update_payoffs(self, payoffs: np.ndarray, nb_strategies: Optional[int] = None):
        """
        Updates the payoff matrix

        Parameters
        ----------
        payoffs: payoff matrix
        nb_strategies: total number of strategies (optional). If not indicated, then the new payoff
                       matrix must have the same dimensions as the previous one
        """
        if nb_strategies is None:
            if payoffs.shape[0] != self.nb_strategies:
                raise ValueError("The number of rows of the payoff matrix must be equal to the number of strategies.")
        else:
            self.nb_strategies = nb_strategies
        self.payoffs = payoffs

    def fitness_pair(self, x: int, i: int, j: int, *args: Optional[list]) -> float:
        """
        Calculates the fitness of strategy i versus strategy j, in
        a population of x i-strategists and (pop_size-x) j strategists, considering
        a 2-player game.

        Parameters
        ----------
        x : int
            number of i-strategists in the population
        i : int
            index of strategy i
        j : int
            index of strategy j
        args : Optional[list]

        Returns
        -------
            float
            the fitness difference among the strategies
        """
        fitness_i = ((x - 1) * self.payoffs[i, i] +
                     (self.pop_size - x) * self.payoffs[i, j]) / (self.pop_size - 1)
        fitness_j = ((self.pop_size - x - 1) * self.payoffs[j, j] +
                     x * self.payoffs[j, i]) / (self.pop_size - 1)
        return fitness_i - fitness_j

    def full_fitness_difference_pairwise(self, i: int, j: int, population_state: np.ndarray) -> float:
        """
        Calculates the fitness of strategy i in a population with state :param population_state,
        assuming pairwise interactions (2-player game).

        Parameters
        ----------
        i : int
            index of the strategy that will reproduce
        j : int
            index of the strategy that will die
        population_state : numpy.ndarray[numpy.int64[m,1]]
                           vector containing the counts of each strategy in the population

        Returns
        -------
        float
        The fitness difference between the two strategies for the given population state
        """
        fitness_i = (population_state[i] - 1) * self.payoffs[i, i]
        for strategy in range(self.nb_strategies):
            if strategy == i:
                continue
            fitness_i += population_state[strategy] * self.payoffs[i, strategy]
        fitness_j = (population_state[j] - 1) * self.payoffs[j, j]
        for strategy in range(self.nb_strategies):
            if strategy == j:
                continue
            fitness_j += population_state[strategy] * self.payoffs[j, strategy]

        return (fitness_i - fitness_j) / (self.pop_size - 1)

    def fitness_group(self, x: int, i: int, j: int, *args: Optional[list]) -> float:
        """
        In a population of x i-strategists and (pop_size-x) j strategists, where players
        interact in group of 'group_size' participants this function
        returns the average payoff of strategies i and j. This function expects
        that

        .. math::
            x \\in [1,pop_size-1]

        Parameters
        ----------
        x : int
            number of individuals adopting strategy i in the population
        i : int
            index of strategy i
        j : int
            index of strategy j
        args : Optional[list]
            Other Parameters. This can be used to pass extra parameters to functions
            stored in the payoff matrix

        Returns
        -------
            float
            Returns the difference in fitness between strategy i and j
        """
        k_array_1 = np.arange(0, self.group_size, dtype=np.int64)
        k_array_2 = np.arange(0, self.group_size, dtype=np.int64)
        i_pmf = hypergeom(self.pop_size - 1, x - 1, self.group_size - 1).pmf(k_array_1)
        j_pmf = hypergeom(self.pop_size - 1, x, self.group_size - 1).pmf(k_array_2)

        fitness_i, fitness_j = 0, 0
        for k in k_array_1:
            fitness_i += self.payoffs[i, j](k + 1, self.group_size, *args) * i_pmf[k]
            fitness_j += self.payoffs[j, i](self.group_size - k, self.group_size, *args) * j_pmf[k]

        return fitness_i - fitness_j

    def full_fitness_difference_group(self, i: int, j: int, population_state: np.ndarray) -> float:
        """
        Calculate the fitness difference between strategies :param i and :param j
        assuming that player interacts in groups of size group_size > 2 (n-player games).

        Parameters
        ----------
        i : int
            index of the strategy that will reproduce
        j : int
            index of the strategy that will die
        population_state : numpy.ndarray[numpy.int64[m,1]]
                           vector containing the counts of each strategy in the population

        Returns
        -------
        float
        The fitness difference between strategies i and j
        """
        copy1 = population_state.copy()
        copy1[i] -= 1
        copy2 = population_state.copy()
        copy2[j] -= 1
        rv_i = multivariate_hypergeom(copy1, self.group_size - 1)
        rv_j = multivariate_hypergeom(copy2, self.group_size - 1)

        fitness_i, fitness_j = 0., 0.
        for group_index in range(self.nb_group_combinations):
            group = sample_simplex(group_index, self.group_size, self.nb_strategies)
            if group[i] > 0:
                group[i] -= 1
                fitness_i += self.payoffs[i, group_index] * rv_i.pmf(group)
                group[i] += 1
            if group[j] > 0:
                group[j] -= 1
                fitness_j += self.payoffs[j, group_index] * rv_j.pmf(group)
                group[j] += 1

        return fitness_i - fitness_j

    @staticmethod
    def fermi(beta: float, fitness_diff: float) -> npt.ArrayLike:
        """
        The fermi function determines the probability that the first type imitates the second.

        Parameters
        ----------
        beta : float
            intensity of selection
        fitness_diff : float
            Difference in fitness between the strategies (f_a - f_b).

        Returns
        -------
        numpy.typing.ArrayLike
            the probability of imitation
        """
        return np.clip(1. / (1. + np.exp(beta * fitness_diff, dtype=np.float64)), 0., 1.)

    def prob_increase_decrease(self, k: int, invader: int, resident: int,
                               beta: float, *args: Optional[list]) -> Tuple[npt.ArrayLike, npt.ArrayLike]:
        """
        This function calculates for a given number of invaders the probability
        that the number increases or decreases with one.

        Parameters
        ----------
        k : int
            number of invaders in the population
        invader: int
            index of the invading strategy
        resident: int
            index of the resident strategy
        beta: float
            intensity of selection
        args: Optional[list]
            other arguments. Can be used to pass extra arguments to functions contained
            in the payoff matrix.
        Returns
        -------
        Tuple[numpy.typing.ArrayLike, numpy.typing.ArrayLike]
            tuple(probability of increasing the number of invaders, probability of decreasing)
        """
        if (k == self.pop_size) or (k == 0):
            increase = 0
            decrease = 0
        else:
            fitness_diff = self.fitness(k, invader, resident, *args)
            increase = (((self.pop_size - k) / self.pop_size) *
                        (k / (self.pop_size - 1))) * StochDynamics.fermi(-beta,
                                                                         fitness_diff)

            decrease = ((k / self.pop_size) * ((self.pop_size - k) /
                                               (self.pop_size - 1))) * StochDynamics.fermi(beta,
                                                                                           fitness_diff)
        return np.clip(increase, 0., 1.), np.clip(decrease, 0., 1.)

    def prob_increase_decrease_with_mutation(self, k: int, invader: int, resident: int, beta: float,
                                             *args: Optional[list]) -> Tuple[float, float]:
        """
        This function calculates for a given number of invaders the probability
        that the number increases or decreases with taking into account a mutation rate.

        Parameters
        ----------
        k : int
            number of invaders in the population
        invader: int
            index of the invading strategy
        resident: int
            index of the resident strategy
        beta: float
            intensity of selection
        args: Optional[list]
            other arguments. Can be used to pass extra arguments to functions contained
            in the payoff matrix.
        Returns
        -------
        Tuple[float, float]
            tuple(probability of increasing the number of invaders, probability of decreasing)
        """
        p_plus, p_less = self.prob_increase_decrease(k, invader, resident, beta, *args)
        p_plus = ((1 - self.mu) * p_plus) + (self.mu * ((self.pop_size - k) / self.pop_size))
        p_less = ((1 - self.mu) * p_less) + (self.mu * (k / self.pop_size))
        return p_plus, p_less

    def gradient_selection(self, k: int, invader: int, resident: int, beta: float, *args: Optional[list]) -> float:
        """
        Calculates the gradient of selection given an invader and a resident strategy.

        Parameters
        ----------
        k : int
            number of invaders in the population
        invader : int
            index of the invading strategy
        resident : int
            index of the resident strategy
        beta : float
            intensity of selection
        args : Optional[List]
            other arguments. Can be used to pass extra arguments to functions contained
            in the payoff matrix.

        Returns
        -------
        float
            The gradient of selection.
        """
        if k == 0:
            return 0
        elif k == self.pop_size:
            return 0
        else:
            return ((self.pop_size - k) / self.pop_size) * (k / (self.pop_size - 1)) * np.tanh(
                (beta / 2) * self.fitness(k, invader, resident, *args))

    def full_gradient_selection(self, population_state: np.ndarray, beta: float) -> np.ndarray:
        """
        Calculates the gradient of selection for an invading strategy, given a population state.

        Parameters
        ----------
        population_state : numpy.ndarray[np.int64[m,1]]
            structure of unsigned integers containing the
            counts of each strategy in the population
        beta : float
            intensity of selection

        Returns
        -------
        numpy.ndarray[numpy.float64[m,m]]
            Matrix indicating the likelihood of change in the population given a starting point.
        """
        probability_selecting_strategy_first = population_state / self.pop_size
        probability_selecting_strategy_second = population_state / self.pop_size
        probabilities = np.outer(probability_selecting_strategy_first, probability_selecting_strategy_second)
        fitness = np.zeros(shape=(self.nb_strategies, self.nb_strategies))
        for j in range(self.nb_strategies):
            if population_state[j] == 0:
                continue
            for i in range(self.nb_strategies):
                if population_state[i] == 0:
                    continue
                fitness[j, i] = self.full_fitness(i, j, population_state)

        return (probabilities * np.tanh((beta / 2) * fitness)).sum(axis=0) * (1 - self.mu) + (
                self.mu / (self.nb_strategies - 1)) * probability_selecting_strategy_second

    def full_gradient_selection_without_mutation(self, population_state: np.ndarray, beta: float) -> np.ndarray:
        """
        Calculates the gradient of selection for an invading strategy, given a population state. It does
        not take into account mutation.

        Parameters
        ----------
        population_state : numpy.ndarray[np.int64[m,1]]
            structure of unsigned integers containing the
            counts of each strategy in the population
        beta : float
            intensity of selection

        Returns
        -------
        numpy.ndarray[numpy.float64[m,m]]
            Matrix indicating the likelihood of change in the population given a starting point.
        """

        probability_selecting_strategy_first = population_state / self.pop_size
        probability_selecting_strategy_second = population_state / self.pop_size
        probabilities = np.outer(probability_selecting_strategy_first, probability_selecting_strategy_second)
        fitness = np.zeros(shape=(self.nb_strategies, self.nb_strategies))
        for j in range(self.nb_strategies):
            if population_state[j] == 0:
                continue
            for i in range(self.nb_strategies):
                if population_state[i] == 0:
                    continue
                fitness[j, i] = self.full_fitness(i, j, population_state)

        return (probabilities * np.tanh((beta / 2) * fitness)).sum(axis=0)

    def fixation_probability(self, invader: int, resident: int, beta: float, *args: Optional[list]) -> float:
        """
        Function for calculating the fixation_probability probability of the invader
        in a population of residents.

        TODO: Requires more testing!

        Parameters
        ----------
        invader : int
            index of the invading strategy
        resident : int
            index of the resident strategy
        beta : float
            intensity of selection
        args : Optional[list]
            Other arguments. Can be used to pass extra arguments to functions contained
            in the payoff matrix.

        Returns
        -------
        float
            The fixation_probability probability.

        See Also
        --------
        egttools.numerical.PairwiseComparisonNumerical
        """
        phi = 0.
        prod = 1.
        for i in range(1, self.pop_size):
            p_plus, p_minus = self.prob_increase_decrease(i, invader, resident, beta, *args)
            # this is necessary to avoid divisions by zero
            if np.isclose(p_plus, 0., atol=1e-12) and not np.isclose(p_plus, p_minus):
                return 0.
            prod *= p_minus / p_plus
            phi += prod
            # We can approximate by zero if phi is too big
            if phi > 1e7:
                return 0.0

        return 1.0 / (1.0 + phi)

    def calculate_full_transition_matrix(self, beta: float, *args: Optional[list]) -> csr_matrix:
        """
        Returns the full transition matrix in sparse representation.

        Parameters
        ----------
        beta : float
            Intensity of selection.
        args : Optional[list]
            Other arguments. Can be used to pass extra arguments to functions contained
            in the payoff matrix.

        Returns
        -------
        scipy.sparse.csr_matrix
            The full transition matrix between the two strategies in sparse format.
        """
        nb_states = calculate_nb_states(self.pop_size, self.nb_strategies)
        mutation_probability = (self.mu / (self.nb_strategies - 1))
        not_mu = 1. - self.mu

        transitions = lil_matrix((nb_states, nb_states), dtype=np.float64)
        possible_transitions = [1, -1]
        for i in range(self.nb_strategies - 2):
            possible_transitions.append(0)

        for i in range(nb_states):
            total_prob = 0.
            current_state = sample_simplex(i, self.pop_size, self.nb_strategies)
            # Check if we are in a monomorphic state
            monomorphic = True if (current_state == self.pop_size).any() else False

            # calculate probability of transitioning from current_tate to next_state
            for permutation in permutations(possible_transitions):
                # get new state
                new_state = current_state + permutation

                # Check if we are trying an impossible transition
                if (new_state < 0).any() or (new_state > self.pop_size).any():
                    continue

                new_state_index = calculate_state(self.pop_size, new_state)

                # If we are in a monomorphic population, transitions
                # can only happen if a mutation event occurs
                if monomorphic:
                    transitions[i, new_state_index] = mutation_probability
                else:
                    increase = np.where(np.array(permutation) == 1)[0][0]
                    decrease = np.where(np.array(permutation) == -1)[0][0]

                    if current_state[increase] == 0:
                        prob = (current_state[decrease] / self.pop_size) * mutation_probability
                    else:
                        # now we calculate the transition probability
                        fitness_diff = self.full_fitness(decrease, increase, current_state)
                        # Probability that the individual that will die is selected and that the individual that
                        # will be imitated is selected times the probability of imitation
                        prob = not_mu * (current_state[increase] / (self.pop_size - 1))
                        prob *= StochDynamics.fermi(beta, fitness_diff)
                        # The probability that there will not be a mutation event times the probability
                        # of the transition
                        # plus the probability that if there is a mutation event, the dying strategy is selected
                        # times the probability that it mutates into the increasing strategy
                        prob = (current_state[decrease] / self.pop_size) * (prob + mutation_probability)
                        total_prob += prob

                    transitions[i, new_state_index] = prob

            if monomorphic:
                transitions[i, i] = not_mu
            else:
                transitions[i, i] = 1. - total_prob

        return transitions.tocsr().transpose()

    def transition_and_fixation_matrix(self, beta: float, *args: Optional[list]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculates the transition matrix (only for the monomorphic states)
        and the fixation_probability probabilities.

        This method calculates the transitions between monomorphic states. Thus, it assumes
        that we are in the small mutation limit (SML) of the moran process. Only
        use this method if this assumption is reasonable.

        Parameters
        ----------
        beta : float
            Intensity of selection.
        args : Optional[list]
            Other arguments. Can be used to pass extra arguments to functions contained
            in the payoff matrix.
        Returns
        -------
        Tuple[numpy.ndarray[numpy.float64[m,m]], numpy.ndarray[numpy.float64[m,m]]]
            This method returns a tuple with the transition matrix as first element, and
            the matrix of fixation probabilities.
        """
        transitions = np.zeros((self.nb_strategies, self.nb_strategies))
        fixation_probabilities = np.zeros((self.nb_strategies, self.nb_strategies))

        for first in range(self.nb_strategies):
            transitions[first, first] = 1.
            for second in range(self.nb_strategies):
                if second != first:
                    fp = self.fixation_probability(second, first, beta, *args)
                    fixation_probabilities[first, second] = fp
                    tmp = fp / float(self.nb_strategies - 1)
                    transitions[first, second] = tmp
                    transitions[first, first] = transitions[first, first] - tmp

        return transitions.transpose(), fixation_probabilities

    def calculate_stationary_distribution(self, beta: float, *args: Optional[list]) -> np.ndarray:
        """
        Calculates the stationary distribution of the monomorphic states is mu = 0 (SML).
        Otherwise, it calculates the stationary distribution including all possible population states.

        This function is recommended only for Hermitian transition matrices.

        Parameters
        ----------
        beta : float
            intensity of selection.
        args : Optional[list]
            extra arguments for calculating payoffs.
        Returns
        -------
        numpy.ndarray
            A vector containing the stationary distribution
        """
        if self.mu > 0:
            t = self.calculate_full_transition_matrix(beta, *args).toarray()
        else:
            t, _ = self.transition_and_fixation_matrix(beta, *args)

        # Check if there is any transition with value 1 - this would mean that the game is degenerate
        if np.isclose(t, 1., atol=1e-11).any():
            warn(
                "Some of the entries in the transition matrix are close to 1 (with a tolerance of 1e-11). "
                "This could result in more than one eigenvalue of magnitude 1 "
                "(the Markov Chain is degenerate), so please be careful when analysing the results.", RuntimeWarning)

        # noinspection PyTupleAssignmentBalance
        eigenvalues, eigenvectors = eig(t)
        # calculate stationary distributions using eigenvalues and eigenvectors
        index_stationary = np.argmin(
            abs(eigenvalues - 1.0))  # look for the element closest to 1 in the list of eigenvalues
        sd = abs(eigenvectors[:, index_stationary].real)  # it is essential to access the matrix by column

        return sd / sd.sum()

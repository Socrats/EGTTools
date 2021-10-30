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
import scipy.sparse
from scipy.sparse import lil_matrix
from scipy.stats import hypergeom, multivariate_hypergeom
from typing import Tuple, Optional
from egttools import sample_simplex, calculate_nb_states


def replicator_equation(x: np.ndarray, payoffs: np.ndarray) -> np.ndarray:
    """
    Produces the discrete time derivative of the replicator dynamics

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
    egttools.numerical.PairwiseMoran
    """
    ax = np.dot(payoffs, x)
    return x * (ax - np.dot(x, ax))


class StochDynamics:
    """A class containing methods to calculate the stochastic evolutionary dynamics of a population.

    Defines a class that contains methods to compute the stationary distribution for
    the limit of small mutation (only the monomorphic states) and the full transition matrix.

    Parameters
    ----------
    nb_strategies : int
                number of strategies in the population
    pop_size : int
            population size
    payoffs : numpy.ndarray[numpy.float64[m,m]]
            payoff matrix indicating the payoff of each strategy (rows) against each other (columns)
    mu : float
        mutation probability

    See Also
    --------
    egttools.numerical.PairwiseMoran
    egttools.analytical.replicator_equation
    """

    def __init__(self, nb_strategies: int, payoffs: np.ndarray, pop_size: int, group_size=2, mu=0) -> None:
        self.nb_strategies = nb_strategies
        self.payoffs = payoffs
        self.Z = pop_size
        self.N = group_size
        self.mu = mu
        self.nb_states_population = calculate_nb_states(pop_size, nb_strategies)
        self.nb_group_combinations = calculate_nb_states(group_size, nb_strategies)
        if group_size > 2:  # pairwise game
            self.fitness = self.fitness_group
            self.full_fitness = self.full_fitness_difference_group
        else:  # group game
            self.fitness = self.fitness_pair
            self.full_fitness = self.full_fitness_difference_pairwise

    def fitness_pair(self, x: int, i: int, j: int, *args: Optional[list]) -> float:
        """
        Calculates the fitness of strategy i versus strategy j, in
        a population of x i-strategists and (Z-x) j strategists, considering
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
                     (self.Z - x) * self.payoffs[i, j]) / (self.Z - 1)
        fitness_j = ((self.Z - x - 1) * self.payoffs[j, j] +
                     x * self.payoffs[j, i]) / (self.Z - 1)
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

        return (fitness_i - fitness_j) / (self.Z - 1)

    def fitness_group(self, x: int, i: int, j: int, *args: Optional[list]) -> float:
        """
        In a population of x i-strategists and (Z-x) j strategists, where players
        interact in group of 'group_size' participants this function
        returns the average payoff of strategies i and j.

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
        k_array = np.arange(0, self.N, dtype=np.int32)
        i_pmf = hypergeom(self.Z - 1, x - 1, self.N - 1).pmf(k_array)
        j_pmf = hypergeom(self.Z - 1, x, self.N - 1).pmf(k_array)

        fitness_i, fitness_j = 0, 0
        for k in k_array:
            fitness_i += self.payoffs[i, j](k + 1, self.N, *args) * i_pmf[k]
            fitness_j += self.payoffs[j, i](self.N - k, self.N, *args) * j_pmf[k]

        return fitness_i - fitness_j

    def full_fitness_difference_group(self, i: int, j: int, population_state: np.ndarray) -> float:
        """
        Calculate the fitness difference between strategies :param i and :param j
        assuming that player interact in groups of size N > 2 (n-player games).

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
        population_state[i] -= 1
        rv_i = multivariate_hypergeom(population_state, self.N - 1)
        population_state[i] += 1
        population_state[j] -= 1
        rv_j = multivariate_hypergeom(population_state, self.N - 1)
        population_state[j] += 1

        fitness_i, fitness_j = 0., 0.
        for state_index in range(self.nb_group_combinations):
            group = sample_simplex(i, self.N, self.nb_strategies)
            if group[i] > 0:
                group[i] -= 1
                fitness_i += self.payoffs[i, state_index] * rv_i.pmf(x=group)
                group[i] += 1
            if group[j] > 0:
                group[i] -= 1
                fitness_j += self.payoffs[j, state_index] * rv_j.pmf(x=group)
                group[i] += 1

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
            Difference in fitneess between the strategies (f_a - f_b).

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
            index of the resitent strategy
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
        fitness_diff = self.fitness(k, invader, resident, *args)
        increase = (((self.Z - k) / float(self.Z)) * (k / float(self.Z - 1))) * StochDynamics.fermi(-beta, fitness_diff)
        decrease = ((k / float(self.Z)) * ((self.Z - k) / float(self.Z - 1))) * StochDynamics.fermi(beta, fitness_diff)
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
            index of the resitent strategy
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
        p_plus = ((1 - self.mu) * p_plus) + (self.mu * ((self.Z - k) / self.Z))
        p_less = ((1 - self.mu) * p_less) + (self.mu * (k / self.Z))
        return p_plus, p_less

    def full_prob_increase_decrease_with_mutation(self, population_state: np.ndarray, beta: float) -> np.ndarray:
        """
        Calculates the probabilities of increasing/decreasing the frequency of a strategy in each possible population
        state.

        Parameters
        ----------
        population_state : numpy.ndarray[numpy.int64[m,1]]
                           structure of unsigned integers containing the
                           counts of each strategy in the population
        beta : float
               intensity of selection

        Returns
        -------
        numpy.ndarray[np.float64[m,m]]
            Returns an ndarray matrix with the probabilities of increasing/decreasing one individual adopting
            a given strategy in the current population state.
            All possible new state transition probabilities are returned.

            e.g., given the population state (34, 33, 33) which represents a population of size 100 with 3
            strategies - obviously the population state may be represented with a 2D tuple (34, 33) - then
            there are 6 possible changes to the population: (35, 32, 33), (35, 33, 32), (33, 34, 33),
            (33, 33, 34), (34, 34, 32), (34, 32, 34).
        """
        transitions = np.zeros(shape=(2 * self.nb_strategies, 2 * self.nb_strategies))
        for i in range(self.nb_strategies):
            # If count of strategy < Z we calculate the probability of increasing
            if population_state[i] < self.Z:
                # calculate p_increase
                pass
            if population_state[i] > 0:
                # calculate p_decrease
                pass
        return transitions

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
        args : Optional[list[
            other arguments. Can be used to pass extra arguments to functions contained
            in the payoff matrix.

        Returns
        -------
        float
            The gradient of selection.
        """
        return ((self.Z - k) / float(self.Z)) * (k / float(self.Z - 1)) * np.tanh(
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
            Matrix indicating the likelihood of change in the population given an starting point.
        """
        probability_selecting_strategy_first = population_state / self.Z
        probability_selecting_strategy_second = population_state / (self.Z - 1)
        probabilities = np.outer(probability_selecting_strategy_first, probability_selecting_strategy_second)
        fitness = np.asarray([[self.full_fitness(i, j, population_state) for i in
                               range(len(population_state))] for j in range(len(population_state))])
        return (probabilities * np.tanh((beta / 2) * fitness)).sum(axis=0) * (1-self.mu) + self.mu

    def fixation_probability(self, invader: int, resident: int, beta: float, *args: Optional[list]) -> float:
        """
        Function for calculating the fixation_probability probability of the invader
        in a population of residents.

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
        egttools.numerical.PairwiseMoran
        """
        phi = 0.
        prod = 1.
        for i in range(1, self.Z):
            p_plus, p_minus = self.prob_increase_decrease(i, invader, resident, beta, *args)
            prod *= p_minus / p_plus
            phi += prod
            # We can approximate by zero if phi is too big
            if phi > 1e7:
                return 0.0

        return 1.0 / (1.0 + phi)

    def calculate_full_transition_matrix(self, beta: float, *args: Optional[list]) -> scipy.sparse.lil_matrix:
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
        scipy.sparse.lil_matrix
            The full transition matrix between the two strategies in sparse format.
        """
        transitions = lil_matrix((self.Z + 1, self.Z + 1), dtype=np.float64)
        # Case of 0:
        transitions[0, 1] = self.mu
        transitions[0, 0] = 1. - self.mu

        # Case of Z:
        transitions[self.Z, self.Z - 1] = self.mu
        transitions[self.Z, self.Z] = 1. - self.mu

        # Rest of transitions
        for i in range(1, self.Z):
            p_plus, p_minus = self.prob_increase_decrease_with_mutation(i, 0, 1, beta, *args)
            transitions[i, i + 1] = p_plus
            transitions[i, i - 1] = p_minus
            transitions[i, i] = 1. - (p_plus + p_minus)

        return transitions.transpose()

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
            the matrix of fixation probabilities as second element.
        """
        transitions = np.zeros((self.nb_strategies, self.nb_strategies))
        fixprobs = np.zeros((self.nb_strategies, self.nb_strategies))

        for first in range(self.nb_strategies):
            transitions[first, first] = 1.
            for second in range(self.nb_strategies):
                if second != first:
                    fp = self.fixation_probability(second, first, beta, *args)
                    fixprobs[first, second] = (fp * self.Z)
                    tmp = fp / float(self.nb_strategies - 1)
                    transitions[first, second] = tmp
                    transitions[first, first] = transitions[first, first] - tmp

        return np.nan_to_num(transitions.transpose(), copy=False), np.nan_to_num(fixprobs, copy=False)

    def calculate_stationary_distribution(self, beta: float, *args: Optional[list]) -> np.ndarray:
        """
        Calculates the stationary distribution of the monomorphic states is mu = 0 (SML).
        Otherwise, it calculates the stationary distribution including all possible population states.

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
        if self.mu == 0:
            t, f = self.transition_and_fixation_matrix(beta, *args)
        else:
            t = np.nan_to_num(self.calculate_full_transition_matrix(beta, *args).toarray())

        # calculate stationary distributions using eigenvalues and eigenvectors
        w, v = np.linalg.eig(t)
        j_stationary = np.argmin(abs(w - 1.0))  # look for the element closest to 1 in the list of eigenvalues
        p_stationary = abs(v[:, j_stationary].real)  # the, is essential to access the matrix by column
        p_stationary /= p_stationary.sum()  # normalize

        return p_stationary

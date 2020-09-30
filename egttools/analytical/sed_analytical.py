"""
Copyright (c) 2019 Elias Fernandez

This python module contains the necessary functions
to calculate analytically the evolutionary dynamics in Infinite and Finite
populations on 2-player games.
"""

import numpy as np
from scipy.sparse import lil_matrix
from scipy.stats import hypergeom


def replicator_equation(x, payoffs):
    """
    Produces the discrete time derivative of the replicator dynamics

    :param x: array with len equal to the number of strategies
    :param payoffs: payoff matrix
    :return time derivative of x
    """
    ax = np.dot(payoffs, x)
    return x * (ax - np.dot(x, ax))


class StochDynamics:
    """An Evolutionary Stochastic Dynamics class.

    Defines a class that contains methods to compute the stationary distribution for
    the limit of small mutation (only the monomorphic states) and the full transition matrix.

    Parameters
    ----------
    nb_strategies : int
                number of strategies in the population
    pop_size : int
            population size
    payoffs : ndarray
            payoff matrix indicating the payoff of each strategy (rows) against each other (columns)
    mu : float
        mutation probability
    """

    def __init__(self, nb_strategies, payoffs, pop_size, group_size=2, mu=0):
        self.nb_strategies = nb_strategies
        self.payoffs = payoffs
        self.Z = pop_size
        self.N = group_size
        self.mu = mu
        if group_size > 2:  # pairwise game
            self.fitness = self.fitness_group
        else:  # group game
            self.fitness = self.fitness_pair

    def fitness_pair(self, x, i, j, *args):
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

        Returns
        -------
            int
            the fitness difference among the strategies
        """
        fitness_i = ((x - 1) * self.payoffs[i, i] +
                     (self.Z - x) * self.payoffs[i, j]) / (self.Z - 1)
        fitness_j = ((self.Z - x - 1) * self.payoffs[j, j] +
                     x * self.payoffs[j, i]) / (self.Z - 1)
        return fitness_i - fitness_j

    def fitness_group(self, x, i, j, *args):
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
        args : List
            Other Parameters. This can be used to pass extra parameters to functions
            stored in the payoff matrix

        Returns
        -------
            int
            Returns the difference in fitness between strategy i and j
        """
        k_array = np.arange(0, self.N, dtype=np.int32)
        i_pmf = hypergeom(self.Z-1, x-1, self.N-1).pmf(k_array)
        j_pmf = hypergeom(self.Z-1, x, self.N-1).pmf(k_array)

        fitness_i, fitness_j = 0, 0
        for k in k_array:
            fitness_i += self.payoffs[i, j](k + 1, self.N, *args)*i_pmf[k]
            fitness_j += self.payoffs[j, i](self.N - k, self.N, *args)*j_pmf[k]

        return fitness_i - fitness_j

    @staticmethod
    def fermi(beta, fitness_diff):
        """
        The fermi function determines the probability that the first type imitates the second.

        :param beta: intensity of selection
        :param fitness_diff: f_a - f_b
        :return: the imitation probability
        :rtype: float
        """
        return np.clip(1. / (1. + np.exp(beta * fitness_diff, dtype=np.float64)), 0., 1.)

    def prob_increase_decrease(self, k, invader, resident, beta, *args):
        """
        This function calculates for a given number of invaders the probability
        that the number increases or decreases with one.

        :param k : number of invaders in the population
        :param invader : index of the invading strategy
        :param resident : index of the resitend strategy
        :param beta : intensity of selection
        :param args : other arguments. Can be used to pass extra arguments to functions contained
                      in the payoff matrix.
        :return tuple(probability of increasing the number of invaders, probability of decreasing)
        :rtype: tuple[float, float]
        """
        fitness_diff = self.fitness(k, invader, resident, *args)
        increase = (((self.Z - k) / float(self.Z)) * (k / float(self.Z - 1))) * StochDynamics.fermi(-beta, fitness_diff)
        decrease = ((k / float(self.Z)) * ((self.Z - k) / float(self.Z - 1))) * StochDynamics.fermi(beta, fitness_diff)
        return increase, decrease

    def prob_increase_decrease_with_mutation(self, k, invader, resident, beta, *args):
        """
        This function calculates for a given number of invaders the probability
        that the number increases or decreases with taking into account a mutation rate.

        :param k: number of invaders in the population
        :param invader: index of the invading strategy
        :param resident: index of the resitend strategy
        :param beta: intensity of selection
        :return tuple[probability of increasing the number of invaders, probability of decreasing]
        :rtype: tuple[float, float]
        """
        p_plus, p_less = self.prob_increase_decrease(k, invader, resident, beta, *args)
        p_plus = ((1 - self.mu) * p_plus) + (self.mu * ((self.Z - k) / self.Z))
        p_less = ((1 - self.mu) * p_less) + (self.mu * (k / self.Z))
        return p_plus, p_less

    def gradient_selection(self, k, invader, resident, beta, *args):
        """
        Calculates the gradient of selection given an invader and a resident strategy.

        :param k: number of invaders in the population
        :param invader: index of the invading strategy
        :param resident: index of the resident strategy
        :param beta: intensity of selection
        :return: gradient of selection
        :rtype: float
        """
        return ((self.Z - k) / float(self.Z)) * (k / float(self.Z)) * np.tanh(
            (beta / 2) * self.fitness(k, invader, resident, *args))

    def fixation_probability(self, invader, resident, beta, *args):
        """
        function for calculating the fixation_probability probability of the invader
        in a population of residents.

        The fixation probability is derived analytically:

        $\phi = $

        :param invader: index of the invading strategy
        :param resident: index of the resident strategy
        :param beta: intensity of selection
        :return: fixation_probability probability
        :rtype: float
        """
        phi = 0.
        prod = 1.
        for i in range(1, self.Z):
            p_plus, p_minus = self.prob_increase_decrease(i, invader, resident, beta, *args)
            prod *= p_minus / p_plus
            phi += prod

        return 1.0 / (1.0 + phi)

    def calculate_full_transition_matrix(self, beta, *args):
        """
        Returns the full transition matrix in sparse representation

        :param beta: intensity of selection
        :return full transition matrix between the two strategies
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

    def transition_and_fixation_matrix(self, beta, *args):
        """
        Calculates the transition matrix (only for the monomorphic states)
        and the fixation_probability probabilities

        :param beta: intensity of selection
        :return tuple(transition matrix, matrix containing the fixation_probability probabilities)
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

        return [np.nan_to_num(transitions.transpose(), copy=False), np.nan_to_num(fixprobs, copy=False)]

    def calculate_stationary_distribution(self, beta, *args):
        """
        Calculates the stationary distribution of the monomorphic states is mu = 0 (SML).
        Otherwise, it calculates the stationary distribution including all possible population states.

        :param beta: intensity of selection
        :return stationary distribution
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

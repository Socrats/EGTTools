import numpy as np
from scipy.sparse import csr_matrix, lil_matrix
from ..games import AbstractGame
from .. import calculate_nb_states
from . import PairwiseComparison

from typing import List, Tuple, Callable


class PairwiseComparison2D:

    def __init__(self, population_size: int, nb_strategies: int, games: List[AbstractGame],
                 game_transitions: csr_matrix, lda: float, delta: float = None,
                 population_transitions: Callable[[np.ndarray, AbstractGame], float] = None):
        """
        Implements a 2 dimensional Markov Chain model.

        This class implements a series of methods that can be used to study the coupled dynamics of behavior and
        environmental change as a 2-dimensional Markov Chain. Behavior change happens through social learning
        by using a Moran process with a pairwise-comparison rule.

        The class needs to receive a list of games, and a Sparse matrix containing the transition probabilities
        between games. Also, lda` controls the probability that a population event will occur (an environment update
        event occurs with probability `1-lda`). It is also a scaling factor to make sure that all transition
        probabilities sum to 1. In effect, it will serve as a timescale factor.

        `delta` determines the influence of the population in the game transitions. If `delta` is 0, the game
        transitions are solely determined by the environment stochasticity. If `delta` is 1, the game transitions
        are solely determined by the population.

        Parameters
        ----------
        population_size
        games
        game_transitions
        lda
        delta
        population_condition
        """
        self.population_size = population_size
        self.nb_strategies = nb_strategies
        self.games = games
        self.game_transitions = game_transitions
        self.lda = lda
        self.not_lda = 1 - lda
        self.delta = delta
        if self.delta is not None:
            self.not_delta = 1 - delta
            self.population_transitions = population_transitions

        self.nb_population_states = calculate_nb_states(population_size, nb_strategies)
        self.nb_game_states = len(games)
        self.nb_states = self.nb_population_states * self.nb_game_states

    def calculate_state_index(self, population_state_index: int, game_state_index: int) -> int:
        return population_state_index * self.nb_game_states + game_state_index

    def get_population_and_game_index_from_state_index(self, state_index: int) -> Tuple[int, int]:
        game_state_index = state_index % self.nb_game_states
        population_state_index = state_index // self.nb_game_states
        return population_state_index, game_state_index

    def calculate_transition_matrix(self, beta: float, mu: float) -> csr_matrix:
        transition_matrix = lil_matrix((self.nb_states, self.nb_states), dtype=np.float64)

        # we can actually fill all the population transition matrices for each game by calling
        # the "calculate_transition_matrix" method in Pairwise comparison

        # First Calculate all population transitions for each game
        for game_index, game in enumerate(self.games):
            evolver = PairwiseComparison(self.population_size, game)
            transitions = evolver.calculate_transition_matrix(beta, mu)
            for row, col in zip(*transitions.nonzero()):
                state_index_out = self.calculate_state_index(row, game_index)
                state_index_in = self.calculate_state_index(col, game_index)

                transition_matrix[state_index_out, state_index_in] = transitions[row, col] * self.lda

        # Then all game transitions for each population state (transitions are from row to column)
        for population_state_index in range(self.nb_population_states):
            for row, col in zip(*self.game_transitions.nonzero()):
                state_index_out = self.calculate_state_index(population_state_index, row)
                state_index_in = self.calculate_state_index(population_state_index, col)

                # The sum is very important to account for the probability of remaining in the current state
                transition_matrix[state_index_out, state_index_in] += self.game_transitions[row, col] * self.not_lda

        return transition_matrix.tocsr()

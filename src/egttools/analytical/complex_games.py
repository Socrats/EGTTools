import numpy as np
import numpy.typing as npt
from scipy.sparse import csr_matrix, lil_matrix
from ..games import AbstractGame
from .. import sample_simplex, calculate_state, calculate_nb_states
from .utils import fermi_update
from . import PairwiseComparison

from typing import List, Tuple


class PairwiseComparison2D:

    def __init__(self, population_size: int, nb_strategies: int, games: List[AbstractGame],
                 game_transitions: csr_matrix, kappa: float,
                 lda: float):
        """
        Implements a 2 dimensional Markov Chain model.

        This class implements a series of methods that can be used to study the coupled dynamics of behavior and
        environmental change as a 2-dimensional Markov Chain. Behavior change happens through social learning
        by using a Moran process with a pairwise-comparison rule.

        The class needs to receive a list of games, and a Sparse matrix containing the transition probabilities
        between games. Also, 2 extra parameters, `kappa` and `lda` which control the probability at which each
        event (behavior or environmental update) may happen. `kappa` represents the probability of a population
        update happening, and `lda` represents the probability

        Parameters
        ----------
        population_size
        games
        game_transitions
        kappa
        lda
        """
        self.population_size = population_size
        self.nb_strategies = nb_strategies
        self.games = games
        self.game_transitions = game_transitions
        self.kappa = kappa
        self.lda = lda

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


        # Then all game transitions for each population state
        # Then re-scaling with the time factors (kappa and lda)
        # Finally, we might try to think whether it's possible to compute
        # analytically the joint probability of a population and environmental change.

        # for i in range(self.nb_states):
        #     population_state_index_out, game_state_index_out = self.get_population_and_game_index_from_state_index(i)
        #
        #     # Calculate fitness of all strategies here
        #
        #     for j in range(self.nb_states):
        #         population_state_index_in, game_state_index_in = self.get_population_and_game_index_from_state_index(j)
        #
        #         #
        #
        #         # The transition probabilities here, are defined by the previous fitness
        #         # we need to know what is the

        return transition_matrix.tocsr()

    def transition_probability(self, population_state: np.ndarray, game_state: int):
        pass

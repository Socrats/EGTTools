from egttools.distributions import multinomial_pmf
from egttools import sample_simplex, calculate_nb_states
from scipy.stats import multinomial
import numpy as np


def test_multinomial_two_strategies():
    group_size = 10
    nb_strategies = 2
    nb_group_combinations = calculate_nb_states(group_size, nb_strategies)

    nb_points = 101
    strategy_i = np.linspace(0, 1, num=nb_points, dtype=np.float64)
    states = [np.array([i, 1 - i]) for i in strategy_i]

    for state in states:
        for group_index in range(nb_group_combinations):
            group = sample_simplex(group_index, group_size, nb_strategies)
            prob_scipy = multinomial.pmf(group, group_size, state)
            prob_egttools = multinomial_pmf(group, group_size, state)

            assert np.isclose(prob_egttools, prob_scipy)

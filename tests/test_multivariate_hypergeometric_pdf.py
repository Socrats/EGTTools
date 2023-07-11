from egttools.distributions import multivariate_hypergeometric_pdf
from egttools import sample_simplex, calculate_nb_states, USES_BOOST
from scipy.stats import multivariate_hypergeom
import numpy as np
import pytest


@pytest.mark.skipif(USES_BOOST is False, reason="requires egttools to be compiled with Boost C++")
def test_multivariate_hypergeometric_distribution_two_strategies():
    population_size = 500
    group_size = 10
    nb_strategies = 2
    nb_states = calculate_nb_states(population_size, nb_strategies)
    nb_group_combinations = calculate_nb_states(group_size, nb_strategies)

    for state_index in range(nb_states):
        state = sample_simplex(state_index, population_size, nb_strategies)
        for group_index in range(nb_group_combinations):
            group = sample_simplex(group_index, group_size, nb_strategies)
            prob_scipy = multivariate_hypergeom.pmf(group, state, group_size)
            prob_egttools = multivariate_hypergeometric_pdf(population_size,
                                                            nb_strategies,
                                                            group_size,
                                                            group,
                                                            state)

            assert np.isclose(prob_egttools, prob_scipy)


@pytest.mark.skipif(USES_BOOST is False, reason="requires egttools to be compiled with Boost C++")
def test_multivariate_hypergeometric_distribution_three_strategies():
    population_size = 50
    group_size = 4
    nb_strategies = 3
    nb_states = calculate_nb_states(population_size, nb_strategies)
    nb_group_combinations = calculate_nb_states(group_size, nb_strategies)

    for state_index in range(nb_states):
        state = sample_simplex(state_index, population_size, nb_strategies)
        for group_index in range(nb_group_combinations):
            group = sample_simplex(group_index, group_size, nb_strategies)
            prob_scipy = multivariate_hypergeom.pmf(group, state, group_size)
            prob_egttools = multivariate_hypergeometric_pdf(population_size,
                                                            nb_strategies,
                                                            group_size,
                                                            group,
                                                            state)

            assert np.isclose(prob_egttools, prob_scipy)

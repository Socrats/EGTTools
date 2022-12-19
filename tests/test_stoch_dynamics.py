from pytest import fixture
import numpy as np
from egttools.analytical import StochDynamics
from egttools import calculate_nb_states


@fixture
def test_analytical_fitness_calculation_stoch_dynamics():
    payoff_matrix = np.array([
        [-0.5, 2],
        [0, 0]
    ])

    evolver = StochDynamics(nb_strategies=2, payoffs=payoff_matrix, pop_size=50, group_size=2, mu=0)

    return evolver


def test_update_population_size(test_analytical_fitness_calculation_stoch_dynamics):
    evolver = test_analytical_fitness_calculation_stoch_dynamics

    evolver.update_population_size(100)

    assert evolver.pop_size == 100
    assert evolver.nb_states_population == calculate_nb_states(100, 2)


def test_update_group_size(test_analytical_fitness_calculation_stoch_dynamics):
    evolver = test_analytical_fitness_calculation_stoch_dynamics

    evolver.update_group_size(4)

    assert evolver.group_size == 4
    assert evolver.nb_group_combinations == calculate_nb_states(4, 2)
    assert evolver.fitness == evolver.fitness_group
    assert evolver.full_fitness == evolver.full_fitness_difference_group


def test_update_payoffs(test_analytical_fitness_calculation_stoch_dynamics):
    evolver = test_analytical_fitness_calculation_stoch_dynamics

    payoff_matrix = np.array([
        [-0.5, 2],
        [0, 5]
    ])

    evolver.update_payoffs(payoff_matrix)

    assert np.allclose(payoff_matrix, evolver.payoffs)

    payoff_matrix = np.array([
        [-0.5, 2, 3],
        [0, 5, 3],
        [0, 5, 3]
    ])

    try:
        evolver.update_payoffs(payoff_matrix)
    except ValueError:
        assert True
    else:
        assert False

    evolver.update_payoffs(payoff_matrix, 3)

    assert np.allclose(payoff_matrix, evolver.payoffs)


@fixture
def get_fitness(test_analytical_fitness_calculation_stoch_dynamics):
    evolver = test_analytical_fitness_calculation_stoch_dynamics

    fitness_i = ((evolver.pop_size - 1) / (evolver.pop_size - 1)) * evolver.payoffs[0, 1]

    fitness_j = ((1 / (evolver.pop_size - 1)) * evolver.payoffs[1, 1]) + (
            ((evolver.pop_size - 2) / (evolver.pop_size - 1)) * evolver.payoffs[1, 0])

    return fitness_i - fitness_j


def test_fitness_pair(test_analytical_fitness_calculation_stoch_dynamics, get_fitness):
    evolver = test_analytical_fitness_calculation_stoch_dynamics
    fitness_diff = get_fitness

    fitness = evolver.fitness_pair(1, 0, 1)

    assert np.isclose(fitness, fitness_diff)


def test_prob_increase_decrease(test_analytical_fitness_calculation_stoch_dynamics, get_fitness):
    evolver = test_analytical_fitness_calculation_stoch_dynamics
    fitness_diff = get_fitness

    prob_increase, prob_decrease = evolver.prob_increase_decrease(1, 0, 1, 1)

    increase = (((evolver.pop_size - 1) / evolver.pop_size) *
                (1 / (evolver.pop_size - 1))) * StochDynamics.fermi(-1, fitness_diff)

    decrease = ((1 / evolver.pop_size) *
                ((evolver.pop_size - 1) / (evolver.pop_size - 1))) * StochDynamics.fermi(1, fitness_diff)

    assert np.isclose(increase, prob_increase)
    assert np.isclose(decrease, prob_decrease)

    prob_increase, prob_decrease = evolver.prob_increase_decrease(0, 0, 1, 1)

    assert np.isclose(prob_increase, 0)
    assert np.isclose(prob_decrease, 0)

    prob_increase, prob_decrease = evolver.prob_increase_decrease(evolver.pop_size, 0, 1, 1)

    assert np.isclose(prob_increase, 0)
    assert np.isclose(prob_decrease, 0)


def test_prob_increase_decrease_with_mutation(test_analytical_fitness_calculation_stoch_dynamics, get_fitness):
    evolver = test_analytical_fitness_calculation_stoch_dynamics
    fitness_diff = get_fitness

    mu = 0.01
    evolver.mu = mu
    prob_increase, prob_decrease = evolver.prob_increase_decrease_with_mutation(1, 0, 1, 1)

    increase = (((evolver.pop_size - 1) / evolver.pop_size) *
                (1 / (evolver.pop_size - 1))) * StochDynamics.fermi(-1, fitness_diff)
    increase = ((1 - mu) * increase) + (mu * ((evolver.pop_size - 1) / evolver.pop_size))

    decrease = ((1 / evolver.pop_size) *
                ((evolver.pop_size - 1) / (evolver.pop_size - 1))) * StochDynamics.fermi(1, fitness_diff)
    decrease = ((1 - mu) * decrease) + (mu * (1 / evolver.pop_size))

    assert np.isclose(increase, prob_increase)
    assert np.isclose(decrease, prob_decrease)

    prob_increase, prob_decrease = evolver.prob_increase_decrease_with_mutation(0, 0, 1, 1)

    assert np.isclose(prob_increase, mu)
    assert np.isclose(prob_decrease, 0)

    prob_increase, prob_decrease = evolver.prob_increase_decrease_with_mutation(evolver.pop_size, 0, 1, 1)

    assert np.isclose(prob_increase, 0)
    assert np.isclose(prob_decrease, mu)

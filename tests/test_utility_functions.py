import egttools as egt
from scipy.special import comb


def calculate_nb_states(pop_size, nb_strategies):
    return comb(pop_size + nb_strategies - 1, pop_size, exact=True)


def test_calculate_nb_states():
    assert egt.calculate_nb_states(10, 3) == 66
    assert egt.calculate_nb_states(10, 10) == 92378
    assert egt.calculate_nb_states(100, 11) == 46897636623981


def test_calculate_state():
    assert egt.calculate_state(2, [0, 1, 0, 1]) == 6
    assert egt.calculate_state(4, [0, 3, 0, 1]) == 22
    assert egt.calculate_state(4, [4, 0, 0, 0]) == 0
    assert egt.calculate_state(4, [0, 0, 0, 4]) == egt.calculate_nb_states(4, 4) - 1


def test_sample_simplex():
    pass


def test_sample_unit_simplex():
    pass


def test_calculate_strategies_distribution():
    pass


def test_random():
    pass

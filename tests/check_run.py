import numpy as np
from egttools.games import Matrix2PlayerGameHolder
from egttools.numerical import PairwiseComparisonNumerical

def payoffs() -> np.ndarray:
    p_o = 0.2
    p_c = 0.5
    e_c = 1
    e_o = 2
    b_c = 15
    b_cf = 2
    b_o = 10
    b_p = 10
    c_cf = 5
    c_c = 50
    c_o = 3
    c_po = 1

    return np.array([
        [0, p_o * b_o + (1 - p_o) * (-c_o) - e_o, c_po],  # O , C, P
        [p_o * (-c_c), (b_cf - c_cf) / 2, p_c * b_c - e_c],  # C
        [-c_po, -p_c * c_c, b_p]  # P
    ])

if __name__ == "__main__":
    pop_size = 100
    nb_strategies = 3
    cache_size = 1000

    beta = 1
    mu = 1e-3
    init_state = np.array([40, 10, 50], dtype=np.uint64)

    nb_generations = 1000

    game = Matrix2PlayerGameHolder(nb_strategies, payoffs())

    pc = PairwiseComparisonNumerical(pop_size, game, cache_size)

    result = pc.run_with_mutation(nb_generations, beta, mu, init_state)

    assert result.shape == (nb_generations + 1, game.nb_strategies())
    assert np.all(np.sum(result, axis=1) == pop_size)


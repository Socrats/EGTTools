"""
Ground-truth correctness test for PairwiseComparison.calculate_transition_matrix.

Strategy
--------
A pure-Python reference implementation of the Markov transition matrix is derived
directly from the mathematical definition of the pairwise-comparison rule with
mutation, independently of the C++ code.  The C++ output is then compared
entry-by-entry against this reference for a variety of games, population sizes,
strategy counts, and (beta, mu) parameter combinations.

Any discrepancy larger than floating-point rounding (atol=1e-14) indicates a bug
in the C++ implementation.

Tested games
------------
- Matrix2PlayerGameHolder   (2-strategy, 3-strategy)
- NPlayerGameHolder         (3-player game, 3 strategies)
- CRD-style N-player game   (4-strategy, group_size=4)
- Python AbstractNPlayerStateGame (PDGame probe)

If all tests pass the transition matrix is numerically identical to the reference.
"""
import numpy as np
import pytest
import egttools as egt
from egttools import sample_simplex, calculate_state, calculate_nb_states
from egttools.games import AbstractNPlayerStateGame


# ---------------------------------------------------------------------------
# Reference fermi function (matches C++ fermi(beta, a, b) = 1/(1+exp(beta*(a-b))))
# ---------------------------------------------------------------------------

def fermi_ref(beta: float, a: float, b: float) -> float:
    """Fermi function identical to the C++ implementation."""
    return 1.0 / (1.0 + np.exp(beta * (a - b)))


# ---------------------------------------------------------------------------
# Reference transition matrix  (pure Python, no C++ path)
# ---------------------------------------------------------------------------

def reference_transition_matrix(game, pop_size: int, beta: float, mu: float):
    """
    Build the full Markov transition matrix from scratch in Python.

    This reimplements the exact formula from PairwiseComparison.cpp without
    reusing any of the C++ logic, so it serves as an independent ground truth.

    Transition probability from state x to x + e_i - e_j
    (strategy i gains one individual, strategy j loses one):

    Monomorphic states (only one strategy present, count = N):
        T(x → x+ei-ej) = mu_eff    for all j ≠ i

    Interior states (multiple strategies present):
        For absent strategy i (x_i = 0), gaining from present j:
            T(x → x+ei-ej) = (x_j / N) * mu_eff

        For present strategy i (x_i > 0), gaining from present j ≠ i:
            T(x → x+ei-ej) = (x_j / N) * [(1-mu)*(x_i/(N-1))*fermi(beta,fj,fi) + mu_eff]

    Fitness fi is game.calculate_fitness(i, N, x - e_i)  (focal player excluded).
    """
    k = int(game.nb_strategies())
    N = pop_size
    S = int(calculate_nb_states(N, k))

    mu_eff = mu / (k - 1) if k > 2 else mu
    one_minus_mu = 1.0 - mu
    inv_N = 1.0 / N
    inv_Nm1 = 1.0 / (N - 1)

    rows_idx, cols_idx, vals = [], [], []

    for row in range(S):
        state = sample_simplex(row, N, k).astype(np.int64)
        present = [i for i in range(k) if state[i] > 0]

        total_offdiag = 0.0

        if len(present) == 1:
            mono = present[0]
            for i in range(k):
                if i == mono:
                    continue
                next_state = state.copy()
                next_state[mono] -= 1
                next_state[i] += 1
                col = int(calculate_state(N, next_state))

                prob = mu_eff
                rows_idx.append(row)
                cols_idx.append(col)
                vals.append(prob)
                total_offdiag += prob

        else:
            # Pre-compute fitness for each present strategy (focal player excluded)
            fitness = {}
            for i in present:
                tmp = state.copy()
                tmp[i] -= 1
                fitness[i] = game.calculate_fitness(i, N, tmp)

            for i in range(k):
                if state[i] == 0:
                    # Absent strategy i: only mutation from each present j
                    for j in present:
                        next_state = state.copy()
                        next_state[i] += 1
                        next_state[j] -= 1
                        col = int(calculate_state(N, next_state))

                        prob = float(state[j]) * inv_N * mu_eff
                        if prob > 0.0:
                            rows_idx.append(row)
                            cols_idx.append(col)
                            vals.append(prob)
                            total_offdiag += prob
                else:
                    # Present strategy i: selection + mutation
                    fi = fitness[i]
                    sel_prefactor = one_minus_mu * float(state[i]) * inv_Nm1

                    for j in present:
                        if j == i:
                            continue
                        fj = fitness[j]
                        next_state = state.copy()
                        next_state[i] += 1
                        next_state[j] -= 1
                        col = int(calculate_state(N, next_state))

                        sel_prob = sel_prefactor * fermi_ref(beta, fj, fi)
                        prob = float(state[j]) * inv_N * (sel_prob + mu_eff)
                        if prob > 0.0:
                            rows_idx.append(row)
                            cols_idx.append(col)
                            vals.append(prob)
                            total_offdiag += prob

        diag = max(0.0, 1.0 - total_offdiag)
        rows_idx.append(row)
        cols_idx.append(row)
        vals.append(diag)

    from scipy.sparse import csr_matrix
    return csr_matrix(
        (np.array(vals), (np.array(rows_idx), np.array(cols_idx))),
        shape=(S, S),
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def compare_matrices(T_cpp, T_ref, label: str, atol: float = 1e-13):
    """Dense comparison with a clear failure message."""
    A = T_cpp.toarray()
    B = T_ref.toarray()

    max_diff = np.max(np.abs(A - B))
    mismatches = np.argwhere(np.abs(A - B) > atol)

    assert mismatches.size == 0, (
        f"{label}: C++ and reference matrices differ.\n"
        f"  Max absolute difference: {max_diff:.3e}  (tolerance {atol:.0e})\n"
        f"  First mismatched (row,col) pairs: {mismatches[:5].tolist()}\n"
        f"  Example C++ values:  {A[mismatches[0,0], mismatches[:5,1]]}\n"
        f"  Example ref values:  {B[mismatches[0,0], mismatches[:5,1]]}"
    )


def make_evolver(game, pop_size):
    return egt.analytical.PairwiseComparison(pop_size, game)


# ---------------------------------------------------------------------------
# Game factories
# ---------------------------------------------------------------------------

def hawkdove_payoff_matrix(v=2.0, c=3.0):
    """Hawk-Dove: 2×2 payoff matrix."""
    return np.array([
        [(v - c) / 2.0, v],
        [0.0,           v / 2.0],
    ])


def rps_payoff_matrix():
    """Rock-Paper-Scissors: 3×3 payoff matrix."""
    return np.array([
        [0.0,  1.0, -1.0],
        [-1.0, 0.0,  1.0],
        [1.0, -1.0,  0.0],
    ])


def coordination_payoff_matrix(a=2.0, b=1.0):
    """2×2 coordination game."""
    return np.array([
        [a,   0.0],
        [0.0, b],
    ])


def make_4strategy_nplayer_game(group_size=4):
    """
    4-strategy N-player game via NPlayerGameHolder.
    Returns a (nb_gc × 4) payoff table filled with synthetic payoffs.
    """
    nb_gc = int(calculate_nb_states(group_size, 4))
    rng = np.random.default_rng(42)
    # MatrixNPlayerGameHolder expects shape (nb_strategies, nb_gc)
    payoffs = rng.standard_normal((4, nb_gc))
    return egt.games.MatrixNPlayerGameHolder(4, group_size, payoffs)


# ---------------------------------------------------------------------------
# Minimal Python game (probe)
# ---------------------------------------------------------------------------

class PDGameRef(AbstractNPlayerStateGame):
    """2×2 PD via AbstractNPlayerStateGame — included to stress the Python path."""

    B = np.array([[2.0, -1.0], [3.0, 0.0]])  # b=3, c=1

    def __init__(self):
        super().__init__(2, 2)
        nb_gc = self.nb_group_configurations()
        self._gc = [sample_simplex(j, 2, 2) for j in range(nb_gc)]

    def get_payoffs_for_player(self, player_type, state_index, state):
        payoffs = np.zeros(len(self._gc))
        for j, gc in enumerate(self._gc):
            if gc[player_type] == 0:
                continue
            for opp in range(2):
                nb_opp = int(gc[opp]) - (1 if opp == player_type else 0)
                if nb_opp > 0:
                    payoffs[j] = self.B[player_type, opp]
                    break
        return payoffs

    def play(self, group_composition, game_payoffs):
        pass

    def calculate_payoffs(self):
        return np.zeros((self.nb_group_configurations(), 2))


# ===========================================================================
# Test class: C++ vs Python reference
# ===========================================================================

_BETA_MU_PAIRS = [
    (0.0, 0.01),   # neutral drift
    (1.0, 0.01),   # moderate selection
    (5.0, 0.001),  # strong selection, low mutation
    (1.0, 0.5),    # high mutation
    (1.0, 1.0),    # pure mutation
]


class TestTransitionMatrixVsReference:
    """
    The C++ transition matrix must be numerically identical (to floating-point
    precision) to the independently implemented Python reference.

    Any failure here means the C++ code produces a different matrix from the
    mathematical definition of the pairwise comparison Markov chain.
    """

    @pytest.mark.parametrize("pop_size", [5, 10, 20])
    @pytest.mark.parametrize("beta, mu", _BETA_MU_PAIRS)
    def test_2strategy_hawkdove(self, pop_size, beta, mu):
        game = egt.games.Matrix2PlayerGameHolder(2, payoff_matrix=hawkdove_payoff_matrix())
        evolver = make_evolver(game, pop_size)
        T_cpp = evolver.calculate_transition_matrix(beta, mu)
        T_ref = reference_transition_matrix(game, pop_size, beta, mu)
        compare_matrices(T_cpp, T_ref,
                         f"HawkDove N={pop_size} beta={beta} mu={mu}")

    @pytest.mark.parametrize("pop_size", [5, 10, 20])
    @pytest.mark.parametrize("beta, mu", _BETA_MU_PAIRS)
    def test_2strategy_coordination(self, pop_size, beta, mu):
        game = egt.games.Matrix2PlayerGameHolder(2, payoff_matrix=coordination_payoff_matrix())
        evolver = make_evolver(game, pop_size)
        T_cpp = evolver.calculate_transition_matrix(beta, mu)
        T_ref = reference_transition_matrix(game, pop_size, beta, mu)
        compare_matrices(T_cpp, T_ref,
                         f"Coordination N={pop_size} beta={beta} mu={mu}")

    @pytest.mark.parametrize("pop_size", [5, 10, 15])
    @pytest.mark.parametrize("beta, mu", _BETA_MU_PAIRS)
    def test_3strategy_rps(self, pop_size, beta, mu):
        game = egt.games.Matrix2PlayerGameHolder(3, payoff_matrix=rps_payoff_matrix())
        evolver = make_evolver(game, pop_size)
        T_cpp = evolver.calculate_transition_matrix(beta, mu)
        T_ref = reference_transition_matrix(game, pop_size, beta, mu)
        compare_matrices(T_cpp, T_ref,
                         f"RPS N={pop_size} beta={beta} mu={mu}")

    @pytest.mark.parametrize("pop_size", [5, 8])
    @pytest.mark.parametrize("beta, mu", [(1.0, 0.01), (5.0, 0.001), (0.0, 0.05)])
    def test_4strategy_nplayer(self, pop_size, beta, mu):
        game = make_4strategy_nplayer_game(group_size=4)
        evolver = make_evolver(game, pop_size)
        T_cpp = evolver.calculate_transition_matrix(beta, mu)
        T_ref = reference_transition_matrix(game, pop_size, beta, mu)
        compare_matrices(T_cpp, T_ref,
                         f"4strat-Nplayer N={pop_size} beta={beta} mu={mu}")

    @pytest.mark.parametrize("pop_size", [5, 10])
    @pytest.mark.parametrize("beta, mu", [(1.0, 0.01), (5.0, 0.001)])
    def test_python_game_pd(self, pop_size, beta, mu):
        """Python-subclassed game (AbstractNPlayerStateGame) must match reference."""
        game = PDGameRef()
        evolver = make_evolver(game, pop_size)
        T_cpp = evolver.calculate_transition_matrix(beta, mu)
        T_ref = reference_transition_matrix(game, pop_size, beta, mu)
        compare_matrices(T_cpp, T_ref,
                         f"PDGameRef N={pop_size} beta={beta} mu={mu}")


# ===========================================================================
# Test class: structural properties
# ===========================================================================

class TestTransitionMatrixStructure:
    """Row-stochastic, non-negative, correct sparsity pattern."""

    @pytest.mark.parametrize("nb_strategies,pop_size", [
        (2, 20), (2, 50), (3, 10), (3, 20), (4, 8),
    ])
    def test_row_stochastic(self, nb_strategies, pop_size):
        if nb_strategies == 2:
            game = egt.games.Matrix2PlayerGameHolder(2, payoff_matrix=hawkdove_payoff_matrix())
        elif nb_strategies == 3:
            game = egt.games.Matrix2PlayerGameHolder(3, payoff_matrix=rps_payoff_matrix())
        else:
            game = make_4strategy_nplayer_game()
        evolver = make_evolver(game, pop_size)
        T = evolver.calculate_transition_matrix(1.0, 0.01)
        row_sums = np.array(T.sum(axis=1)).ravel()
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-10,
                                   err_msg=f"k={nb_strategies} N={pop_size}: row sums not 1")

    @pytest.mark.parametrize("nb_strategies,pop_size", [(2, 30), (3, 15), (4, 8)])
    def test_non_negative(self, nb_strategies, pop_size):
        if nb_strategies == 2:
            game = egt.games.Matrix2PlayerGameHolder(2, payoff_matrix=hawkdove_payoff_matrix())
        elif nb_strategies == 3:
            game = egt.games.Matrix2PlayerGameHolder(3, payoff_matrix=rps_payoff_matrix())
        else:
            game = make_4strategy_nplayer_game()
        evolver = make_evolver(game, pop_size)
        T = evolver.calculate_transition_matrix(1.0, 0.01)
        assert T.min() >= -1e-14, \
            f"k={nb_strategies} N={pop_size}: negative transition probability"

    def test_sparsity_matches_reference(self):
        """The non-zero sparsity pattern must match the reference exactly."""
        game = egt.games.Matrix2PlayerGameHolder(3, payoff_matrix=rps_payoff_matrix())
        pop_size = 10
        evolver = make_evolver(game, pop_size)
        T_cpp = evolver.calculate_transition_matrix(1.0, 0.01)
        T_ref = reference_transition_matrix(game, pop_size, 1.0, 0.01)

        # Compare sparsity patterns via boolean masks
        A = T_cpp.toarray()
        B = T_ref.toarray()
        nz_cpp = set(zip(*np.where(A != 0)))
        nz_ref = set(zip(*np.where(B != 0)))
        extra = nz_cpp - nz_ref
        missing = nz_ref - nz_cpp
        assert not extra and not missing, (
            f"Sparsity mismatch:\n"
            f"  Extra non-zeros in C++: {list(extra)[:5]}\n"
            f"  Missing non-zeros in C++: {list(missing)[:5]}"
        )

    @pytest.mark.parametrize("beta,mu", [
        (0.0, 0.01), (1.0, 0.01), (5.0, 0.001),
    ])
    def test_beta0_is_neutral_drift_2strategies(self, beta, mu):
        """At beta=0 selection disappears; transition probabilities depend only on counts."""
        game = egt.games.Matrix2PlayerGameHolder(2, payoff_matrix=hawkdove_payoff_matrix())
        pop_size = 10
        evolver = make_evolver(game, pop_size)
        T = evolver.calculate_transition_matrix(0.0, mu).toarray()
        T_ref = reference_transition_matrix(game, pop_size, 0.0, mu).toarray()
        np.testing.assert_allclose(T, T_ref, atol=1e-14,
                                   err_msg="beta=0 matrices differ")


# ===========================================================================
# Test class: parameter boundary cases
# ===========================================================================

class TestParameterBoundaries:
    """Edge cases: mu=0, mu=1, very small/large beta."""

    def test_pure_mutation_matrix(self):
        """With mu=1 the Fermi term vanishes; T depends only on counts."""
        game = egt.games.Matrix2PlayerGameHolder(2, payoff_matrix=hawkdove_payoff_matrix())
        pop_size = 8
        evolver = make_evolver(game, pop_size)
        T_cpp = evolver.calculate_transition_matrix(5.0, 1.0)
        T_ref = reference_transition_matrix(game, pop_size, 5.0, 1.0)
        compare_matrices(T_cpp, T_ref, "mu=1 pure mutation")

    def test_very_large_beta(self):
        """Strong selection: numerical stability."""
        game = egt.games.Matrix2PlayerGameHolder(2, payoff_matrix=hawkdove_payoff_matrix())
        pop_size = 8
        evolver = make_evolver(game, pop_size)
        T_cpp = evolver.calculate_transition_matrix(100.0, 0.01)
        T_ref = reference_transition_matrix(game, pop_size, 100.0, 0.01)
        compare_matrices(T_cpp, T_ref, "beta=100 strong selection", atol=1e-12)

    def test_small_population(self):
        """pop_size=2 is the minimum; verify all entries exactly."""
        game = egt.games.Matrix2PlayerGameHolder(2, payoff_matrix=hawkdove_payoff_matrix())
        evolver = make_evolver(game, 2)
        T_cpp = evolver.calculate_transition_matrix(1.0, 0.1)
        T_ref = reference_transition_matrix(game, 2, 1.0, 0.1)
        compare_matrices(T_cpp, T_ref, "N=2 minimum population")

    @pytest.mark.parametrize("pop_size", [5, 10, 20])
    def test_3strategy_all_beta_mu(self, pop_size):
        game = egt.games.Matrix2PlayerGameHolder(3, payoff_matrix=rps_payoff_matrix())
        evolver = make_evolver(game, pop_size)
        for beta, mu in _BETA_MU_PAIRS:
            T_cpp = evolver.calculate_transition_matrix(beta, mu)
            T_ref = reference_transition_matrix(game, pop_size, beta, mu)
            compare_matrices(T_cpp, T_ref,
                             f"RPS-3strat N={pop_size} beta={beta} mu={mu}")

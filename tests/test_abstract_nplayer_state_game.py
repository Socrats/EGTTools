"""
Tests for AbstractNPlayerStateGame — the C++ base class for state-dependent
N-player games that reduces Python callbacks from O(nb_group_configurations)
to exactly 1 per fitness evaluation.

Key correctness invariant being tested here:
  The `state_index` passed to `get_payoffs_for_player` must be the linear
  index of the FULL POPULATION state (including the focal player) in the
  population-simplex state space, i.e. the result of

      calculate_state(pop_size, full_population_state)

  NOT `calculate_state(group_size, full_population_state)`, which collapses
  almost all states to index 0 and silently corrupts payoff lookups for any
  population larger than the group.
"""
import numpy as np
import pytest
import egttools as egt
from egttools import sample_simplex
from egttools.games import AbstractNPlayerStateGame


# ---------------------------------------------------------------------------
# Minimal game fixture: 2-strategy Prisoner's Dilemma, group_size=2
# ---------------------------------------------------------------------------

class PDGame(AbstractNPlayerStateGame):
    """2×2 Prisoner's Dilemma used as a probe game.

    The primary purpose is to record every (state_index, state) pair that
    arrives in get_payoffs_for_player so the tests can assert correctness.
    """

    def __init__(self, pop_size, b=3.0, c=1.0):
        super().__init__(2, 2)   # nb_strategies=2, group_size=2
        self.pop_size = pop_size
        self.b = b
        self.c = c
        nb_gc = self.nb_group_configurations()
        self._group_configs = [sample_simplex(j, 2, 2) for j in range(nb_gc)]

        # Diagnostics logged by every get_payoffs_for_player call
        self.log_state_index = []
        self.log_state = []

    def get_payoffs_for_player(self, player_type, state_index, state):
        self.log_state_index.append(int(state_index))
        self.log_state.append(np.array(state, dtype=int).copy())

        # Payoff matrix B[i, j] = payoff to strategy i when playing against j.
        # B = [[b-c, -c], [b, 0]]  (standard 2-player PD)
        B = np.array([[self.b - self.c, -self.c],
                      [self.b,          0.0]])

        payoffs = np.zeros(len(self._group_configs))
        for j, gc in enumerate(self._group_configs):
            # gc is a group composition of group_size=2 including the focal player.
            # Skip configurations where the focal player is absent.
            if gc[player_type] == 0:
                continue
            # There is exactly 1 opponent (group_size=2).
            # The opponent's type is determined by which other strategy is present.
            for opp_type in range(2):
                nb_opp = int(gc[opp_type]) - (1 if opp_type == player_type else 0)
                if nb_opp > 0:
                    payoffs[j] = B[player_type, opp_type]
                    break
        return payoffs

    def play(self, group_composition, game_payoffs):
        pass

    def calculate_payoffs(self):
        return np.zeros((self.nb_group_configurations(), 2))


# ---------------------------------------------------------------------------
# 3-strategy Rock-Paper-Scissors-style game, group_size=2
# ---------------------------------------------------------------------------

class RPSGame(AbstractNPlayerStateGame):
    """3-strategy game — lets us verify multi-strategy state_index handling."""

    def __init__(self, pop_size):
        super().__init__(3, 2)    # nb_strategies=3, group_size=2
        self.pop_size = pop_size
        nb_gc = self.nb_group_configurations()
        self._group_configs = [sample_simplex(j, 2, 3) for j in range(nb_gc)]
        self.log_state_index = []

    def get_payoffs_for_player(self, player_type, state_index, state):
        self.log_state_index.append(int(state_index))
        payoffs = np.zeros(len(self._group_configs))
        return payoffs

    def play(self, group_composition, game_payoffs):
        pass

    def calculate_payoffs(self):
        return np.zeros((self.nb_group_configurations(), 3))


# ---------------------------------------------------------------------------
# Reference: pure C++ Matrix2PlayerGameHolder (same PD payoffs)
# ---------------------------------------------------------------------------

def pd_payoff_matrix(b=3.0, c=1.0):
    return np.array([
        [b - c, -c],
        [b,      0.0],
    ])


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def build_transition_matrix(pop_size, game, beta=1.0, mu=0.01):
    evolver = egt.analytical.PairwiseComparison(pop_size, game)
    T = evolver.calculate_transition_matrix(beta, mu)
    nb_states = evolver.nb_states()
    return T, nb_states


# ===========================================================================
# Tests
# ===========================================================================

class TestStateIndexCorrectness:
    """The state_index received by get_payoffs_for_player must lie within the
    population-state index range [0, nb_states_population)."""

    @pytest.mark.parametrize("pop_size", [2, 5, 10, 20])
    def test_state_index_in_population_range_2strategies(self, pop_size):
        game = PDGame(pop_size)
        T, nb_states = build_transition_matrix(pop_size, game)
        assert game.log_state_index, "get_payoffs_for_player was never called"
        bad = [idx for idx in game.log_state_index if idx < 0 or idx >= nb_states]
        assert not bad, (
            f"pop_size={pop_size}: received out-of-range state_index values: "
            f"{bad[:5]}, expected [0, {nb_states})"
        )

    @pytest.mark.parametrize("pop_size", [2, 5, 10])
    def test_state_index_in_population_range_3strategies(self, pop_size):
        game = RPSGame(pop_size)
        T, nb_states = build_transition_matrix(pop_size, game)
        assert game.log_state_index, "get_payoffs_for_player was never called"
        bad = [idx for idx in game.log_state_index if idx < 0 or idx >= nb_states]
        assert not bad, (
            f"pop_size={pop_size} k=3: out-of-range state_index: "
            f"{bad[:5]}, expected [0, {nb_states})"
        )

    def test_state_index_covers_all_interior_states(self):
        """Every interior population state must produce a distinct state_index
        so that state-dependent payoffs can differ across states."""
        pop_size = 10
        game = PDGame(pop_size)
        T, nb_states = build_transition_matrix(pop_size, game)

        # There are pop_size - 1 = 9 interior states (excluding monomorphic).
        # The distinct interior indices must be exactly {1, 2, ..., pop_size-1}.
        received = set(game.log_state_index)
        interior = set(range(1, pop_size))      # indices 1..9
        missing = interior - received
        assert not missing, (
            f"These interior state indices were never sent to get_payoffs_for_player: "
            f"{missing}.  If all collapse to 0, the bug is present."
        )


class TestTransitionMatrixCorrectness:
    """The transition matrix produced by a Python AbstractNPlayerStateGame
    must match the equivalent pure-C++ game for the same payoffs."""

    @pytest.mark.parametrize("pop_size", [5, 10, 15])
    def test_pd_matches_c_reference(self, pop_size):
        """PDGame (Python) and Matrix2PlayerGameHolder (C++) encode the same
        2×2 Prisoner's Dilemma; their transition matrices must be identical."""
        b, c = 3.0, 1.0
        beta, mu = 1.0, 0.01

        game_py = PDGame(pop_size, b=b, c=c)
        game_cpp = egt.games.Matrix2PlayerGameHolder(2, payoff_matrix=pd_payoff_matrix(b, c))

        ev_py = egt.analytical.PairwiseComparison(pop_size, game_py)
        ev_cpp = egt.analytical.PairwiseComparison(pop_size, game_cpp)

        T_py = ev_py.calculate_transition_matrix(beta, mu).toarray()
        T_cpp = ev_cpp.calculate_transition_matrix(beta, mu).toarray()

        np.testing.assert_allclose(
            T_py, T_cpp, atol=1e-10,
            err_msg=(
                f"Python PDGame and C++ reference give different transition matrices "
                f"for pop_size={pop_size}.  Likely cause: wrong state_index in "
                f"get_payoffs_for_player (calculate_state called with group_size "
                f"instead of pop_size)."
            ),
        )

    def test_transition_matrix_is_row_stochastic(self):
        for pop_size in (5, 10, 20):
            game = PDGame(pop_size)
            T, _ = build_transition_matrix(pop_size, game)
            row_sums = np.array(T.sum(axis=1)).ravel()
            np.testing.assert_allclose(
                row_sums, 1.0, atol=1e-10,
                err_msg=f"Row-stochastic property failed for pop_size={pop_size}",
            )

    def test_non_negative_entries(self):
        game = PDGame(10)
        T, _ = build_transition_matrix(10, game)
        assert T.min() >= -1e-14, "Negative transition probability found"


class TestCallbackCount:
    """get_payoffs_for_player must be called exactly once per fitness
    evaluation — not once per group configuration."""

    def test_single_callback_per_state(self):
        """For a 2-strategy game with pop_size=N, there are N-1 interior states,
        each requiring 2 fitness evaluations (one per strategy).  Monomorphic
        states require 1 fitness evaluation for the single present strategy.
        Total calls ≤ 2*(N-1) + 2 = 2*N."""
        pop_size = 10
        game = PDGame(pop_size)

        # Force re-evaluation by not pre-populating the LRU cache.
        evolver = egt.analytical.PairwiseComparison(pop_size, game, 0)  # cache_size=0
        evolver.calculate_transition_matrix(1.0, 0.01)

        max_expected = 2 * pop_size
        assert len(game.log_state_index) <= max_expected, (
            f"Expected at most {max_expected} Python callbacks but got "
            f"{len(game.log_state_index)}.  The hypergeometric loop may be "
            f"running inside Python (not C++)."
        )

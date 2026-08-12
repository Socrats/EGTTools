# Copyright (c) 2019-2026  Elias Fernandez
#
# This file is part of EGTtools.
#
# EGTtools is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# EGTtools is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with EGTtools.  If not, see <http://www.gnu.org/licenses/>

"""
Tests for the RNG/seeding small wins:

1. egttools.Random.get_state()/set_state() -- full engine state round-trip,
   distinct from current_seed() which only reports the last explicitly-set
   master seed.
2. Optional seed= constructor kwarg on PairwiseComparisonNumerical, giving
   independent, reproducible per-instance streams for the sequential
   trajectory methods (run/run_with_mutation/evolve).
3. Optional seed= constructor kwarg on NetworkMCEstimator* classes, giving
   independent, reproducible per-instance streams for the step-by-step
   session API (initialize/step).

Neither seed= kwarg affects the OpenMP-parallel batched estimate_* methods,
which still draw from the process-global egttools.Random singleton -- that
is documented as out of scope for this pass (see the MC improvements plan).
"""

import numpy as np
import pytest

egt = pytest.importorskip("egttools")

PairwiseComparisonNumerical = egt.numerical.PairwiseComparisonNumerical


# ---------------------------------------------------------------------------
# egttools.Random.get_state() / set_state()
# ---------------------------------------------------------------------------

class TestSeedGeneratorState:
    def test_get_state_returns_string(self):
        egt.Random.init_with_seed(1)
        state = egt.Random.get_state()
        assert isinstance(state, str)
        assert len(state) > 0

    def test_round_trip_reproduces_future_draws(self):
        egt.Random.init_with_seed(123)
        state = egt.Random.get_state()
        drawn_after_capture = [egt.Random.generate() for _ in range(10)]

        egt.Random.set_state(state)
        drawn_after_restore = [egt.Random.generate() for _ in range(10)]

        assert drawn_after_capture == drawn_after_restore

    def test_state_advances_between_draws(self):
        """The captured state must actually reflect engine position, not just
        the original seed -- two captures separated by draws must differ."""
        egt.Random.init_with_seed(456)
        state_a = egt.Random.get_state()
        egt.Random.generate()
        state_b = egt.Random.get_state()
        assert state_a != state_b

    def test_set_state_does_not_change_current_seed(self):
        """current_seed() reflects the last explicitly-set master seed, not
        the live engine position -- restoring a captured state must not
        change what it reports."""
        egt.Random.init_with_seed(789)
        seed_before = egt.Random.current_seed()
        egt.Random.generate()
        state = egt.Random.get_state()
        egt.Random.set_state(state)
        assert egt.Random.current_seed() == seed_before


# ---------------------------------------------------------------------------
# PairwiseComparisonNumerical(seed=...)
# ---------------------------------------------------------------------------

class TestPairwiseComparisonNumericalSeed:
    @staticmethod
    def _game():
        payoffs = np.array([[1.0, 0.0], [0.0, 1.0]])
        return egt.games.Matrix2PlayerGameHolder(2, payoffs)

    def test_no_seed_still_constructs(self, ):
        game = self._game()
        pc = PairwiseComparisonNumerical(20, game, 1000)
        assert pc.pop_size == 20

    def test_same_seed_reproduces_trajectory(self):
        game = self._game()
        init = np.array([10, 10], dtype=np.uint64)
        pc1 = PairwiseComparisonNumerical(20, game, 1000, seed=42)
        pc2 = PairwiseComparisonNumerical(20, game, 1000, seed=42)
        r1 = pc1.run_with_mutation(200, 1.0, 0.05, init)
        r2 = pc2.run_with_mutation(200, 1.0, 0.05, init)
        np.testing.assert_array_equal(r1, r2)

    def test_different_seed_diverges(self):
        game = self._game()
        init = np.array([10, 10], dtype=np.uint64)
        pc1 = PairwiseComparisonNumerical(20, game, 1000, seed=42)
        pc2 = PairwiseComparisonNumerical(20, game, 1000, seed=43)
        r1 = pc1.run_with_mutation(200, 1.0, 0.05, init)
        r2 = pc2.run_with_mutation(200, 1.0, 0.05, init)
        assert not np.array_equal(r1, r2)

    def test_explicit_seed_independent_of_global_singleton(self):
        """Two instances with the same explicit seed must reproduce each
        other's trajectory regardless of how many draws the global
        singleton has consumed in between construction calls."""
        game = self._game()
        init = np.array([10, 10], dtype=np.uint64)
        pc1 = PairwiseComparisonNumerical(20, game, 1000, seed=7)
        # Perturb the global singleton between constructions.
        for _ in range(50):
            egt.Random.generate()
        pc2 = PairwiseComparisonNumerical(20, game, 1000, seed=7)
        r1 = pc1.run_with_mutation(100, 1.0, 0.05, init)
        r2 = pc2.run_with_mutation(100, 1.0, 0.05, init)
        np.testing.assert_array_equal(r1, r2)


# ---------------------------------------------------------------------------
# NetworkMCEstimator*(seed=...)
# ---------------------------------------------------------------------------

class NeutralSpatialGame(egt.games.AbstractSpatialGame):
    """All fitnesses = 1 (neutral drift); enough to exercise session stepping."""

    def calculate_fitness(self, strategy_index, state):
        return 1.0

    def nb_strategies(self):
        return 2

    def toString(self):
        return "NeutralSpatialGame"

    def type(self):
        return "NeutralSpatialGame"


def _ring_topology(n):
    return {i: [(i - 1) % n, (i + 1) % n] for i in range(n)}


class TestNetworkMCEstimatorSeed:
    @staticmethod
    def _trajectory(seed, n_steps=20):
        from egttools.numerical.structure import NetworkMCEstimatorPC

        topo = _ring_topology(30)
        est = NetworkMCEstimatorPC(NeutralSpatialGame(), topo, 2, beta=1.0, mu=0.0, seed=seed)
        est.initialize()
        states = []
        for _ in range(n_steps):
            est.step()
            states.append(np.array(est.mean_population_state()).copy())
        return states

    def test_no_seed_still_constructs(self):
        from egttools.numerical.structure import NetworkMCEstimatorPC

        topo = _ring_topology(10)
        est = NetworkMCEstimatorPC(NeutralSpatialGame(), topo, 2, beta=1.0, mu=0.0)
        est.initialize()
        est.step()

    def test_same_seed_reproduces_session_trajectory(self):
        t1 = self._trajectory(seed=7)
        t2 = self._trajectory(seed=7)
        for a, b in zip(t1, t2):
            np.testing.assert_array_equal(a, b)

    def test_different_seed_diverges(self):
        t1 = self._trajectory(seed=7)
        t2 = self._trajectory(seed=8)
        assert any(not np.array_equal(a, b) for a, b in zip(t1, t2))

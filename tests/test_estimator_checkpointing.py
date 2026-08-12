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
Tests for the Python-chunk-granularity checkpointing added to
egttools.numerical.estimators (PairwiseComparisonEstimator / NetworkEstimator
share the same underlying helper, _run_chunks_with_checkpoint).

Split into:
1. Unit tests against _run_chunks_with_checkpoint directly -- deterministic,
   no MC randomness involved, exercise resume/kill/atomic-write/mismatch
   behavior precisely.
2. One small end-to-end integration test through PairwiseComparisonEstimator
   confirming the plumbing (params_key construction, checkpoint_path kwarg)
   actually wires the helper in correctly.
"""

import pickle

import numpy as np
import pytest

egt = pytest.importorskip("egttools")

from egttools.numerical.estimators import (  # noqa: E402
    PairwiseComparisonEstimator,
    _params_key,
    _run_chunks_with_checkpoint,
)


# ---------------------------------------------------------------------------
# Unit tests: _run_chunks_with_checkpoint
# ---------------------------------------------------------------------------

class TestRunChunksWithCheckpoint:
    def test_no_checkpoint_path_runs_all_chunks(self):
        sizes = [10, 10, 10]
        calls = []
        result = _run_chunks_with_checkpoint(
            lambda size: calls.append(size) or size * 2,
            sizes, verbose=0, desc="test", checkpoint_path=None, params_key="k",
        )
        assert result == [20, 20, 20]
        assert calls == sizes

    def test_checkpoint_resume_skips_completed_chunks(self, tmp_path):
        ckpt = str(tmp_path / "ckpt.pkl")
        sizes = [1, 1, 1, 1, 1, 1]
        params_key = _params_key("test", sizes)
        call_log = []

        def failing_compute(size):
            call_log.append(size)
            if len(call_log) == 3:
                raise RuntimeError("simulated kill")
            return len(call_log)

        with pytest.raises(RuntimeError, match="simulated kill"):
            _run_chunks_with_checkpoint(
                failing_compute, sizes, verbose=0, desc="test",
                checkpoint_path=ckpt, params_key=params_key,
            )
        # 2 chunks (idx 0, 1) completed and were checkpointed; the 3rd call
        # (idx 2) is the one that raised, before its result could be saved.
        assert len(call_log) == 3
        assert call_log == [1, 1, 1]

        # Resume: a fresh compute function that never fails.
        resumed_log = []

        def resumed_compute(size):
            resumed_log.append(size)
            return "resumed"

        result = _run_chunks_with_checkpoint(
            resumed_compute, sizes, verbose=0, desc="test",
            checkpoint_path=ckpt, params_key=params_key,
        )
        # Resume must restart at idx 2 (the chunk that never completed) and
        # process the remaining idx 2,3,4,5 -- 4 chunks, not 6: idx 0,1 are
        # NOT recomputed.
        assert len(resumed_log) == 4
        # Final result covers all 6 chunks: 2 from before the kill (values
        # 1, 2) plus 4 "resumed" placeholders.
        assert len(result) == 6
        assert result[:2] == [1, 2]
        assert result[2:] == ["resumed"] * 4

    def test_checkpoint_atomic_write_no_tmp_leftover(self, tmp_path):
        ckpt = str(tmp_path / "ckpt.pkl")
        sizes = [1, 1]
        _run_chunks_with_checkpoint(
            lambda size: size, sizes, verbose=0, desc="test",
            checkpoint_path=ckpt, params_key="k",
        )
        assert not (tmp_path / "ckpt.pkl.tmp").exists()

    def test_checkpoint_deleted_on_completion(self, tmp_path):
        ckpt = str(tmp_path / "ckpt.pkl")
        sizes = [1, 1, 1]
        _run_chunks_with_checkpoint(
            lambda size: size, sizes, verbose=0, desc="test",
            checkpoint_path=ckpt, params_key="k",
        )
        assert not (tmp_path / "ckpt.pkl").exists()

    def test_checkpoint_left_behind_on_failure(self, tmp_path):
        ckpt = str(tmp_path / "ckpt.pkl")
        sizes = [1, 1, 1]

        def fail_on_second_call(size, _count=[0]):
            _count[0] += 1
            if _count[0] == 2:
                raise RuntimeError("boom")
            return size

        with pytest.raises(RuntimeError):
            _run_chunks_with_checkpoint(
                fail_on_second_call, sizes, verbose=0, desc="test",
                checkpoint_path=ckpt, params_key="k",
            )
        assert (tmp_path / "ckpt.pkl").exists()

    def test_params_mismatch_raises(self, tmp_path):
        ckpt = str(tmp_path / "ckpt.pkl")
        sizes = [1, 1, 1]

        def fail_on_second_call(size, _count=[0]):
            _count[0] += 1
            if _count[0] == 2:
                raise RuntimeError("boom")
            return size

        with pytest.raises(RuntimeError):
            _run_chunks_with_checkpoint(
                fail_on_second_call, sizes, verbose=0, desc="test",
                checkpoint_path=ckpt, params_key="key-A",
            )
        assert (tmp_path / "ckpt.pkl").exists()

        with pytest.raises(ValueError, match="different parameters"):
            _run_chunks_with_checkpoint(
                lambda size: size, sizes, verbose=0, desc="test",
                checkpoint_path=ckpt, params_key="key-B",
            )

    def test_small_n_chunks_warns(self, tmp_path):
        ckpt = str(tmp_path / "ckpt.pkl")
        sizes = [1, 1]  # < 5 chunks
        with pytest.warns(RuntimeWarning, match="n_chunks"):
            _run_chunks_with_checkpoint(
                lambda size: size, sizes, verbose=0, desc="test",
                checkpoint_path=ckpt, params_key="k",
            )

    def test_rng_state_captured_in_checkpoint(self, tmp_path):
        ckpt = str(tmp_path / "ckpt.pkl")
        sizes = [1, 1, 1]

        def fail_on_second_call(size, _count=[0]):
            _count[0] += 1
            if _count[0] == 2:
                raise RuntimeError("boom")
            return size

        egt.Random.init_with_seed(1)
        with pytest.raises(RuntimeError):
            _run_chunks_with_checkpoint(
                fail_on_second_call, sizes, verbose=0, desc="test",
                checkpoint_path=ckpt, params_key="k",
            )
        with open(ckpt, "rb") as f:
            data = pickle.load(f)
        assert data["rng_state"] is not None
        assert isinstance(data["rng_state"], str)


# ---------------------------------------------------------------------------
# Integration test through PairwiseComparisonEstimator
# ---------------------------------------------------------------------------

class TestPairwiseComparisonEstimatorCheckpointing:
    @staticmethod
    def _estimator():
        payoffs = np.array([[1.0, 0.0], [0.0, 1.0]])
        game = egt.games.Matrix2PlayerGameHolder(2, payoffs)
        return PairwiseComparisonEstimator(game, pop_size=20, cache_size=500)

    def test_resume_completes_and_matches_call_count(self, tmp_path):
        ckpt = str(tmp_path / "ckpt.pkl")
        est = self._estimator()

        # The pybind11 C++ object doesn't allow monkeypatching its bound
        # methods directly (read-only attributes), so swap in a thin proxy
        # around it that makes estimate_strategy_distribution flaky while
        # forwarding everything else untouched.
        real_est = est._est
        call_count = {"n": 0}

        class FlakyProxy:
            def __getattr__(self, name):
                return getattr(real_est, name)

            def estimate_strategy_distribution(self, *args, **kwargs):
                call_count["n"] += 1
                if call_count["n"] == 2:
                    raise RuntimeError("simulated cluster kill")
                return real_est.estimate_strategy_distribution(*args, **kwargs)

        est._est = FlakyProxy()

        with pytest.raises(RuntimeError, match="simulated cluster kill"):
            est.estimate_strategy_distribution(
                nb_runs=60, nb_generations=50, transitory=5,
                beta=1.0, mu=0.05, n_chunks=6, checkpoint_path=ckpt,
            )
        assert call_count["n"] == 2  # 1 succeeded, 2nd raised

        # Un-flake (restore the real estimator) and resume with identical parameters.
        est._est = real_est
        result = est.estimate_strategy_distribution(
            nb_runs=60, nb_generations=50, transitory=5,
            beta=1.0, mu=0.05, n_chunks=6, checkpoint_path=ckpt,
        )
        assert result.mean.shape == (2,)
        assert not (tmp_path / "ckpt.pkl").exists()

    def test_checkpoint_path_none_is_unaffected(self):
        """Default behavior (no checkpointing) must be unchanged."""
        est = self._estimator()
        result = est.estimate_strategy_distribution(
            nb_runs=20, nb_generations=50, transitory=5,
            beta=1.0, mu=0.05, n_chunks=4,
        )
        assert result.mean.shape == (2,)
        assert np.all(np.isfinite(result.mean))

import numpy as np
from egttools.behaviors.CPR.cpr_strategies import (FairExtraction, HighExtraction, CommitmentStrategy, FakeStrategy,
                                                   FreeStrategy, NashExtraction)


def test_nash_extraction():
    a = 23
    b = 0.25
    N = 4
    strategy = NashExtraction()
    assert np.round(strategy.get_extraction(a, b, N), 1) == 18.4


def test_high_extraction():
    a = 23
    b = 0.25
    N = 4
    strategy = HighExtraction()
    assert np.round(strategy.get_extraction(a, b, N), 1) == 23.0


def test_fair_extraction():
    a = 23
    b = 0.25
    N = 4
    strategy = FairExtraction()
    assert np.round(strategy.get_extraction(a, b, N), 1) == 11.5


def test_commitment_strategy_threshold():
    strategy = CommitmentStrategy(3)

    assert strategy.is_commitment_validated(0) is False
    assert strategy.is_commitment_validated(2) is False
    assert strategy.is_commitment_validated(3) is True
    assert strategy.is_commitment_validated(4) is True


def test_commitment_strategy_extraction():
    strategy = CommitmentStrategy(3)
    a = 23
    b = 0.25
    N = 4

    assert strategy.get_extraction(a, b, N, commitment=False) == 23.0
    assert strategy.get_extraction(a, b, N, commitment=True) == 11.5

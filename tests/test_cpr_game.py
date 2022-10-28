from pytest import fixture

import numpy as np
from egttools.behaviors.CPR.abstract_cpr_strategy import AbstractCPRStrategy
from egttools.behaviors.CPR.cpr_strategies import (FairExtraction, HighExtraction, CommitmentStrategy, FakeStrategy,
                                                   FreeStrategy, NashExtraction)
from egttools.games.nonlinear_games import CommonPoolResourceDilemmaCommitment

from typing import Tuple, List


@fixture
def setup_cpr_strategy_parameters() -> Tuple[float, float, int]:
    a = 23
    b = 0.25
    N = 4

    return a, b, N


@fixture
def setup_cpr_game_parameters(setup_cpr_strategy_parameters) -> Tuple[
    int, float, float, float, float, List[AbstractCPRStrategy]]:
    a, b, N = setup_cpr_strategy_parameters
    fine = 5
    cost = 5
    strategies = [FairExtraction(), HighExtraction(), FakeStrategy(), FreeStrategy(),
                  CommitmentStrategy(1), CommitmentStrategy(2),
                  CommitmentStrategy(3), CommitmentStrategy(4)]

    return N, a, b, cost, fine, strategies


def test_nash_extraction(setup_cpr_strategy_parameters):
    a, b, N = setup_cpr_strategy_parameters

    strategy = NashExtraction()
    assert np.round(strategy.get_extraction(a, b, N, commitment=False), 1) == 18.4
    assert np.round(strategy.get_extraction(a, b, N, commitment=True), 1) == 18.4


def test_nash_strategy_commit_proposal():
    strategy = NashExtraction()

    assert strategy.proposes_commitment() is False


def test_nash_strategy_would_like_to_commit():
    strategy = NashExtraction()

    assert strategy.would_like_to_commit() is False


def test_nash_strategy_validates_commit():
    strategy = NashExtraction()

    assert strategy.is_commitment_validated(False) is False
    assert strategy.is_commitment_validated(True) is False


def test_high_extraction(setup_cpr_strategy_parameters):
    a, b, N = setup_cpr_strategy_parameters
    strategy = HighExtraction()
    assert np.round(strategy.get_extraction(a, b, N, commitment=False), 1) == 23.0
    assert np.round(strategy.get_extraction(a, b, N, commitment=True), 1) == 23.0


def test_high_strategy_commit_proposal():
    strategy = HighExtraction()

    assert strategy.proposes_commitment() is False


def test_high_strategy_would_like_to_commit():
    strategy = HighExtraction()

    assert strategy.would_like_to_commit() is False


def test_high_strategy_validates_commit():
    strategy = HighExtraction()

    assert strategy.is_commitment_validated(False) is False
    assert strategy.is_commitment_validated(True) is False


def test_fair_extraction(setup_cpr_strategy_parameters):
    a, b, N = setup_cpr_strategy_parameters
    strategy = FairExtraction()
    assert np.round(strategy.get_extraction(a, b, N, commitment=False), 1) == 11.5
    assert np.round(strategy.get_extraction(a, b, N, commitment=True), 1) == 11.5


def test_fair_payoff(setup_cpr_strategy_parameters):
    a, b, N = setup_cpr_strategy_parameters
    fine = 5
    cost = 5
    strategy = FairExtraction()
    assert strategy.get_payoff(a, b, strategy.get_extraction(a, b, N, True), 46, fine, cost, commitment=True) == 132.25
    assert strategy.get_payoff(a, b, strategy.get_extraction(a, b, N, True), 46, fine, cost, commitment=False) == 132.25


def test_fair_strategy_commit_proposal():
    strategy = FairExtraction()

    assert strategy.proposes_commitment() is False


def test_fair_strategy_would_like_to_commit():
    strategy = FairExtraction()

    assert strategy.would_like_to_commit() is True


def test_fair_strategy_validates_commit():
    strategy = FairExtraction()

    assert strategy.is_commitment_validated(False) is False
    assert strategy.is_commitment_validated(True) is True


def test_commitment_strategy_threshold():
    strategy = CommitmentStrategy(3)

    assert strategy.is_commitment_validated(0) is False
    assert strategy.is_commitment_validated(2) is False
    assert strategy.is_commitment_validated(3) is True
    assert strategy.is_commitment_validated(4) is True


def test_commitment_strategy_extraction(setup_cpr_strategy_parameters):
    a, b, N = setup_cpr_strategy_parameters
    strategy = CommitmentStrategy(3)

    assert strategy.get_extraction(a, b, N, commitment=False) == 23.0
    assert strategy.get_extraction(a, b, N, commitment=True) == 11.5


def test_commitment_strategy_commit_proposal():
    strategy = CommitmentStrategy(3)

    assert strategy.proposes_commitment() is True


def test_commitment_strategy_would_like_to_commit():
    strategy = CommitmentStrategy(3)

    assert strategy.would_like_to_commit() is True


def test_commitment_strategy_validates_commit():
    strategy = CommitmentStrategy(3)

    assert strategy.is_commitment_validated(0) is False
    assert strategy.is_commitment_validated(1) is False
    assert strategy.is_commitment_validated(2) is False
    assert strategy.is_commitment_validated(3) is True
    assert strategy.is_commitment_validated(4) is True


def test_fake_strategy(setup_cpr_strategy_parameters):
    a, b, N = setup_cpr_strategy_parameters
    strategy = FakeStrategy()

    assert strategy.get_extraction(a, b, N, commitment=False) == 23.0
    assert strategy.get_extraction(a, b, N, commitment=True) == 23.0


def test_fake_strategy_commit_proposal():
    strategy = FakeStrategy()

    assert strategy.proposes_commitment() is False


def test_fake_strategy_would_like_to_commit():
    strategy = FakeStrategy()

    assert strategy.would_like_to_commit() is True


def test_fake_strategy_validates_commit():
    strategy = FakeStrategy()

    assert strategy.is_commitment_validated(False) is False
    assert strategy.is_commitment_validated(True) is True


def test_free_strategy(setup_cpr_strategy_parameters):
    a, b, N = setup_cpr_strategy_parameters
    strategy = FreeStrategy()

    assert strategy.get_extraction(a, b, N, commitment=False) == 23.0
    assert strategy.get_extraction(a, b, N, commitment=True) == 11.5


def test_free_strategy_commit_proposal():
    strategy = FreeStrategy()

    assert strategy.proposes_commitment() is False


def test_free_strategy_would_like_to_commit():
    strategy = FreeStrategy()

    assert strategy.would_like_to_commit() is True


def test_free_strategy_validates_commit():
    strategy = FreeStrategy()

    assert strategy.is_commitment_validated(False) is False
    assert strategy.is_commitment_validated(True) is True


def test_cpr_game_play(setup_cpr_game_parameters):
    # There are 8 strategies:
    # Fair, High, Fake, Free, COM1, COM2, COM3, COM4
    group_size, a, b, cost, fine, strategies = setup_cpr_game_parameters
    game = CommonPoolResourceDilemmaCommitment(group_size, a, b, cost, fine, strategies)

    game_payoffs = np.zeros(shape=(game.nb_strategies(),))
    # Case of all 3 Fair 1 High
    group_composition = [3, group_size - 3, 0, 0, 0, 0, 0, 0]

    game.play(group_composition, game_payoffs)

    assert game_payoffs[0] == 99.1875
    assert game_payoffs[1] == 198.375

    game_payoffs.fill(0)

    # Case of all 3 Fair 1 High
    group_composition = [2, 1, 1, 0, 0, 0, 0, 0]

    game.play(group_composition, game_payoffs)

    assert game_payoffs[0] == 66.125
    assert game_payoffs[1] == 132.25
    assert game_payoffs[2] == 132.25


def test_cpr_calculate_payoffs(setup_cpr_game_parameters):
    pass


def test_cpr_calculate_total_extraction(setup_cpr_game_parameters):
    # There are 8 strategies:
    # Fair, High, Fake, Free, COM1, COM2, COM3, COM4
    group_size, a, b, cost, fine, strategies = setup_cpr_game_parameters
    game = CommonPoolResourceDilemmaCommitment(group_size, a, b, cost, fine, strategies)

    commitment_accepted = np.zeros(shape=(game.nb_strategies(),), dtype=bool)

    # Case of all Fair
    group_composition = [group_size, 0, 0, 0, 0, 0, 0, 0]
    nb_committed = game.get_nb_committed(group_composition)
    game.check_if_commitment_validated(nb_committed, group_composition, commitment_accepted)
    assert game.calculate_total_extraction(commitment_accepted, group_composition) == 46

    commitment_accepted.fill(0)

    # Case of all High
    group_composition = [0, group_size, 0, 0, 0, 0, 0, 0]
    nb_committed = game.get_nb_committed(group_composition)
    game.check_if_commitment_validated(nb_committed, group_composition, commitment_accepted)
    assert game.calculate_total_extraction(commitment_accepted, group_composition) == 92

    commitment_accepted.fill(0)

    # Case of all Fake
    group_composition = [0, 0, group_size, 0, 0, 0, 0, 0]
    nb_committed = game.get_nb_committed(group_composition)
    game.check_if_commitment_validated(nb_committed, group_composition, commitment_accepted)
    assert game.calculate_total_extraction(commitment_accepted, group_composition) == 92

    commitment_accepted.fill(0)

    # Case of all Free
    group_composition = [0, 0, 0, group_size, 0, 0, 0, 0]
    nb_committed = game.get_nb_committed(group_composition)
    game.check_if_commitment_validated(nb_committed, group_composition, commitment_accepted)
    assert game.calculate_total_extraction(commitment_accepted, group_composition) == 92

    commitment_accepted.fill(0)

    # Case of all COM1
    group_composition = [0, 0, 0, 0, group_size, 0, 0, 0]
    nb_committed = game.get_nb_committed(group_composition)
    game.check_if_commitment_validated(nb_committed, group_composition, commitment_accepted)
    assert game.calculate_total_extraction(commitment_accepted, group_composition) == 46

    commitment_accepted.fill(0)

    # Case of all COM2
    group_composition = [0, 0, 0, 0, 0, group_size, 0, 0]
    nb_committed = game.get_nb_committed(group_composition)
    game.check_if_commitment_validated(nb_committed, group_composition, commitment_accepted)
    assert game.calculate_total_extraction(commitment_accepted, group_composition) == 46

    commitment_accepted.fill(0)

    # Case of all COM3
    group_composition = [0, 0, 0, 0, 0, 0, group_size, 0]
    nb_committed = game.get_nb_committed(group_composition)
    game.check_if_commitment_validated(nb_committed, group_composition, commitment_accepted)
    assert game.calculate_total_extraction(commitment_accepted, group_composition) == 46

    commitment_accepted.fill(0)

    # Case of all COM4
    group_composition = [0, 0, 0, 0, 0, 0, 0, group_size]
    nb_committed = game.get_nb_committed(group_composition)
    game.check_if_commitment_validated(nb_committed, group_composition, commitment_accepted)
    assert game.calculate_total_extraction(commitment_accepted, group_composition) == 46

    commitment_accepted.fill(0)

    # Case of all 3 Fair 1 High
    group_composition = [3, group_size - 3, 0, 0, 0, 0, 0, 0]
    nb_committed = game.get_nb_committed(group_composition)
    game.check_if_commitment_validated(nb_committed, group_composition, commitment_accepted)
    assert all(commitment_accepted == 0)
    assert game.calculate_total_extraction(commitment_accepted, group_composition) == 57.5


def test_cpr_get_nb_committed(setup_cpr_game_parameters):
    # There are 8 strategies:
    # Fair, High, Fake, Free, COM1, COM2, COM3, COM4

    group_size, a, b, cost, fine, strategies = setup_cpr_game_parameters
    game = CommonPoolResourceDilemmaCommitment(group_size, a, b, cost, fine, strategies)

    # Case of all Fair
    group_composition = [group_size, 0, 0, 0, 0, 0, 0, 0]
    assert game.get_nb_committed(group_composition) == group_size

    # Case of all High
    group_composition = [0, group_size, 0, 0, 0, 0, 0, 0]
    assert game.get_nb_committed(group_composition) == 0

    # Case of all Fake
    group_composition = [0, 0, group_size, 0, 0, 0, 0, 0]
    assert game.get_nb_committed(group_composition) == group_size

    # Case of all Free
    group_composition = [0, 0, 0, group_size, 0, 0, 0, 0]
    assert game.get_nb_committed(group_composition) == group_size

    # Case of all COM1
    group_composition = [0, 0, 0, 0, group_size, 0, 0, 0]
    assert game.get_nb_committed(group_composition) == group_size

    # Case of all COM2
    group_composition = [0, 0, 0, 0, 0, group_size, 0, 0]
    assert game.get_nb_committed(group_composition) == group_size

    # Case of all COM2
    group_composition = [0, 0, 0, 0, 0, 0, group_size, 0]
    assert game.get_nb_committed(group_composition) == group_size

    # Case of all COM4
    group_composition = [0, 0, 0, 0, 0, 0, 0, group_size]
    assert game.get_nb_committed(group_composition) == group_size

    # Let's also check a mix of High and another strategy
    group_composition = [0, group_size - 1, 0, 0, 0, 0, 0, 1]
    assert game.get_nb_committed(group_composition) == 1


def test_cpr_check_if_commitment_validated(setup_cpr_game_parameters):
    # There are 8 strategies:
    # Fair, High, Fake, Free, COM1, COM2, COM3, COM4

    group_size, a, b, cost, fine, strategies = setup_cpr_game_parameters
    game = CommonPoolResourceDilemmaCommitment(group_size, a, b, cost, fine, strategies)

    commitment_accepted = np.zeros(shape=(game.nb_strategies(),), dtype=bool)

    # Case of all Fair
    group_composition = [group_size, 0, 0, 0, 0, 0, 0, 0]
    nb_committed = game.get_nb_committed(group_composition)
    game.check_if_commitment_validated(nb_committed, group_composition, commitment_accepted)
    assert np.all(commitment_accepted == 0)

    commitment_accepted.fill(0)

    # Case of all High
    group_composition = [0, group_size, 0, 0, 0, 0, 0, 0]
    nb_committed = game.get_nb_committed(group_composition)
    game.check_if_commitment_validated(nb_committed, group_composition, commitment_accepted)
    assert np.all(commitment_accepted == 0)

    commitment_accepted.fill(0)

    # Case of all Fake
    group_composition = [0, 0, group_size, 0, 0, 0, 0, 0]
    nb_committed = game.get_nb_committed(group_composition)
    game.check_if_commitment_validated(nb_committed, group_composition, commitment_accepted)
    assert np.all(commitment_accepted == 0)

    commitment_accepted.fill(0)

    # Case of all Free
    group_composition = [0, 0, 0, group_size, 0, 0, 0, 0]
    nb_committed = game.get_nb_committed(group_composition)
    game.check_if_commitment_validated(nb_committed, group_composition, commitment_accepted)
    assert np.all(commitment_accepted == 0)

    commitment_accepted.fill(0)

    # Case of all COM1
    group_composition = [0, 0, 0, 0, group_size, 0, 0, 0]
    nb_committed = game.get_nb_committed(group_composition)
    game.check_if_commitment_validated(nb_committed, group_composition, commitment_accepted)
    assert commitment_accepted[4]

    commitment_accepted.fill(0)

    # Case of all COM2
    group_composition = [0, 0, 0, 0, 0, group_size, 0, 0]
    nb_committed = game.get_nb_committed(group_composition)
    game.check_if_commitment_validated(nb_committed, group_composition, commitment_accepted)
    assert commitment_accepted[5]

    commitment_accepted.fill(0)

    # Case of all COM3
    group_composition = [0, 0, 0, 0, 0, 0, group_size, 0]
    nb_committed = game.get_nb_committed(group_composition)
    game.check_if_commitment_validated(nb_committed, group_composition, commitment_accepted)
    assert commitment_accepted[6]

    commitment_accepted.fill(0)

    # Case of all COM4
    group_composition = [0, 0, 0, 0, 0, 0, 0, group_size]
    nb_committed = game.get_nb_committed(group_composition)
    game.check_if_commitment_validated(nb_committed, group_composition, commitment_accepted)
    assert commitment_accepted[7]

    commitment_accepted.fill(0)

    # Let's also check a mix of High and another strategy
    group_composition = [0, group_size - 1, 0, 0, 0, 0, 0, 1]
    nb_committed = game.get_nb_committed(group_composition)
    game.check_if_commitment_validated(nb_committed, group_composition, commitment_accepted)
    assert not commitment_accepted[7]
    assert not commitment_accepted[1]

    commitment_accepted.fill(0)

    # Let's check if 1 COM1 is present with the rest High
    group_composition = [0, group_size - 1, 0, 0, 1, 0, 0, 0]
    nb_committed = game.get_nb_committed(group_composition)
    game.check_if_commitment_validated(nb_committed, group_composition, commitment_accepted)
    assert not commitment_accepted[1]
    assert commitment_accepted[4]

    commitment_accepted.fill(0)

    # Let's check if 1 COM2 is present with the rest High
    group_composition = [0, group_size - 1, 0, 0, 0, 1, 0, 0]
    nb_committed = game.get_nb_committed(group_composition)
    game.check_if_commitment_validated(nb_committed, group_composition, commitment_accepted)
    assert not commitment_accepted[1]
    assert not commitment_accepted[5]

    commitment_accepted.fill(0)

    # Let's check if 1 COM3 is present with the rest High
    group_composition = [0, group_size - 1, 0, 0, 0, 0, 0, 1]
    nb_committed = game.get_nb_committed(group_composition)
    game.check_if_commitment_validated(nb_committed, group_composition, commitment_accepted)
    assert not commitment_accepted[1]
    assert not commitment_accepted[6]

    commitment_accepted.fill(0)

    # Let's check if 1 COM4 is present with the rest High
    group_composition = [0, group_size - 1, 0, 0, 0, 0, 0, 1]
    nb_committed = game.get_nb_committed(group_composition)
    game.check_if_commitment_validated(nb_committed, group_composition, commitment_accepted)
    assert not commitment_accepted[1]
    assert not commitment_accepted[7]

    commitment_accepted.fill(0)

    # Let's check if 1 COM4 with Fair, Fake and Free
    group_composition = [1, 0, 1, 1, 0, 0, 0, group_size - 3]
    nb_committed = game.get_nb_committed(group_composition)
    game.check_if_commitment_validated(nb_committed, group_composition, commitment_accepted)
    assert commitment_accepted[0]
    assert commitment_accepted[2]
    assert commitment_accepted[3]
    assert commitment_accepted[7]

    commitment_accepted.fill(0)

    # Let's check if 1 COM4 with High, Fake and Free
    group_composition = [0, 1, 1, 1, 0, 0, 0, group_size - 3]
    nb_committed = game.get_nb_committed(group_composition)
    assert nb_committed == 3
    game.check_if_commitment_validated(nb_committed, group_composition, commitment_accepted)
    assert not commitment_accepted[1]
    assert not commitment_accepted[2]
    assert not commitment_accepted[3]
    assert not commitment_accepted[7]

# def test_fair_payoff_from_game(setup_cpr_game_parameters):
#     # There are 8 strategies:
#     # Fair, High, Fake, Free, COM1, COM2, COM3, COM4
#
#     group_size, a, b, cost, fine, strategies = setup_cpr_game_parameters
#     game = CommonPoolResourceDilemmaCommitment(group_size, a, b, cost, fine, strategies)
#
#     commitment_accepted = np.zeros(shape=(game.nb_strategies(),), dtype=bool)
#
#     strategy = FairExtraction()
#
#     game.grou
#
#     assert strategy.get_payoff(a, b, strategy.get_extraction(a, b, group_size, True), 46, fine, cost, commitment=True) == 132.25
#     assert strategy.get_payoff(a, b, strategy.get_extraction(a, b, group_size, True), 46, fine, cost, commitment=False) == 132.25

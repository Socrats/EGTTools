"""
API reference documentation for `behaviors.CRD` submodule.
"""

from egttools.numerical.numerical.behaviors.CRD import AbstractCRDStrategy, CRDMemoryOnePlayer
from .moving_average import MovingAverageCRDStrategy
from .goal_based import GoalBasedCRDStrategy
from .time_based import TimeBasedCRDStrategy

__all__ = ['AbstractCRDStrategy', 'CRDMemoryOnePlayer', 'MovingAverageCRDStrategy', 'GoalBasedCRDStrategy',
           'TimeBasedCRDStrategy']

"""
API reference documentation for `behaviors.CRD` submodule.
"""
try:
    from egttools.numerical.numerical_.behaviors.CRD import AbstractCRDStrategy, CRDMemoryOnePlayer
except Exception:
    raise Exception("numerical package not initialized")
else:
    from .moving_average import MovingAverageCRDStrategy
    from .goal_based import GoalBasedCRDStrategy
    from .time_based import TimeBasedCRDStrategy

__all__ = ['AbstractCRDStrategy', 'CRDMemoryOnePlayer', 'MovingAverageCRDStrategy', 'GoalBasedCRDStrategy',
           'TimeBasedCRDStrategy']

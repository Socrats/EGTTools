"""The `numerical` module contains functions and classes to simulate evolutionary dynamics in finite populations."""

try:
    import egttools.numerical.numerical_ as numerical
except Exception:
    raise Exception("numerical package not initialized")
else:
    from .numerical_ import PairwiseComparisonNumerical
    from .numerical_ import PairwiseComparisonTransitionOperator
    from .numerical_ import GeneralPopulationEvolver
    from .numerical_ import NetworkEvolver
    from .indicators import StationaryIndicatorResult
    from . import linear_operator
    from .linear_operator import stationary_distribution_from_sparse
    from .results import (
        FixationResult,
        AbsorptionTimeResult,
        AbsorptionProbabilityResult,
        StrategyDistributionResult,
        StationaryDistributionResult,
        AGoSResult,
    )
    from .estimators import PairwiseComparisonEstimator, NetworkEstimator

__all__ = [
    'numerical', 'PairwiseComparisonNumerical', 'PairwiseComparisonTransitionOperator',
    'GeneralPopulationEvolver', 'NetworkEvolver', 'StationaryIndicatorResult',
    'linear_operator', 'stationary_distribution_from_sparse',
    'FixationResult', 'AbsorptionTimeResult', 'AbsorptionProbabilityResult',
    'StrategyDistributionResult', 'StationaryDistributionResult', 'AGoSResult',
    'PairwiseComparisonEstimator', 'NetworkEstimator',
]

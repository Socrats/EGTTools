"""The `numerical` module contains functions and classes to simulate evolutionary dynamics in finite populations."""

try:
    import egttools.numerical.numerical_ as numerical
except Exception:
    raise Exception("numerical package not initialized")
else:
    from .numerical_ import PairwiseComparisonNumerical
    from .numerical_ import GeneralPopulationEvolver
    from .numerical_ import NetworkEvolver

__all__ = ['numerical', 'PairwiseComparisonNumerical', 'GeneralPopulationEvolver', 'NetworkEvolver']

"""The `numerical` module contains functions and classes to simulate evolutionary dynamics in finite populations."""

try:
    import egttools.numerical.numerical as numerical
except Exception:
    raise Exception("numerical package not initialized")
else:
    from egttools.numerical.numerical import PairwiseComparisonNumerical
    from egttools.numerical.numerical import GeneralPopulationEvolver

__all__ = ['numerical', 'PairwiseComparisonNumerical', 'GeneralPopulationEvolver']

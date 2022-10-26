"""The `numerical` module contains functions and classes to simulate evolutionary dynamics in finite populations."""

try:
    import egttools.numerical.numerical as numerical
except Exception:
    raise Exception("numerical package not initialized")
else:
    from egttools.numerical.numerical import PairwiseMoran

__all__ = ['numerical', 'PairwiseMoran']

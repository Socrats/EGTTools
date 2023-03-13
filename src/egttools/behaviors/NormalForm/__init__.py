"""
API reference documentation for `behaviors.NormalForm` submodule.
"""
try:
    from egttools.numerical.numerical_.behaviors.NormalForm import AbstractNFGStrategy
except Exception:
    raise Exception("numerical package not initialized")
else:
    import egttools.behaviors.NormalForm.TwoActions as TwoActions

__all__ = ['AbstractNFGStrategy', 'TwoActions']

"""
The :mod:`egttools` package implements methods to study evolutionary dynamics.
"""
# Workaround for mac OSX
# import numpy

try:
    import egttools.numerical as numerical
except Exception:
    raise Exception("numerical package not initialized")
else:
    # Now we delete the pre-loaded numpy, as it is not necessary
    # del numpy
    from .numerical.numerical_ import __version__
    from .numerical.numerical_ import VERSION
    from .numerical.numerical_ import Random
    from .numerical.numerical_ import USES_BOOST
    from .numerical.numerical_ import (sample_simplex, sample_unit_simplex, calculate_nb_states,
                                       calculate_state,
                                       calculate_strategies_distribution, )

    import egttools.games as games
    import egttools.behaviors as behaviors
    import egttools.analytical as analytical
    import egttools.utils as utils
    import egttools.plotting as plotting
    import egttools.distributions as distributions
    import egttools.datastructures as datastructures

__all__ = ['utils', 'plotting', 'analytical',
           'games', 'behaviors', 'numerical',
           'distributions', 'datastructures', '__version__', 'VERSION', 'Random',
           'sample_simplex', 'sample_unit_simplex', 'calculate_nb_states', 'calculate_state',
           'calculate_strategies_distribution']

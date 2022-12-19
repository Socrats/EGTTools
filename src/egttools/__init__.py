"""
The :mod:`egttools` package implements methods to study evolutionary dynamics.
"""
try:
    import egttools.numerical as numerical
except Exception:
    raise Exception("numerical package not initialized")
else:
    from egttools.numerical.numerical import __version__
    from egttools.numerical.numerical import VERSION
    from egttools.numerical.numerical import Random
    from egttools.numerical.numerical import (sample_simplex, sample_unit_simplex, calculate_nb_states, calculate_state,
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

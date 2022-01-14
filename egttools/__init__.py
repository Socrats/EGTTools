"""
The :mod:`egttools` package implements methods to study evolutionary dynamics.
"""
try:
    import egttools.numerical as numerical
except Exception:
    raise Exception("numerical package not initialized")
else:
    from egttools.numerical import __version__
    from egttools.numerical import VERSION
    from egttools.numerical import Random
    from egttools.numerical import (sample_simplex, calculate_nb_states, calculate_state,
                                    calculate_strategies_distribution, )

    import egttools.games as games
    import egttools.behaviors as behaviors
    import egttools.analytical as analytical
    import egttools.utils as utils
    import egttools.plotting as plotting

__all__ = ['utils', 'plotting', 'analytical',
           'games', 'behaviors', 'numerical', '__version__', 'VERSION', 'Random',
           'sample_simplex', 'calculate_nb_states', 'calculate_state', 'calculate_strategies_distribution']

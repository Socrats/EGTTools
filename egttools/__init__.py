"""
The :mod:`egttools` package implements methods to study evolutionary dynamics.
"""

from .utils import find_saddle_type_and_gradient_direction
import egttools.plotting as plotting
import egttools.analytical as analytical
import egttools.games as games
import egttools.behaviors as behaviors

import egttools.numerical as numerical
from egttools.numerical import __version__
from egttools.numerical import VERSION
from egttools.numerical import Random
from egttools.numerical import (sample_simplex, calculate_nb_states, calculate_state,
                                calculate_strategies_distribution, )

__all__ = ['find_saddle_type_and_gradient_direction',
           'plotting', 'analytical',
           'games', 'behaviors', 'numerical', '__version__', 'VERSION', 'Random',
           'sample_simplex', 'calculate_nb_states', 'calculate_state', 'calculate_strategies_distribution']

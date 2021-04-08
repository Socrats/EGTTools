try:
    from egttools.numerical.behaviors import *
except ImportError:
    raise ImportError("Cannot import numerical.behaviors. The numerical model might not have been compiled correctly.")

from .pgg_behaviors import PGGOneShotStrategy, player_factory

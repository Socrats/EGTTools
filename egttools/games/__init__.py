try:
    from egttools.numerical.games import *
except ImportError:
    raise ImportError("Cannot import numerical.games. The numerical model might not have been compiled correctly.")

from .pgg import PGG

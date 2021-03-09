from .utils import find_saddle_type_and_gradient_direction
import egttools.plotting as plotting
import egttools.analytical as analytical

try:
    import egttools.numerical as numerical
except ImportError:
    raise ImportError("Numerical module cannot be imported. It might not have been compiled correctly.")
else:
    from egttools.numerical import (sample_simplex, calculate_nb_states, calculate_state, )

from .utils import find_saddle_type_and_gradient_direction
import egttools.plotting as plotting
import egttools.analytical as analytical

try:
    import egttools.numerical as numerical
except ImportError:
    print("Numerical module cannot be imported. It might not have been compiled correctly.")

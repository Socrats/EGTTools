"""
The :mod:`egttools` package implements methods to study evolutionary dynamics.
"""
import importlib.resources

try:
    from importlib.resources import files
except ImportError:
    from importlib_resources import files

try:
    # Load raw build info text
    _build_info_file = files(__package__).joinpath('numerical/egttools_build_info.txt')
    _raw_build_info = _build_info_file.read_text(encoding='utf-8').strip()

    # Parse into a structured dictionary
    _build_info_lines = _raw_build_info.splitlines()
    _build_info_dict = {}

    for line in _build_info_lines:
        if ':' in line:
            key, value = line.split(':', 1)
            _build_info_dict[key.strip()] = value.strip()

    # Expose structured info
    __build_info__ = _build_info_dict

except Exception:
    __build_info__ = {"Error": "Build information unavailable."}


def show_build_info():
    """
    Nicely print the EGTtools build information.
    """
    if not isinstance(__build_info__, dict):
        print("Build information unavailable.")
        return

    if "Error" in __build_info__:
        print("Build information unavailable.")
        print("(No 'egttools_build_info.txt' found inside the package.)")
    else:
        print("\nEGTtools Build Information")
        print("-" * 30)
        for key, value in __build_info__.items():
            print(f"{key:<20}: {value}")


try:
    import egttools.numerical as numerical
except Exception:
    raise Exception("numerical package not initialized")
else:
    # Now we delete the pre-loaded numpy, as it is not necessary
    # del numpy
    from .numerical.numerical_ import __version__
    from .numerical.numerical_ import VERSION
    from .numerical.numerical_ import is_openmp_enabled
    from .numerical.numerical_ import is_blas_lapack_enabled
    from .numerical.numerical_ import USES_BOOST
    from .numerical.numerical_.random import Random
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

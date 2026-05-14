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
    Nicely print the EGTtools build information and runtime feature support.
    """
    if not isinstance(__build_info__, dict) or "Error" in __build_info__:
        print("Build information unavailable.")
        print("(No 'egttools_build_info.txt' found inside the package.)")
    else:
        print("\nEGTtools Build Information")
        print("-" * 30)
        for key, value in __build_info__.items():
            print(f"{key:<22}: {value}")

    print("\nFeature Support (runtime)")
    print("-" * 30)
    print(f"{'OpenMP':<22}: {'ON' if is_openmp_enabled() else 'OFF'}")
    print(f"{'BLAS/LAPACK':<22}: {'ON' if is_blas_lapack_enabled() else 'OFF'}")
    print(f"{'Boost':<22}: {'ON' if is_boost_enabled() else 'OFF'}")
    print(f"{'ARPACK':<22}: {'ON' if is_arpack_enabled() else 'OFF'}")
    print(f"{'PETSc/SLEPc':<22}: {'ON' if is_petsc_enabled() else 'OFF'}")


try:
    from ._version import __version__, version_tuple as __version_info__
except ImportError:
    # Fallback when _version.py hasn't been generated yet (e.g. editable install
    # from a git checkout before a build, or missing setuptools-scm).
    try:
        from .numerical.numerical_ import __version__, __version_info__
    except Exception:
        __version__ = "0.0.0"
        __version_info__ = (0, 0, 0)

try:
    import egttools.numerical as numerical
except Exception:
    raise Exception("numerical package not initialized")
else:
    from .numerical.numerical_ import is_openmp_enabled
    from .numerical.numerical_ import is_blas_lapack_enabled
    from .numerical.numerical_ import is_boost_enabled
    from .numerical.numerical_ import is_arpack_enabled
    from .numerical.numerical_ import is_petsc_enabled
    from .numerical.numerical_ import USES_BOOST
    from .numerical.numerical_.random import Random
    from .numerical.numerical_ import (sample_simplex, sample_unit_simplex, calculate_nb_states,
                                       calculate_state,
                                       calculate_strategies_distribution,
                                       calculate_expected_payoff,
                                       calculate_expected_indicator,
                                       calculate_expected_indicators,
                                       calculate_expected_indicators_precomputed,
                                       calculate_expected_group_success,
                                       calculate_expected_state_indicator,
                                       calculate_expected_state_indicators,
                                       calculate_expected_state_indicators_precomputed,
                                       precompute_group_to_state_indicator_matrix,
                                       calculate_hypergeometric_expected_value,
                                       calculate_hypergeometric_fitness, )

    import egttools.games as games
    import egttools.behaviors as behaviors
    import egttools.analytical as analytical
    import egttools.utils as utils
    import egttools.plotting as plotting
    import egttools.distributions as distributions
    import egttools.datastructures as datastructures

__all__ = ['utils', 'plotting', 'analytical',
           'games', 'behaviors', 'numerical',
           'distributions', 'datastructures', '__version__', '__version_info__', 'Random',
           'show_build_info', '__build_info__', 'USES_BOOST',
           'is_openmp_enabled', 'is_blas_lapack_enabled', 'is_boost_enabled',
           'is_arpack_enabled', 'is_petsc_enabled',
           'sample_simplex', 'sample_unit_simplex', 'calculate_nb_states', 'calculate_state',
           'calculate_strategies_distribution',
           'calculate_expected_payoff',
           'calculate_expected_indicator',
           'calculate_expected_indicators',
           'calculate_expected_indicators_precomputed',
           'calculate_expected_group_success',
           'calculate_expected_state_indicator',
           'calculate_expected_state_indicators',
           'calculate_expected_state_indicators_precomputed',
           'precompute_group_to_state_indicator_matrix',
           'calculate_hypergeometric_expected_value',
           'calculate_hypergeometric_fitness']

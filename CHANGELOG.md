# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/) and this project adheres
to [Semantic Versioning](http://semver.org/spec/v2.0.0.html).

## [0.1.10] - 06-08-2022

### Fixed

- fixed issue with the comparison of different types in `PairwiseMoran`
- fixed issue with colorbar so that now the axis is passed to maptlolib colorbar, this way it will be plotted correctly
  when drawing multiple subplots
- fixed issue with hardcoded x-axis in `plot_gradients`
- fixed error on `calculate_full_transition_matrix`. The error happened when calculating the transition probability:
  - a) although the literature is a bit confusing on this, the original paper by Traulsen 2006 says that for the transition we consider that the strategies to reproduce and die are pricked simultaneously (so both with probability 1/Z).
  - b) the is more accumulated numerical error when doing probability of transitioning from A to B P(A, B) = fermi(-beta, B - A) than when doing fermi(beta, A - B). This is probably specific to Python and Numpy, but must be taken into account in the future.
  - c) The schur decomposition (egttools.utils.calculate_stationary_distribution_non_hermitian) works better in this case (although still has a slight numerical error) and should be used for full transition matrices).
- normalized transition probabilities to use the definition in Traulsen et al. 2006
  - now we assume that both death and birth individuals are selected simultaneously with probability n_i/Z, where n_i is the number of individuals of that strategy in the population

### Changed

- updated installation instructions
- updated to PEP 621 syntax
- updated setup.py since now scikit-build supports VS2019
- updated `draw_stationary_distribution` to make the display of labels optional
- changed stability calculation for the replicator dynamics to use the Jacobian matrix
- updated `plot_gradients` to check for all possible types of roots (stable, unstable and saddle)
- removed stability checks for the stochastic dynamics
    - if T+ is too small, phi will be approximated to infinity and the fixation probability will be approximated to 0.
    - This may not be correct, since if p_minus is also very small or equal to p_plus, the outcome would be different.
      So it might change in a future version
- updated default language for documentation to `en`
- updated docstrings
- changed colorbar default label to gradient of selection
- droped pin to Sphinx <= 4.5.0
- updated variable name in AbstractGame
- changed name of variable in `calculate_fitness` method of Abstract game

### Added

- added input parameter checks for `run` and `evolve` methods of `PairwiseMoran`
- created a method to calculate roots
- created a method to check the stability of the replicator dynamics through the Jacobian matrix
- added a check for the limit case in which the only non-negative eigenvalue is close to atol
- added new notebook with examples of use
- added an extra check when calculating fixation probabilities
- added Python 3.10 binary - except for Windows and manylinuxi686 as non numpy or scipy builds yet available
- added new CRD strategy
- added extra tolerance controls in `check_replicator_stability_pairwise_games`
- added new abstract game classes to simplify game implementation
- added `NPlayerStagHunt` game

## [0.1.9-patch6] - 16-02-2022

### Fixed

- Fixed error on version formatting, it should be `0.1.9.dev6` instead of `0.1.9.patch6`.

## [0.1.9-patch5] - 16-02-2022

### Fixed

- Fixed wrong version tag on git.

## [0.1.9-patch4] - 16-02-2022

## Fixed

- There was a problem with setting a `geometric_distribution` in C++ as a private variable for OpenMP which caused some
  errors when estimating stationary distributions. This was fixed by setting it as a shared variable.

## Added

- Binder links to run examples and updated notebooks

## [0.1.9-patch3] - 03-02-2022

### Added

- Added `gitter` chat and `binder` launch.

### Fixed

- Added missing `seaborn` dependency. This dependency is only needed to be able to automatically generate colorblind
  colors to plot the invasion diagram, so it might be dropped in the future. But, for the moments, it provides the
  easiest way to do this.

### Changed

- Updated docs and notebooks to use the latest `egttools` API.

## [0.1.9-patch2] - 26-01-2022

### Fixed

- This release fixes an issue with a modulo operation that was causing an index error when calculating stability for
  plotting dynamics on a simplex.

## [0.1.9-patch1] - 19-01-2022

### Fixed

- This release fixes an issue with CITATION.cff which prevented zenodo from publishing a new doi.

## [0.1.9] - 18-01-2022

### Added

- Added `plot_replicator_dynamics_in_simplex` and `plot_moran_dynamics_in_simplex` to simplify the plotting of 2
  Simplexes when using `replicator_equation` and `StochDynamics` provided in `egttools`.
- Added an extra example to the docs to showcase the simplified plotting.

### Fixed

- Fixed wrong numpy use in `egttools.utils.calculate_stationary_distribution`. Instead of `numpy.eig` the correct use
  is `numpy.linalg.eig`.
- Fixed issue with plotting edges which have random drift.
  Before `egttools.plotting.helpers.calculate_stationary_points` would logically find many roots in an edge with random
  drift and `Simplex2D.draw_stationary_points` would attempt to draw all of them. Now, we first search of edges in which
  there is random drift, and mask them, so that no stationary points nor trajectories are plot in the edge. Instead, we
  draw a `dashed` line to represent the random drift.

### Changed

- Updated `Readme.md` to use the simplified `plot_replicator_dynamics_in_simplex` in the example of 2 Simplex plotting.

## [0.1.8] - 15-01-2022

### Added

- Added `Simplex2D` class which can be used to plot the evolutionary dynamics in a 2-Simplex.
- Added `OneShotCRD` game.
- Added `egttools.utils.calculate_stationary_distribution_non_hermitian` which uses `scipy.linalg.schur` instead
  of `numpy.linalg.eig`. This function should be used when the transition matrix is not Hermitian (not symmetric if all
  the values are real). This may happen when studying the full state-space in a simplex of higher than 3 dimensions.

### Fixed

- Fixed issue with extending `AbstractGame` in Python.
- Fixed bug in `egttools.utils.get_payoff_function`.

### Changed

- Updated output of `StochDynamics.transition_and_fixation_matrix` so that the correct fixation probabilities are
  returned, instead of `fixation_probabilities/(1/Z)`.
- `draw_stationary_distribution` was also updated so that it expects the fixation probabilities as input and
  not `fixation_probabilities/(1/Z)`.
- Several improvements in documentation and docstrings.

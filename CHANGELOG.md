# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/) and this project adheres
to [Semantic Versioning](http://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- **Function signatures in docstrings and type stubs.** The bindings no longer disable pybind11's
  function signatures, so every docstring starts with the real signature and the `.pyi` stubs list
  real parameters and types (with one `@overload` per variant) instead of `*args, **kwargs`. The stubs
  were regenerated with a pinned `pybind11-stubgen` (2.5.1), and CI fails if they are out of date. In
  the documentation, overloaded functions show their overloads numbered, each with its signature.
- `TimingUncertainty.calculate_end` and `calculate_full_end` take an optional `seed` instead of a C++
  random-number engine, which could not be created from Python, so these methods were not usable
  before. With `seed=None` they draw from the global seed generator (`egttools.Random.init_with_seed`).
- An out-of-range strategy index raises `IndexError`; other invalid arguments (equal invader and
  resident, malformed states) raise `ValueError`. `PairwiseComparisonNumerical.estimate_fixation_probability`
  used to raise `ValueError` for an out-of-range index.

### Changes to numerical results

- **`egttools::binomial_precision` (C++ API).** The intermediate product overflowed 128 bits from
  n = 126 on, so results were wrong even when C(n, k) fits (e.g. C(126, 63)), and results that do
  not fit wrapped around silently. It is now exact for every result below 2^128 and throws
  `std::overflow_error` above. It runs in fixed-width arithmetic while the value is below 2^64 and
  switches to arbitrary precision only for larger results. No Python function uses it.
- **Number of population states.** `PairwiseComparison` computed C(Z + S - 1, S - 1) in 32-bit
  `int`, so `nb_states()` was wrong (negative) from 6 strategies at Z = 136, 8 at Z = 52 and 10 at
  Z = 31, although the true count fits. Every class now computes the count exactly in 64 bits. When
  it does not fit in a signed 64-bit integer, the class still constructs and the methods that do not
  enumerate states (fixation probabilities, edge fitnesses) work; `nb_states` and the methods that
  need every state (transition matrices, stationary-distribution estimates,
  `PairwiseComparisonTransitionOperator`) raise `OverflowError` naming Z and S, instead of using a
  wrapped count (the transition matrix could try to allocate without bound).
  `egttools.calculate_nb_states` is unchanged: it already returned exact integers of any size.
- **Integer binomial coefficients (C++ API).** `egttools::binomialCoeff` with an integer output type
  is now exact or throws `std::overflow_error`; it used to wrap around silently in the intermediate
  product (for 64-bit outputs from n = 63). Floating-point and arbitrary-precision outputs are
  unchanged. The common case stays as fast as before; a slower exact step runs only near the limit.
- **Fitness cache of `PairwiseComparison`.** Fixation probabilities and edge fitnesses identified each
  state by its global index, packed into a 64-bit key with `index << 32`. With many strategies that
  index does not fit in 64 bits (e.g. 29 strategies, Z = 100: about 1e28 states) and was computed
  with wrap-around, and the shift discarded the upper 32 bits of any index. Edge states now have
  their own exact keys (pair of strategies and count), and full-state keys are exact
  (`index * S + strategy`). In every configuration we checked (up to 29 strategies) the old keys did
  not collide, so earlier fixation probabilities were correct there, but this was not guaranteed for
  very large state spaces; recompute such results if in doubt. Fixation probabilities and SML
  matrices are about 28% faster, as they no longer compute global indices.
  `pre_calculate_edge_fitnesses` no longer evaluates fitness in states that do not sum to Z.

### Testing

- The C++ tests now check their results in Release builds: `NDEBUG` is undefined for the test
  programs and the test copy of the library, so `assert()` runs, including the library's own
  precondition checks. A new test fails if assertions are compiled out.
- `test_binomial_coefficient_with_time` compares against exact reference values instead of
  printing them.
- `test_timing_uncertainty` uses a fixed seed and a tolerance of 4 standard errors, and checks that
  a wrong parameter is rejected; its former tolerance (about 2 standard errors, unseeded) failed in
  about 4% of runs once assertions were enabled.
- New CMake option `EGTTOOLS_SANITIZE` (e.g. `-DEGTTOOLS_SANITIZE="address;undefined"`, or
  `thread`) builds the library, the extension and the C++ tests with the compiler's sanitizers,
  for development and CI only. Undefined behaviour aborts the program, so it fails the test that
  triggers it. In such builds, `testSanitizerCanary` checks that each enabled sanitizer reports a
  deliberate error.
- CI runs the C++ tests under AddressSanitizer (with leak detection) and
  UndefinedBehaviorSanitizer on Linux with clang. The CMake target `egttools_cpp_tests` builds
  every C++ test program registered on the platform.
- CI and the pre-push hook run `scripts/check_repo_hygiene.py`, which checks that no tool or
  environment files, absolute home paths or credentials are tracked; `--artifacts` checks built packages.
- `scripts/memguard.py` runs a command and kills it if its memory (the whole process tree)
  exceeds a limit, for test runs that may allocate without bound.

## [0.2.0] - 2026-11-19

First release since 0.1.14. Several fixes change numerical results; read
*Changes to numerical results* first if you have results computed with earlier versions.

### Changes to numerical results

- **Monte Carlo estimates of the pairwise comparison process.** In homogeneous populations the
  simulation waited one update too long before each mutation. `estimate_stationary_distribution`,
  `estimate_stationary_distribution_sparse` and `estimate_stationary_indicators` overweighted
  monomorphic states by a factor of about 1/(1−μ), `estimate_strategy_distribution` skipped the
  state reached by each such mutation, and `run_with_mutation` mutated at rate μ/(1+μ) in
  homogeneous states. Waiting times that run past the end of a simulation window no longer
  trigger a mutation, which affected the final state of `evolve` and the start of the counting
  phase. The effect grows with μ.
- **Gradient of selection.** `PairwiseComparison.calculate_gradient_of_selection` was scaled by
  2/k for k ≥ 3 strategies, and `calculate_gradient_of_selection_with_mutation` weighted mutation
  and selection inconsistently (wrong for every k when μ > 0). Both now return the expected change
  in strategy counts per update, consistent with `calculate_transition_matrix`. Without mutation,
  directions and roots are unchanged. `StochDynamics.full_gradient_selection{,_without_mutation}`
  drew the role model from all Z individuals instead of the other Z−1, and its mutation term did
  not conserve the population size.
- **Fixation probabilities at strong selection.** The product accumulation could exceed an
  early-exit threshold in non-monotone fitness landscapes (e.g. coordination games at β ≥ 5) and
  return ρ = 0 for strategies with non-zero fixation probability, which collapsed small-mutation
  stationary distributions to uniform. Fixation probabilities are now computed in log space
  (`PairwiseComparison` and `StochDynamics`).
- **Expected indicators.** The `calculate_expected_*` and `calculate_strategies_distribution`
  bindings only used the first state when the stationary distribution was passed as a column
  (nb_states × 1) sparse matrix; both orientations are now accepted.
- **`AbstractNPlayerStateGame`.** Fitness used the wrong state index for populations larger than
  the group, mapping most states to state 0.
- **Network gradient of selection with `LinearProportional`.** `estimate_agos` returned zeros for
  this synchronous rule; it now estimates the gradient from the observed one-generation change.
- **`calculate_stationary_distribution_non_hermitian`** could return a wrong distribution: it used
  a Schur vector as the eigenvector, which is only valid when eigenvalue 1 comes first in the
  Schur form.
- **Simplex plots.** `calculate_stability` ignored the direction of each perturbation, so stable
  vertices were reported as unstable, stable edge points never as stable, and unstable interior
  points as saddles; this affected the stability markers of `plot_replicator_dynamics_in_simplex`
  and `plot_pairwise_comparison_rule_dynamics_in_simplex`. The pairwise comparison plots also
  evaluated the gradient at invalid states (rounded down to fewer than Z individuals, or
  fractional states truncated to integers); they now interpolate the exact gradient between
  population states and find equilibria as its exact zeros.

### Added

- **Monte Carlo estimation** (`PairwiseComparisonNumerical`): `estimate_stationary_indicators`
  for expected values of state or group indicators (evaluated directly on the population state
  when a precomputed matrix would be too large, `precompute_limit`),
  `estimate_mean_absorption_time`, `estimate_absorption_probabilities`, a working
  `estimate_gradient_of_selection`, optional early stopping (`tolerance`, `check_every`), biased
  mutation (`set_mutation_matrix`, `set_mutation_weights`), `change_game`, a per-instance `seed`,
  and a warning when μ is too small for the simulated window.
- **Estimator wrappers** `PairwiseComparisonEstimator` and `NetworkEstimator`: result objects
  (`FixationResult`, `StationaryDistributionResult`, `AGoSResult`, …) with confidence intervals,
  chunked runs with standard errors, an optional progress bar, and `checkpoint_path` to resume
  long runs. `egttools.Random.get_state` / `set_state` capture the full generator state.
- **Stationary distributions and fixation**: the matrix-free
  `PairwiseComparisonTransitionOperator` with scipy `LinearOperator` wrappers
  (`egttools.numerical.linear_operator`, `stationary_distribution_from_sparse`); native power
  iteration (`compute_stationary_distribution`) and ARPACK (`compute_stationary_arpack`, built
  with `EGTTOOLS_ENABLE_ARPACK`); an MPI-distributed PETSc/SLEPc solver
  (`egttools.numerical.mpi`, built with `EGTTOOLS_ENABLE_PETSC`);
  `calculate_log_fixation_probability`, a high-precision (Boost) fixation probability, and
  log-space small-mutation matrices (`calculate_transition_and_log_fixation_matrix_sml`, and the
  `StochDynamics` equivalents); `egttools.utils.calculate_spectral_gap`; and the
  `calculate_expected_*` indicator and hypergeometric utilities.
- **Games**: `AbstractNPlayerStateGame` for payoffs that depend on the population state, and
  `AbstractReplicatorGame` for infinite populations.
- **Games on networks**: update rules (pairwise comparison, birth–death, death–birth,
  time-dependent pairwise comparison, linear proportional), Monte Carlo estimators for fixation,
  strategy distributions and the average gradient of selection, co-evolutionary networks with
  random and homophilic rewiring, `LocalRedistributionGame`, network games subclassable in
  Python, a parallel parameter sweep (`run_network_sweep`), and `egttools.plotting.network_plots`.
- **Multi-level selection**: `MLSTraulsen` and `MLSGarcia`.
- **Plotting**: `Simplex3D` for four-strategy simplices (plotly); discrete stationary
  distributions in `Simplex2D`; mutation in the pairwise comparison simplex plots (`mu`);
  `invasion_graph` and named invasion-diagram styles (`INVASION_DIAGRAM_STYLES`);
  `lattice_gradient_interpolator` and `find_roots_of_lattice_gradient`; and colormaps (`cmap`)
  in the gradient plots and the `Simplex2D` trajectory methods, with matplotlib colormaps also
  accepted by `Simplex3D`.
- **Build information**: `is_arpack_enabled`, `is_petsc_enabled` and `is_boost_enabled`, and a
  runtime section in `show_build_info`.
- **Packaging and documentation**: conda packages (channel `socrats`, with ARPACK and MPI
  variants), type stubs and `py.typed`, a theory page specifying the pairwise comparison process
  with tests that check every implementation against it, and community files (issue forms, pull
  request template, `SECURITY.md`, `SUPPORT.md`, `RELEASING.md`, a rewritten `CONTRIBUTING.md`).

### Changed

- **Breaking:** `draw_invasion_diagram(graph, ax=None, ...)` draws a graph built by
  `invasion_graph` and returns the matplotlib axes. The old signature mixed classification with
  drawing, needed the neutral fixation probability passed by hand (`drift`) and keyed nodes by
  label (merging strategies with equal names). Migration: `draw_invasion_diagram(labels, 1/Z, fp,
  sd, ...)` becomes `draw_invasion_diagram(invasion_graph(fp, Z, sd, labels), ...)`;
  `min_strategy_frequency` becomes `invasion_graph(..., min_frequency=...)` and `atol` becomes
  `invasion_graph(..., drift_tolerance=...)` (default 0.005, so ratios shown as 1.00 count as
  drift). Each pair of strategies is classified once (selection, drift or none); pairs that invade
  each other are drawn as two curved arrows. New options include `style` (`"simple"` by default,
  `"proportional"`, `"classic"`, `"compact"`, or a dict), `layout`, `colors` (palette name,
  colormap, sequence or dict), label placement (`name_position`, `frequency_position`,
  `label_offsets`), `frequency_format`, node and edge scaling, `min_ratio`, and keyword dicts for
  every element.
- `PairwiseComparisonNumerical.run`, removed earlier, now raises `NotImplementedError` naming
  `run_with_mutation` and `run_without_mutation`, instead of warning and returning `None`.
- Python 3.11–3.14 are supported; Python 3.10 (end of life in October 2026) is no longer
  supported. Wheels are built for Linux x86_64, macOS arm64 (macOS 15 or later) and Windows
  x86_64 and arm64; macOS x86_64 wheels are no longer built.
- Versions are derived from git tags (setuptools-scm).
- `calculate_transition_matrix` is parallelised with OpenMP, and game fitness computations share
  faster hypergeometric helpers.
- `calculate_stationary_distribution{,_non_hermitian}` keep sparse input sparse (solved with a
  sparse direct solver instead of a dense eigendecomposition), accept `scipy.sparse` arrays, and
  warn when the chain has no unique stationary distribution.
- Repository clean-up: unused build scripts, `setup.cfg`, an obsolete installation script and a
  broken `Dockerfile` were removed; `environment.yml`, `requirements.txt` and
  `CMakePresets.json` were updated; Docker images now live in `docker/memcheck/` and
  `docker/mpi/`.
- Package metadata declares the license as an SPDX expression (`GPL-3.0-or-later`, with
  `license-files`), which newer setuptools requires; building now needs setuptools ≥ 77.

### Fixed

- Crash when a Python game passed to `PairwiseComparison` was garbage-collected (wrong
  `keep_alive` in the bindings).
- Deadlock on Python 3.13 when building transition matrices for games subclassed in Python.
- `PairwiseComparisonNumerical`: `change_game` had no effect; `run` without mutation wrote
  incomplete rows after fixation; two out-of-bounds row writes in the `run` methods with a
  transient period; an integer overflow in `evolve` for very small μ; a race condition in the
  seed generator.
- `ThreadSafeLRUCache` with size 0 aborted instead of acting as disabled.
- `plot_pairwise_comparison_rule_dynamics_in_simplex` could crash on states with negative
  entries.
- `Simplex3D.draw_streamlines` ignored `cone_scale`.
- Example notebooks used removed APIs (`egt.Random._seed`, `PairwiseComparisonNumerical.run`).
- Builds: Windows wheels (a POSIX-only type), wheels labelled `0.0.0`, source distributions
  missing files, OpenMP builds with GCC 14 and Apple Clang 21, and ARPACK detection on HPC
  (EasyBuild) systems.

## [0.1.14] - 20-07-2025

### Added

- Added support for building **universal macOS wheels** and **ARM builds** for Linux and Windows.
- Added fallback logic to skip `OpenMP` when not supported (e.g., Windows/macOS universal).
- Added automatic installation of dependencies via `vcpkg`, including caching and toolchain integration.
- Introduced modular CMake configuration using `CMakePresets.json` and modular `EnableXXX.cmake` utilities.
- Added `Dockerfile` to support containerized builds and local testing.
- Implemented efficient factorial calculation in C++.

### Changed

- Refactored build system for maintainability and CI compatibility:
    - CI workflows now support `cibuildwheel` v2.23.3 and manually triggered docs builds.
    - Moved doc generation to separate `docs.yml` workflow and switched to `furo` theme.
    - Enforced Python 3.10+ as the minimum supported version.
- Improved README and build documentation.
- Updated pybind11 and vcpkg submodules to latest versions.
- Simplified `setup.py` and enforced proper `MACOSX_DEPLOYMENT_TARGET` across CI environments.
- Made BLAS/LAPACK optional and adjusted linking strategy across platforms (including Accelerate on macOS).

### Fixed

- Fixed multiple CI build issues related to:
    - `libomp` installation and linking.
    - `vcpkg` initialization and caching.
    - RTD (ReadTheDocs) upload and broken builds.
- Fixed memory management bugs in `pybind11` bindings (e.g., garbage collection of `NDArray` objects).
- Fixed initialization bugs in simulation state (deep copy and move semantics added to `sample_simplex` and related
  methods).
- Resolved `Boost` installation issues on some platforms.

## [0.1.13-patch5] - 22-11-2023

### Fixed

- Fixed state initialization in `NetworkSync` and `NetworkGroupSync`. The state was not being randomly initialized.
- Fixed incorrect use of `sample_without_replacement`. It must only be used when the sample is smaller than the
  population.

### Changed

- Exposed the expected payoffs of `NormalFormNetworkGame` to Python

## [0.1.13-patch4] - 12-11-2023

### Fixed

- Fixed segfault in `PairwiseComparisonNumerical` run method. when called using a `transient` period, the function
  surpass
  the array index limit, thus causing a bad memory access and a segfault

### Added

- Added a version of run with transient and no mutation
- Added new argument checks

### Changed

- Changed the generation and transient parameter types from `int` to `int_fast64_t`
- The methods that have a mutation parameter enforce that mu>0 to avoid errors
- There will be a bad argument error if the beta parameter used in the python call is not explicitly a float

## [0.1.13-patch3] - 9-11-2023

### Fixed

- fixed `NetworkSync` and `NetworkGroupSync` structures updates

## [0.1.13-patch2] - 30-9-2023

### Fixed

- fixed issue with dynamic linking of `OpenMP`

### Changed

- dropped support for `OpenMP` on macOS temporally
- now OpenMP is linked statically again for linux

## [0.1.13] - 29-9-2023

### Fixed

- fixed issue with environmental variables not working on MacOS builds in Github Actions
- fixed gradient of selection estimation algorithm
- fixed issue with multiple `libomp.dylib` copies loading
- fixed issue with the citation of current version of egttools
- fixed link to anaconda documentation

### Changed

- now openmp is linked dynamically to avoid conflicts on MacOS
- increased minimum required cmake version to `3.18`
- changed int to `int_fast64_t` to support larger number of generations

### Added

- added NetworkSync evolver
- added more support for network simulations
- added Cache to accelerate `PairwiseComparison`
- added calculation of the averaged gradient of selection
- added OneShotCRDNetworkGame

## [0.1.12.patch1] - 7-07-2023

### Fixed

- Fixed issue with link to Simplex2D class
- Match scatter output to matplotlib API changes [by cvanelteren]
- Fixed issue with compiling `Distributions.cpp` without Boost

### Changed

- Removed support for `Python 3.7`
- Dropped support for `Win32` architectures
- Removed `MACOSX_DEPLOYMENT_TARGET` constraint
- Updated egttools citation
- Skip `multivariate_hypergeometric` tests when compiling without Boost
- Upgraded cibuildwheel to v2.14.0
- Set minimum Boost version to 1.70.0

### Added

- Added basic unittests for plotting [by cvanelteren]

## [0.1.12] - 15-12-2022

### Fixed

- Fixed stability plot and minor syntax issues.
- Fixed issue with fitness calculations of strategies with 0 counts in the population.
- Fixed LNK2005 error on windows.
- Fixed version naming convention to adhere to PEP440.

### Changed

- Changed name of `PairwiseMoran` class to `PairwiseComparisonNumerical`.
- Changed function and class names that used "moran" to "pairwise_comparison_rule".
- Removed signatures of overloaded methods in docstrings.
- Moved all headers for pybind11 to .hpp files to avoid issues on windows.
- Updated pybind11 version to 2.10.
- Removed specialization of `binomialCoeff` to avoid issues on Windows.
- Refactored the binding code from C++ to Python so that now we use different files for defining each Python submodule.
  This makes the code much more clear. These files can be found in `cpp/src/pybind11_files`.
- Now the GitHub CI builds download and install Boost C++ library, as it is required to build egttools.
- Dropped support for manylinux_i686. In the future only 64 bit architectures will be supported.

### Added

- Added support for multiprecision types with boost::multiprecision.
- Added replicator equation for n-player games (implementation in c++).
- Added `replicator_equation` implementation in C++.
- Added a new C++ implementation of the analytical stochastic dynamics named `PairwiseComparison`.
- Added `Matrix2PlayerGameHolder` and `MatrixNPlayerGameHolder` classes, so that a game can be defined using a payoff
  matrix.
- Added several tests (coverage now is ~50%).
- Added tutorials to the documentation.

## [0.1.11.patch2] - 16-11-2022

### Fixed

- Fixed error in `calculate_full_transition_matrix`. The previous implementation tried to calculate the fitness for
  strategies with 0 counts in the population. This would lead to an error when calculating the probability of group
  forming. Now, we calculate directly the probability of a strategy with 0 counts increasing by 1 individual in the
  population, i.e., prob = (n_decreasing_strategy / population_size) * (mu / (1 - nb_strategies)).

## [0.1.11.patch1] - 10-11-2022

### Fixed

- Fixed issue with changing mutation rate in StochDynamics. When the mutation rate was updated
  using `evolver.mu = new_value`, the variable `evolver.not_mu` was not udpated. This variable was used
  by `calculate_full_stationary_distribution` and it created an error. Now `not_mu` is calculated inside
  of `calculate_full_stationary_distribution`.
- Fixed issue with `binomialCoeff` implemented in c++. In some cases, the intermediary calculations were overflowing
  an `uint64_t` type when calling it from `multivariateHypergeometricPDF`. Now, we no longer calculate the exact value
  in `multivariateHypergeometricPDF`, instead use double types. This will be improved in the new version by
  using `boost/multiprecision` `uint128_t` types.

## [0.1.11] - 25-10-2022

### Fixed

- fixed errors in docstrings examples
- fixed missing headers
- fixed error in `full_fitness_difference_group` and `calculate_fulll_transition_matrix`. There was a missing
  multiplying factor in the probability of transition due to mutation. The probability of selecting the strategy to dies
  must also be taken into account. There was also an issue when instantiating the `multi_hypergeometric_distribution`
  class from scipy. It does not copy the array containing the counts of each strategy. Now we create a copy before
  passing the state vector to avoid the issue.
- fixed issue with `AbstractNPlayerGame`. For N-player games it was not a good idea to calculate the fitness in Python
  as this part of the class becomes a bottleneck, as it will be called more often then in the 2-player case (because
  there are more states - so less likely the fitness will be stored in cache). For this reason we now implemented this
  abstract class in C++ and the fitness calculation is done in this language. Everything else remains the same, and it
  should be equally easy to create new games.
- fixed issues of missing initialization of internal parameters of some `egttools.games.NormalForm.TwoAction`
  strategies. Internal parameters should be initialized when `time == 0`, i.e., at the beginning of each game.

### Changed

- changed `egttools` to `src-layout`. This should help fix issues with tests and make the overall structure of the
  library cleaner.
- moved C++ code to the `cpp` folder. This way the code is more organized.
- Bump pypa/cibuildwheel from 2.8.1 to 2.11.1
- Bump pypa/cibuildwheel from 2.11.1 to 2.11.2
- Bump robinraju/release-downloader from 1.4 to 1.5
- Bump ncipollo/release-action from 1.10.0 to 1.11.1
- removed support for win32 and manylinux_i686 for Python > 3.7

### Added

- added new controls to `draw_stationary_distribution`
- added `enhancement_factor` parameter to `CRDGame`. This parameter serves as a multiplying factor for the payoff of
  each player in case the target is achieved. If `enhancement_factor = 1` the `CRDGame` behaves as usual.
  For `enhancement_factor > 1`, we are incentivizing strategies that reach the target.
- added `MemoryOneStrategy` to `egttools.games.NormalForm.TwoAction` strategies.
- added `CommonPoolResourceDilemma` game - However it has still not been extensively tested!!
- added `ninja` as a requirement for the build.
- added `TimeBasedCRDStrategy` to `egttools.games.CRD` strategies. These strategies make contributions to the Public
  Good in function of the round of the game.
- added `sdist` to build.
- added labels to the lines plot by `plot_gradients` so that several lines can be plotted.
- added more unit testing, but this still needs a lot of improvement.
- added missing libraries on C++ code.

## [0.1.10] - 06-08-2022

### Fixed

- fixed issue with the comparison of different types in `PairwiseMoran`
- fixed issue with colorbar so that now the axis is passed to maptlolib colorbar, this way it will be plotted correctly
  when drawing multiple subplots
- fixed issue with hardcoded x-axis in `plot_gradients`
- fixed error on `calculate_full_transition_matrix`. The error happened when calculating the transition probability:
    - a) although the literature is a bit confusing on this, the original paper by Traulsen 2006 says that for the
      transition we consider that the strategies to reproduce and die are pricked simultaneously (so both with
      probability 1/Z).
    - b) the is more accumulated numerical error when doing probability of transitioning from A to B P(A, B) = fermi(
      -beta, B - A) than when doing fermi(beta, A - B). This is probably specific to Python and Numpy, but must be taken
      into account in the future.
    - c) The schur decomposition (egttools.utils.calculate_stationary_distribution_non_hermitian) works better in this
      case (although still has a slight numerical error) and should be used for full transition matrices).
- normalized transition probabilities to use the definition in Traulsen et al. 2006
    - now we assume that both death and birth individuals are selected simultaneously with probability n_i/Z, where n_i
      is the number of individuals of that strategy in the population

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

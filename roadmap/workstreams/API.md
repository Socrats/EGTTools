# API: API evolution

> LLMs have been used to create this list automatically based on the project definitions of @Socrats.

Goal: make module names describe what the methods do, without breaking users
abruptly.

| ID | Task | Status | Pri | Size | MS | Deps | Ref |
|----|------|--------|-----|------|----|------|-----|
| API-01 | Rename `analytical` → `exact`, `numerical` → `inexact`; class renames are API-10 | todo | P1 | L | 0.4 | — | — |
| API-02 | Deprecation shims for old module paths (`DeprecationWarning`, removal ≥ 2 minor releases later) | todo | P1 | M | 0.4 | API-01 | — |
| API-03 | Deprecation policy written down in `CONTRIBUTING.md` | todo | P1 | S | 0.4 | — | — |
| API-04 | Review hard dependencies (seaborn, plotly) and move plotting extras to optional | todo | P2 | S | 0.4 | — | — |
| API-05 | Make removed methods raise instead of warning and returning `None` (e.g. `PairwiseComparisonNumerical.run`) | done | P1 | S | 0.2 | — | branch `api-05-removed-methods-raise` |
| API-06 | One transition-matrix orientation: `StochDynamics.transition_and_fixation_matrix` returns it transposed (column-stochastic), the C++ SML version row-stochastic | todo | P1 | S | 0.3 | — | — |
| API-07 | Vector-quality 3D simplex plots: add a matplotlib (mplot3d) backend to `Simplex3D`, sharing its geometry code; keep plotly for interactive figures | todo | P2 | M | 0.4 | — | — |
| API-08 | Assess the random-number and seeding design against NumPy and JAX; decide the target design and file the migration tasks | todo | P1 | M | 0.3 | — | — |
| API-09 | Real signatures in the `.pyi` stubs: stop disabling pybind11 function signatures, regenerate the stubs, check them in CI | done | P1 | M | 0.3 | — | `8535760b` |
| API-10 | Rename `PairwiseComparison` and `PairwiseComparisonNumerical` to names that say exact vs approximate (estimators), with deprecation aliases | todo | P1 | M | 0.4 | API-01 | — |
| API-11 | Audit the pybind11 bindings' declared types: argument types and conversions, returned dtypes, Eigen `Ref` vs copies, overloads, defaults, argument names, GIL release | todo | P1 | L | 0.3 | MEM-10 | — |
| API-12 | Replace the unprefixed macros in `Distributions.h` (`int_type_`, `uint_type_`, `float_type_`, `binomial_coeff_`, `CONVERT_TO_`) with type aliases and functions in `egttools::` | todo | P3 | S | 0.4 | CI-15 | — |
| API-13 | `NPlayerStagHunt` (C++ and Python): defectors in mixed groups get k F / N instead of k F c / N; fix the payoffs and the docstring (copied from `OneShotCRD`) | todo | P1 | S | 0.3 | TEST-09 | — |
| API-14 | Deprecate `StochDynamics` (Python, legacy): `DeprecationWarning` on construction, docs point to `PairwiseComparison` | done | P2 | S | 0.3 | — | — |

## Notes

- **API-14**: decided 2026-11-27: `StochDynamics` is kept only for backward compatibility and is
  no longer listed on the theory pages. The warning uses `stacklevel=2`, so it names the caller's
  line; `analytical.utils.check_if_there_is_random_drift` still builds one internally (with
  `population_size`) and suppresses the warning. Its spec tests stay; pytest ignores the warning
  except in `test_stoch_dynamics_is_deprecated`. The example notebooks still use it (DOC-14 should
  move them to `PairwiseComparison`). Removal (later): port the drift helper, drop the class, and
  API-06 then only needs a CHANGELOG note.

- **API-13**: Pacheco, Santos, Souza & Skyrms 2009 (Proc. R. Soc. B 276, 315): Pi_D(k) =
  k F c / N theta(k - M), Pi_C(k) = Pi_D(k) - c. `NPlayerStagHunt::play` (and the Python class in
  `egttools/games/nonlinear_games.py`) gives defectors k F / N in mixed groups, while the
  all-cooperator group gets c (F - 1); they agree only for c = 1. CHANGELOG under "Changes to
  numerical results". Remove the `[!shouldfail]` tag in `tests/cpp/test_games.cpp`.

- **API-08**: current design (2026-11-19): a process-global `SeedGenerator` singleton
  (`egttools.Random`), seeded once and drawn from at 44 call sites in 17 C++ files; each
  draw seeds an `std::mt19937_64` from a single 64-bit value. Only a few classes accept a
  `seed` argument (e.g. `PairwiseComparisonNumerical`, the network structures). The
  Python layer mixes three approaches: the global `np.random` state (`plotting/helpers.py`,
  `plotting/simplex2d.py`), an unseeded `np.random.default_rng()`
  (`numerical/indicators.py`), and `default_rng(Random.generate())` (`nfg_strategies.py`).
  Compare with:
  - NumPy: explicit `Generator` objects, `SeedSequence` for high-entropy seeding and
    `spawn()` for independent streams, and the Scientific Python SPEC 7 convention of an
    `rng` argument that accepts `None`, an int or a `Generator`;
  - JAX: explicit, functional keys (`split`, `fold_in`) with counter-based generators, so
    a parallel stream is a deterministic function of the key and its index.
  Questions to answer: explicit `rng`/`seed` arguments versus the global singleton;
  stream derivation for parallel runs (PERF-04's per-run seeding fix uses the result);
  counter-based engines (Philox, as in NumPy and JAX) versus `mt19937_64`; whether the C++
  layer can accept a NumPy `SeedSequence` or `Generator` state; saving and restoring RNG
  state for checkpoints (MC-10). Compatibility: `egttools.Random` keeps working unless the
  CHANGELOG states a reason to change it. Deliverable: a design note with the decision in
  `DECISIONS.md`, and one task per migration step. DOC-06 documents the result.

- **API-07**: plotly and its static exporter (Kaleido) are open source (MIT). The limitation is
  that 3D scenes are rendered with WebGL, so SVG/PDF exports contain them as raster images.
  matplotlib's mplot3d writes true vector output and covers what `Simplex3D` draws (edges,
  streamlines, arrows, points); its limitation is per-artist depth sorting. PyVista/VTK
  (`save_graphic`) is the alternative if lit surfaces are needed.
- **API-09**: `options.disable_function_signatures()` in `methods.cpp` (4 blocks) and
  `distributions.cpp` removes the signature line from the docstrings, so `pybind11-stubgen` writes
  `*args, **kwargs` (and wrong `@staticmethod`s) for most methods. Re-enable signatures, check that the
  Sphinx pages still render (autodoc reads the same first line), regenerate the stubs
  (`pybind11-stubgen egttools.numerical.numerical_ -o src`, CONTRIBUTING.md), and add a CI step that
  regenerates them and fails on a diff.
  Done (2026-11-19, option A agreed): signatures re-enabled everywhere; `docs/pybind11_docstrings.py`
  numbers the overloads of overloaded functions (each headed by its signature, `(...)` in the title)
  and shortens the stub-style types for display; docs warnings 1891 → 1634 (unresolved references
  1707 → 1416). Stubs regenerated with `pybind11-stubgen==2.5.1` (the committed ones were stale, from
  an older generator); `*args` stubs left: pybind11's internal `_pybind11_conduit_v1_` and the removed
  `PairwiseComparisonNumerical.run`. CI step "Check that the type stubs are up to date". The
  `TimingUncertainty` sampling methods took a C++ `std::mt19937_64&` (unusable from Python, and its
  type name differed between libc++ and libstdc++, so stubs differed per platform); they now take an
  optional `seed` (API-08 may revise this).
- **API-10**: both classes are numerical methods; the distinction is exact methods
  (`PairwiseComparison`) vs approximate methods and estimators (`PairwiseComparisonNumerical`). Names to
  be decided in the task; old names stay as deprecated aliases for at least two minor releases (API-02,
  API-03).
- **API-11**: complements MEM-05 (ownership: return value policies, `keep_alive`), MEM-10 (C++
  types) and API-09 (stubs). Found 2026-11-19:
  - A negative value for a `size_t` parameter, or a value above 2^31 for an `int` one, raises
    "incompatible constructor arguments" (`TypeError`), which blames the type, not the value.
    Decide per argument: 64-bit signed types (MEM-10) and a `ValueError` with the valid range.
  - Population states accept floats and truncate them silently: `[3.5, 3.5, 4.0]` with Z = 10 is
    accepted as `[3, 3, 4]`. Count arrays should not convert from floats (`py::arg().noconvert()` or
    an explicit dtype check).
  - Counts are returned as `uint64` arrays (e.g. `sample_simplex`), so `state - 1` wraps around in
    NumPy (the TEST-13 helper bug). Return signed 64-bit counts, consistent with MEM-10.
  Also check: Eigen `Ref<const T>` arguments whose storage order differs from NumPy's default
  (silent copies of large matrices), overload order and ambiguity, Python defaults matching the C++
  ones, `py::arg` names matching the docstrings, and `py::call_guard<py::gil_scoped_release>` on long
  computations that do not call back into Python (PERF-04). One test per finding.
- **API-12**: `Distributions.h` is a public header, so its macros leak into every file that
  includes it, including C++ users' code; 15 uses in `cpp/`. Found during CI-15, which made
  them unconditional (they used to switch between Boost and plain types).

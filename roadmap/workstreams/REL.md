# REL: Release v0.2.0

> LLMs have been used to create this list automatically based on the project definitions of @Socrats.

Goal: publish everything merged since 0.1.14 (about 80 commits) with CI green
and the known numerical bugs fixed. PyPI is still at 0.1.14.2.

| ID | Task | Status | Pri | Size | MS | Deps | Ref |
|----|------|--------|-----|------|----|------|-----|
| REL-01 | Clean up `pyproject.toml` configuration | done | P0 | S | 0.2 | — | merge `83d6789` |
| REL-02 | Move non-test scripts out of `tests/` | done | P1 | S | 0.2 | — | merge `5c43eaf` |
| REL-03 | Fix `calculate_stationary_distribution` (+ `_non_hermitian`) | done | P0 | M | 0.2 | — | merge `233473c` |
| REL-04 | Fix fractional-state cast in simplex gradient plotting | done | P1 | M | 0.2 | REL-10 | merge `3050629` |
| REL-05 | Diagnose and fix failing wheel jobs (sdist, win_amd64, win_arm64) | review | P0 | M | 0.2 | — | merges `4aee21d`, `754c5f6`, `75b426a`, `fc50406`; awaiting version check |
| REL-06 | Support Python 3.11–3.14 (add cp314, drop 3.10) | done | P1 | S | 0.2 | — | merge `8f5b901` |
| REL-07 | Add a fast PR test workflow (pytest + CTest) | todo | P0 | M | 0.2 | — | — |
| REL-08 | Write CHANGELOG for 0.2.0 | todo | P0 | M | 0.2 | REL-01..07 | — |
| REL-09 | Tag `v0.2.0`, publish GitHub release and PyPI | todo | P0 | S | 0.2 | REL-08 | — |
| REL-10 | Specify the pairwise comparison process and verify every implementation against it | done | P0 | L | 0.2 | — | merge `50d98c4` |
| REL-11 | Apply the homogeneous-state fix to `ImitationMultipleGames` (not exposed to Python) | todo | P3 | S | later | REL-10 | — |

## Notes

- **REL-01**: remove the stray entry in `[tool.check-manifest].ignore` that names
  a local-only folder. Remove the `[tool.scikit-build]` table: it is
  scikit-build-core syntax and the current backend (scikit-build 0.x) ignores
  it. Merge `pytest.ini` into `[tool.pytest.ini_options]`. Today `pytest.ini`
  takes precedence, so the `addopts` in `pyproject.toml` never apply. Keep the
  `slow` marker.
- **REL-02**: *Done:* the two glibc reproducers moved to `scripts/diagnostics/`;
  `tests/somthing.py` untracked again (committed by accident in `96f818a`).
- **REL-03**: `utils.calculate_stationary_distribution` converts sparse input
  to dense and runs a full `np.linalg.eig` (O(n³)).
  `calculate_stationary_distribution_non_hermitian` takes a column of the Schur
  basis as the eigenvector. That is only correct when eigenvalue 1 is the first
  diagonal entry of the Schur form. Fix both: keep sparse input sparse, and use
  a proper left-eigenvector or linear-system solve. Add tests against chains
  whose stationary distribution is known in closed form. Keep the public
  signatures; the full solver API is `SOL-01`.
  *Done:* Schur form reordered; sparse input solved via a reduced sparse system
  (SuperLU), ARPACK fallback for reducible chains; `scipy.sparse` arrays accepted.
  The plain `eig` path was already correct.
- **REL-04**: `plotting/simplified.py` lines ~413/416 pass fractional states to
  `calculate_gradient_of_selection*`. NumPy warns "invalid value encountered in
  cast", which suggests truncation to integers. First confirm whether the
  plotted field is affected, then fix it.
  *Findings:* both plotted outputs are affected. `scipy.optimize.root` evaluates the
  gradient at continuous points that get truncated to integers, so roots,
  stability and the returned trajectory function are unreliable. The arrow field
  uses `floor`, which yields states summing to Z−1 or Z−2 at most grid points.
  The same code is in the `_without_roots` variant.
  *Decision:* option A, i.e. interpolate the exact lattice gradient (Definition 5.1)
  linearly between lattice states, and use valid lattice states for the arrows.
  *Done:* exact per-triangle roots; `calculate_stability` rewritten (it inverted stable
  vertices and never found stable edge points, which also affected replicator plots).
- **REL-05**: *Causes found:* `ssize_t` (POSIX-only) in `structure.cpp` broke MSVC
  since May; `check-manifest` failed on untracked-in-sdist files. Also fixed: CI
  shallow clone gave version `0.0.1.dev1` (now `fetch-depth: 0`), and `include *.md`
  could pull local Markdown files into a locally built sdist.
- **REL-07**: the current tests run only inside cibuildwheel. Add a separate
  workflow on `push` (all branches) and `pull_request` that builds once (Linux, one Python), runs pytest
  (skipping `slow`), and runs the C++ tests registered in `tests/CMakeLists.txt`.
  This is a prerequisite for branch protection (`PM-09`).
- **REL-05** (cont.): fetching tags exposed that CMake's `project(VERSION)` rejected
  the `v0.1.14-patch2` tag; it now uses the numeric prefix. `v0.1.15` exists only
  locally (never pushed). Wheels stayed `0.0.0` until `setup.py`, whose explicit
  setuptools-scm arguments override `pyproject.toml`, got the same `tag_regex`.
- **REL-10**: spec in `docs/theory/pairwise_comparison.md`, tests in
  `tests/test_pairwise_comparison_equations.py`. Fixed: off-by-one waiting time in
  homogeneous states across all MC estimators (bias ≈ 1/(1−μ) on monomorphic
  states), an out-of-bounds row write in `run_with_mutation` with a transient, the
  2/k scaling and mutation term of the C++ gradient, and `StochDynamics`' gradient
  (Z vs Z−1, non-conserving mutation term).
- **REL-09**: the existing `v0.1.15` tag was never released. Leave it in place;
  0.2.0 supersedes it.

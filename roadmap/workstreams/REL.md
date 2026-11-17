# REL: Release v0.2.0

> LLMs have been used to create this list automatically based on the project definitions of @Socrats.

Goal: publish everything merged since 0.1.14 (about 80 commits) with CI green
and the known numerical bugs fixed. PyPI is still at 0.1.14.2.

| ID | Task | Status | Pri | Size | MS | Deps | Ref |
|----|------|--------|-----|------|----|------|-----|
| REL-01 | Clean up `pyproject.toml` configuration | review | P0 | S | 0.2 | — | branch `rel-01-pyproject-cleanup` |
| REL-02 | Move non-test scripts out of `tests/` | todo | P1 | S | 0.2 | — | — |
| REL-03 | Fix `calculate_stationary_distribution` (+ `_non_hermitian`) | todo | P0 | M | 0.2 | — | — |
| REL-04 | Fix fractional-state cast in simplex gradient plotting | todo | P1 | S | 0.2 | — | — |
| REL-05 | Diagnose and fix failing wheel jobs (sdist, win_amd64, win_arm64) | todo | P0 | M | 0.2 | — | — |
| REL-06 | Add CPython 3.14 wheels | todo | P1 | S | 0.2 | REL-05 | — |
| REL-07 | Add a fast PR test workflow (pytest + CTest) | todo | P0 | M | 0.2 | — | — |
| REL-08 | Write CHANGELOG for 0.2.0 | todo | P0 | M | 0.2 | REL-01..07 | — |
| REL-09 | Tag `v0.2.0`, publish GitHub release and PyPI | todo | P0 | S | 0.2 | REL-08 | — |

## Notes

- **REL-01**: remove the stray entry in `[tool.check-manifest].ignore` that names
  a local-only folder. Remove the `[tool.scikit-build]` table: it is
  scikit-build-core syntax and the current backend (scikit-build 0.x) ignores
  it. Merge `pytest.ini` into `[tool.pytest.ini_options]`. Today `pytest.ini`
  takes precedence, so the `addopts` in `pyproject.toml` never apply. Keep the
  `slow` marker.
- **REL-02**: `tests/somthing.py`, `tests/check_run.py` and
  `tests/glibc_memory_violation.py` are scripts, not tests. Move them to
  `scripts/` or delete them.
- **REL-03**: `utils.calculate_stationary_distribution` converts sparse input
  to dense and runs a full `np.linalg.eig` (O(n³)).
  `calculate_stationary_distribution_non_hermitian` takes a column of the Schur
  basis as the eigenvector. That is only correct when eigenvalue 1 is the first
  diagonal entry of the Schur form. Fix both: keep sparse input sparse, and use
  a proper left-eigenvector or linear-system solve. Add tests against chains
  whose stationary distribution is known in closed form. Keep the public
  signatures; the full solver API is `SOL-01`.
- **REL-04**: `plotting/simplified.py` lines ~413/416 pass fractional states to
  `calculate_gradient_of_selection*`. NumPy warns "invalid value encountered in
  cast", which suggests truncation to integers. First confirm whether the
  plotted field is affected, then fix it.
- **REL-05**: Wheels workflow red on `master` since 2026-08-12; logs have expired.
  Re-run, read logs, fix.
- **REL-07**: the current tests run only inside cibuildwheel. Add a separate
  workflow on `push` (all branches) and `pull_request` that builds once (Linux, one Python), runs pytest
  (skipping `slow`), and runs the C++ tests registered in `tests/CMakeLists.txt`.
  This is a prerequisite for branch protection (`PM-09`).
- **REL-09**: the existing `v0.1.15` tag was never released. Leave it in place;
  0.2.0 supersedes it.

# TEST: Test coverage & correctness

> LLMs have been used to create this list automatically based on the project definitions of @Socrats.

Goal: measured, rising coverage, and tests that check results are correct
rather than only that code runs. Today 539 Python tests pass in ~13 s, most of
them smoke tests. Coverage is not measured, and the C++ tests don't run in CI.

| ID | Task | Status | Pri | Size | MS | Deps | Ref |
|----|------|--------|-----|------|----|------|-----|
| TEST-01 | Coverage tooling: pytest-cov for Python, llvm-cov/gcov for C++, uploaded from CI | doing | P0 | M | 0.3 | REL-07 | — |
| TEST-02 | Coverage baseline report and per-module gap list | todo | P1 | S | 0.3 | TEST-01 | — |
| TEST-03 | Regression tests against published results (e.g. Traulsen et al. 2006; Santos et al. 2008) | todo | P1 | L | 0.3 | — | — |
| TEST-04 | Property-based tests (hypothesis): stochastic matrices, normalised distributions, symmetry | todo | P2 | M | 0.3 | — | — |
| TEST-05 | Register all `tests/cpp` programs with CTest and fix any that fail | done | P0 | M | 0.3 | — | via REL-07 (CTest in Tests workflow) |
| TEST-06 | Statistical-test helpers (compare MC estimate to exact within k·SE, fixed seeds) | todo | P1 | S | 0.3 | — | — |
| TEST-07 | Coverage floor enforced in CI, raised each release | todo | P1 | S | 0.4 | TEST-02 | — |
| TEST-08 | Execute documentation notebooks as tests | todo | P2 | M | 0.4 | — | — |
| TEST-09 | Migrate the C++ tests to Catch2 v3 (vcpkg): real assertions, one test per behaviour, CTest discovery | todo | P1 | L | 0.3 | REL-07 | — |
| TEST-10 | Make the C++ tests check results in Release builds (they use `assert`, removed by `NDEBUG`); fix the unused-variable warnings | todo | P0 | S | 0.3 | — | — |

## Notes

- **TEST-01**: Python coverage runs in the Tests workflow (REL-14, badge via
  python-coverage-comment-action, no external service): 65% line coverage on 2026-11-18,
  lowest in plotting/barplots.py, games/opinion_game.py, behaviors/CRD and games/nonlinear_games.py.
  C++ coverage (gcov/llvm-cov) still to do. The build/test job is read-only; the badge is
  updated by a separate job with `contents: write`, only on pushes to master. Pull requests
  get the coverage table in the run summary, without PR comments.
- **TEST-10**: the Release build defines `NDEBUG`, so every `assert` in `tests/cpp`
  is compiled out and those programs pass without checking anything (the conda build
  shows it as "set but not used" warnings, e.g. `row_sum` and `has_diag` in
  `test_pairwise_comparison_analytical.cpp`). Stopgap until TEST-09: a check macro that
  always runs and returns a non-zero exit code, or `NDEBUG` undefined for the test
  targets. Also remove the unused `payoff` in `CRDGame.cpp:120`.

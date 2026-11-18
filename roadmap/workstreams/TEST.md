# TEST: Test coverage & correctness

> LLMs have been used to create this list automatically based on the project definitions of @Socrats.

Goal: measured, rising coverage, and tests that check results are correct
rather than only that code runs. Today 539 Python tests pass in ~13 s, most of
them smoke tests. Coverage is not measured, and the C++ tests don't run in CI.

| ID | Task | Status | Pri | Size | MS | Deps | Ref |
|----|------|--------|-----|------|----|------|-----|
| TEST-01 | Coverage tooling: pytest-cov for Python, llvm-cov/gcov for C++, uploaded from CI | todo | P0 | M | 0.3 | REL-07 | — |
| TEST-02 | Coverage baseline report and per-module gap list | todo | P1 | S | 0.3 | TEST-01 | — |
| TEST-03 | Regression tests against published results (e.g. Traulsen et al. 2006; Santos et al. 2008) | todo | P1 | L | 0.3 | — | — |
| TEST-04 | Property-based tests (hypothesis): stochastic matrices, normalised distributions, symmetry | todo | P2 | M | 0.3 | — | — |
| TEST-05 | Register all `tests/cpp` programs with CTest and fix any that fail | todo | P0 | M | 0.3 | — | — |
| TEST-06 | Statistical-test helpers (compare MC estimate to exact within k·SE, fixed seeds) | todo | P1 | S | 0.3 | — | — |
| TEST-07 | Coverage floor enforced in CI, raised each release | todo | P1 | S | 0.4 | TEST-02 | — |
| TEST-08 | Execute documentation notebooks as tests | todo | P2 | M | 0.4 | — | — |
| TEST-09 | Migrate the C++ tests to Catch2 v3 (vcpkg): real assertions, one test per behaviour, CTest discovery | todo | P1 | L | 0.3 | REL-07 | — |

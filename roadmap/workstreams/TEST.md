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
| TEST-06 | Statistical-test helpers (compare MC estimate to exact within k·SE, fixed seeds) | done | P1 | S | 0.3 | — | `ccec953c` |
| TEST-07 | Coverage floor enforced in CI, raised each release | todo | P1 | S | 0.4 | TEST-02 | — |
| TEST-08 | Execute documentation notebooks as tests | todo | P2 | M | 0.4 | — | — |
| TEST-09 | Migrate the C++ tests to Catch2 v3 (vcpkg): real assertions, one test per behaviour, CTest discovery | todo | P1 | L | 0.3 | REL-07 | — |
| TEST-10 | Make the C++ tests check results in Release builds (they use `assert`, removed by `NDEBUG`); fix the unused-variable warnings | done | P0 | S | 0.3 | — | `1512fc95` |
| TEST-11 | Shared test chains with known stationary distributions (Python test helper module), used by SOL-10 and MC-09 | review | P0 | S | 0.3 | — | branch `test-11-shared-chains` |
| TEST-12 | Numerical-stability audit of the library (overflow, cancellation, log-space computation); tests at extreme parameters; one task per fix | todo | P1 | L | 0.3 | TEST-06 | — |
| TEST-13 | Clean up the warnings in the test suite, then make warnings errors in CI (explicit, commented ignores only) | todo | P1 | M | 0.3 | — | — |

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
  `test_pairwise_comparison_analytical.cpp`). Stopgap until TEST-09: undefine `NDEBUG` in
  CMake for the test programs and the `egttoolsNumerical` test library (MSVC too: the
  tests are built in every wheel), so the library's own asserts (`Group`, `GarciaGroup`,
  `Utils.hpp`, `LruCache.hpp`) also run; add a guard test that fails if `NDEBUG` is
  defined. Checked 2026-11-19: with asserts enabled all 23 programs pass on macOS.
  Also remove the unused `payoff` in `CRDGame.cpp:120`, and compare
  `test_binomial_coefficient_with_time.cpp` against its `correct_solution1/2`.
- **TEST-09**: only 8 of the 23 programs assert anything; the other 15 print output.
  Every migrated test gets real checks. Start with the programs that check nothing and
  with C++ code that has no Python test (`NetworkSync`, `NetworkEvolver`, the binomial
  and hypergeometric helpers, the LRU caches).
- **TEST-11**: start from the chains in `tests/test_stationary_distribution_utils.py`
  (random irreducible chain, dense reference solver, closed-form birth-death chain) and
  add a doubly stochastic chain (uniform distribution), the two-state chain, and the
  neutral Moran chain with mutation. Split out of SOL-10 so phase 1 does not depend on
  the solvers.
  Done: `tests/chains.py` (random irreducible chain + dense reference solver, two-state, doubly
  stochastic via Birkhoff mixtures, birth-death, neutral two-strategy pairwise comparison with mutation
  from Definition 4.1 via detailed balance); `tests/test_chains.py` self-tests, including a check that
  egttools' β = 0 transition matrix equals the neutral chain (to 1e-14).
- **TEST-12**: every place that exponentiates fitness or payoffs, sums probabilities, or divides
  by small differences. Found 2026-11-19: the analytical fixation probability is already computed
  in log space (streaming log-sum-exp in `calculate_log_phi_`, plus a stable log-fixation method);
  `fermi()` (`Utils.cpp`) is correct in IEEE arithmetic (overflow gives 0, never NaN) but raises
  overflow in hot loops; the network update rules `BirthDeath` and `DeathBirth` normalise
  `exp(β·f)` without subtracting the maximum, so β·f above ~709 gives `inf` weights and NaN or
  invalid sampling (fix: subtract max β·f first); `sed_analytical.py` clips an overflowing
  `np.exp`. Check also the hypergeometric log-space PDFs (PERF-07), the stationary-distribution
  helpers and the gradients. Tests use extreme β, payoffs and population sizes and compare with
  a high-precision reference. No fast-math flags are used; keep it that way (DECISIONS.md).
- **TEST-06**: `tests/stats_helpers.py` (independent of egttools): iid and batch-means SEs, `assert_within_k_se`
  (z-scores in the message), `assert_detects` (power check), Sokal's τ_int, run length from a pilot run;
  fixed-seed `rng` fixture in `conftest.py`. Constants from the literature (cited in the module):
  c = 5 and n > 50 τ (Sokal, via emcee), batch size ⌊√n⌋ with overlapping batch means (Flegal & Jones
  2010), burn-in 20 τ (Janke 2008: 20 τ_exp). A-priori spectral bounds were dropped: they cost as much as
  solving the chain exactly, and a gap-based run-length recipe is not standard (kept for the EMCUtils
  paper). Self-tests on AR(1) processes with known τ_int and long-run variance.
- **TEST-13**: 54 warnings in a normal run (2026-11-19), grouped as:
  - Test-helper defects in `test_pairwise_comparison_equations.py`: `fitness_two_player` computes
    `n[m] - (l == m)` on `uint64` counts, which wraps for an absent strategy (value unused, but the
    helper is wrong there); the `fermi` reference overflows in `exp` (use the stable form, TEST-12).
  - Library warnings the tests trigger but do not assert: the small-μ warning of the MC
    estimators, the checkpoint `n_chunks` warning. Use parameters that avoid them, or assert them
    with `pytest.warns` where the warning is the point.
  - Third-party: a `DeprecationWarning` from `pytz` (`datetime.utcfromtimestamp`); ignore it by
    message in `pyproject.toml`, with a comment.
  - The extra warnings on a first run (up to ~240) are `DeprecationWarning: ast.Str` from pytest 7.3.1's
    assertion rewriting on Python 3.13, emitted only when a test file changed; upgrade pytest in egtenv.
  End state: `filterwarnings = ["error", ...]` in `[tool.pytest.ini_options]`, so a new warning
  fails CI, with each ignore listed and justified.

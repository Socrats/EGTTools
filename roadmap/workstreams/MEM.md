# MEM: Memory safety & leak detection

> LLMs have been used to create this list automatically based on the project definitions of @Socrats.

Goal: catch memory errors and leaks automatically, both now and in future
changes. Today nothing checks for them in CI.

| ID | Task | Status | Pri | Size | MS | Deps | Ref |
|----|------|--------|-----|------|----|------|-----|
| MEM-01 | CMake option `EGTTOOLS_SANITIZE=address;undefined` | review | P0 | S | 0.3 | — | branch `mem-01-sanitize-option` |
| MEM-02 | CI job: C++ tests under ASan + UBSan (Linux, clang) | todo | P0 | M | 0.3 | MEM-01, TEST-05, MEM-08 | — |
| MEM-03 | CI job: Python test suite against an ASan build (`LD_PRELOAD` the runtime, leak check on exit) | todo | P1 | M | 0.3 | MEM-01 | — |
| MEM-04 | Scheduled (nightly/weekly) valgrind memcheck run on long MC tests | doing | P2 | M | 0.3 | MEM-02 | `docker/memcheck/` (not yet built) |
| MEM-05 | Audit pybind11 ownership: `return_value_policy::reference` and `keep_alive` uses | todo | P1 | M | 0.3 | — | — |
| MEM-06 | Memory-growth regression tests on long-running estimators (pytest-memray limits) | todo | P1 | S | 0.3 | — | — |
| MEM-07 | Document how to run sanitizer builds locally (Linux and macOS) | todo | P2 | S | 0.3 | MEM-01 | — |
| MEM-08 | State counts in 64-bit: `starsBars` deduced as `int` overflows (UB) in `PairwiseComparison` and other constructors; mark counts that do not fit as too large | todo | P0 | S | 0.3 | MEM-01 | — |
| MEM-09 | Validate strategy indices and state vectors at public entry points (`PairwiseComparison` first, then audit); raise `ValueError` | todo | P0 | M | 0.3 | — | — |
| MEM-10 | Consistent 64-bit types: audit `Types.h` and every size, count and index type (no `int`/`long` for sizes; `long` is 32-bit on Windows) | todo | P1 | L | 0.3 | MEM-08 | — |

Found and fixed so far: out-of-bounds row write in `run_with_mutation` with a transient (REL-10).
Found by the first sanitizer run (MEM-01, 2026-11-19): signed overflow in the state count of
`PairwiseComparison` (MEM-08). Found while checking it: no input validation in `PairwiseComparison` (MEM-09).

## Notes

- **MEM-01**: `cmake/Modules/EnableSanitizers.cmake`; values `address`, `undefined`, `thread`,
  `leak` (MSVC: `address` only); disables LTO; `-fno-sanitize-recover=undefined`.
  `testSanitizerCanary` (registered only in sanitizer builds) commits one deliberate error per
  enabled sanitizer and passes only if the report appears. Checked on macOS (Apple clang):
  ASan + UBSan canaries pass, the TSan canary reports the race; 23 of 24 C++ tests pass, and
  `testFixations` fails on a real bug (MEM-08).
- **MEM-08**: `PairwiseComparison` stores `nb_states_` as `int64_t` but calls
  `starsBars(population_size_, nb_strategies_)` with two `int`s, so the count is computed in `int`.
  Wrong (negative) from 6 strategies at Z = 136, 8 at Z = 52, 10 at Z = 31, though the true count
  fits in `int`. `testFixations` (29 strategies, Z = 100, count ≈ 1e28) computes it in the
  constructor although fixation probabilities never use it. Fix: compute every state count in
  64-bit at the call sites where `int` is deduced; when it does not fit, store "too large" so
  methods that need the full state space raise a clear error and edge/fixation methods still work.
- **MEM-09**: with 4 strategies and Z = 100, `calculate_fixation_probability(7, 0, β)` returns 1.0
  and `(-1, 0, β)` returns 1.7e-15; `calculate_gradient_of_selection` accepts states that do not sum
  to Z, a length-2 state (reads past the vector: ±2.2e7) and a length-5 state (extra entry ignored).
  `PairwiseComparisonNumerical` already raises `ValueError` for invalid indices. Validate at the
  public entry points only (O(S) per call, never inside internal loops); add tests with invalid
  inputs for each check; then audit the other classes that take indices or states.
- **MEM-10**: `Types.h` is mostly 64-bit already (`VectorXui` = `size_t`, `VectorXi` = `int64_t`),
  but `Table2D`, `SparseTable2D`, `SparseMatrix2DXi` and the sparse index type use `long`, which is
  32-bit on Windows (76 more uses of `long` elsewhere); `VectorXi` and `VectorXli` are identical.
  Sizes and indices are often `int` (e.g. `PairwiseComparison::population_size_, nb_strategies_`;
  about 225 `int` parameters in the headers). Decide one signed 64-bit index type (consistent with
  `Eigen::Index` and NumPy's `int64`) and one count type, define them in `Types.h`, and use them
  everywhere; 32-bit only where a GPU kernel needs it. Check the Python-visible dtypes (CHANGELOG
  if any change). Before the MC redesign, which rewrites much of this code.

- **MEM-03**: CPython's own allocations produce false positives. Use
  `PYTHONMALLOC=malloc` and a suppressions file kept in the repo.
- **MEM-05**: there are 17 explicit policy annotations in
  `cpp/src/pybind11_files`. Check each returned reference for a lifetime that
  can outlive its owner.
- **MEM-04**: `docker/memcheck/` (from REL-13) runs Valgrind on the C++ test programs (`cpp`,
  default) or the Python suite (`python`). Written without Docker available locally: not yet built.
- **MEM-06**: covers the trajectory-storing paths and estimators that run for
  many generations. Assert that peak memory does not grow with
  `nb_generations`.

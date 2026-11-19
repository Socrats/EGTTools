# MEM: Memory safety & leak detection

> LLMs have been used to create this list automatically based on the project definitions of @Socrats.

Goal: catch memory errors and leaks automatically, both now and in future
changes. Today nothing checks for them in CI.

| ID | Task | Status | Pri | Size | MS | Deps | Ref |
|----|------|--------|-----|------|----|------|-----|
| MEM-01 | CMake option `EGTTOOLS_SANITIZE=address;undefined` | done | P0 | S | 0.3 | — | `07f312c8` |
| MEM-02 | CI job: C++ tests under ASan + UBSan (Linux, clang) | todo | P0 | M | 0.3 | MEM-01, TEST-05, MEM-08 | — |
| MEM-03 | CI job: Python test suite against an ASan build (`LD_PRELOAD` the runtime, leak check on exit) | todo | P1 | M | 0.3 | MEM-01 | — |
| MEM-04 | Scheduled (nightly/weekly) valgrind memcheck run on long MC tests | doing | P2 | M | 0.3 | MEM-02 | `docker/memcheck/` (not yet built) |
| MEM-05 | Audit pybind11 ownership: `return_value_policy::reference` and `keep_alive` uses | todo | P1 | M | 0.3 | — | — |
| MEM-06 | Memory-growth regression tests on long-running estimators (pytest-memray limits) | todo | P1 | S | 0.3 | — | — |
| MEM-07 | Document how to run sanitizer builds locally (Linux and macOS) | todo | P2 | S | 0.3 | MEM-01 | — |
| MEM-08 | State counts in 64-bit: `starsBars` deduced as `int` overflows (UB) in `PairwiseComparison` and other constructors; mark counts that do not fit as too large | done | P0 | S | 0.3 | MEM-01 | `0fb0fcbe` |
| MEM-09 | Validate strategy indices and state vectors at the user-facing entry points of `PairwiseComparison` and `PairwiseComparisonNumerical` | review | P0 | M | 0.3 | — | branch `mem-09-input-validation` |
| MEM-10 | Consistent 64-bit types: audit `Types.h` and every size, count and index type (no `int`/`long` for sizes; `long` is 32-bit on Windows) | todo | P1 | L | 0.3 | MEM-08 | — |
| MEM-11 | Input-validation audit of the other user-facing classes (games, network structures, `StochDynamics`, MLS, utilities), following the MEM-09 rule | todo | P1 | M | 0.3 | MEM-09 | — |

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
  constructor although fixation probabilities never use it. The `size_t` call sites
  (`PairwiseComparisonNumerical`, `ImitationMultipleGames`, the transition operator,
  `NetworkEvolver`, the PETSc operator) overflow silently above 2^64.
  Done (2026-11-19), with two changes to the first design, both measured:
  - The exact overflow check went into `binomialCoeff` itself for integer outputs (one function for
    every caller), not into a separate `checked_binomial`. Fast path: a multiply with an overflow
    flag (`__builtin_mul_overflow`; one division on MSVC); past the limit, a gcd-reduced exact step
    in an out-of-line cold function (`detail::binomial_finish_reduced`) that throws
    `std::overflow_error`. Inlined with the slow path, the check made the transition matrix
    15-20% slower (it calls `calculate_state` for every neighbour); out of line, +1-4% (noise level).
  - Fitness-cache keys stay 64-bit (a 16-byte key struct was measurably slower): full states
    `index * S + strategy` (checked once to stay below 2^63), edge states `2^63 | (edge_code * S +
    strategy)` with edge_code = (lo * S + hi) * (Z + 1) + count of lo. Fixation probabilities and
    edge fitnesses never compute global indices any more (the index of an edge state with 29
    strategies is ~1e28): fixation and SML about 28% faster. The old `index << 32` keys did not
    collide in any configuration checked (Z ≤ 1000, S ≤ 29, simulated exactly in Python).
  - Counts are `std::optional<int64_t>` (`nb_states_if_representable`); `nb_states()` and every
    state-space method throw `std::overflow_error` (Python `OverflowError`) naming Z and S, before
    any OpenMP region. `PairwiseComparisonTransitionOperator` and the PETSc operator throw at
    construction (PETSc also checks the `PetscInt` range). `egttools.calculate_nb_states` was
    already exact (`cpp_int`) and is unchanged.
  - `pre_calculate_edge_fitnesses` skips j == i (states that do not sum to Z).
  Tests: `tests/cpp/test_binomial_coeff_overflow.cpp` (against Pascal's triangle up to n = 80, int64
  and uint64 boundaries, SIZE_MAX), `tests/test_state_counts.py` (32 tests: counts vs `math.comb`,
  OverflowError cases, fixation with 29 strategies vs a cache-free reference). Confirmed to fail on
  master first, under `scripts/memguard.py` (an unguarded run of the transition-matrix test
  allocated over 100 GB).
- **MEM-09**: with 4 strategies and Z = 100, `calculate_fixation_probability(7, 0, β)` returns 1.0
  and `(-1, 0, β)` returns 1.7e-15; `calculate_gradient_of_selection` accepts states that do not sum
  to Z, a length-2 state (reads past the vector: ±2.2e7) and a length-5 state (extra entry ignored).
  `PairwiseComparisonNumerical` already raises `ValueError` for invalid indices. Validate at the
  public entry points only (O(S) per call, never inside internal loops); add tests with invalid
  inputs for each check; then audit the other classes that take indices or states.
  Design (agreed 2026-11-19): only user-facing entry points validate; kernels the library calls in
  inner loops (e.g. a game's `calculate_fitness`) document their preconditions and are validated in
  their Python bindings, so internal calls pay nothing. Out-of-range index → `std::out_of_range`
  (Python `IndexError`); everything else → `std::invalid_argument` (`ValueError`). Shared helpers in
  `finite_populations/Validation.hpp`: `validate_strategy_index`, `validate_invader_resident`,
  `validate_population_state` (bounds each entry by Z before summing, so unsigned wrap-around cannot
  reach Z). Done for `PairwiseComparison` (fixation ×3, gradients ×2) and `PairwiseComparisonNumerical`
  (fixation and gradient indices, now also negative; all state checks via the helper).
  `tests/test_input_validation.py` (62 tests); on master an out-of-range index crashed the interpreter.
- **MEM-11**: list every user-facing method that takes strategy indices, states or counts, and
  apply the MEM-09 rule (validate in the method, or in the Python binding for inner-loop kernels).
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

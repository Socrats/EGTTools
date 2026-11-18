# MEM: Memory safety & leak detection

> LLMs have been used to create this list automatically based on the project definitions of @Socrats.

Goal: catch memory errors and leaks automatically, both now and in future
changes. Today nothing checks for them in CI.

| ID | Task | Status | Pri | Size | MS | Deps | Ref |
|----|------|--------|-----|------|----|------|-----|
| MEM-01 | CMake option `EGTTOOLS_SANITIZE=address;undefined` | todo | P0 | S | 0.3 | — | — |
| MEM-02 | CI job: C++ tests under ASan + UBSan (Linux, clang) | todo | P0 | M | 0.3 | MEM-01, TEST-05 | — |
| MEM-03 | CI job: Python test suite against an ASan build (`LD_PRELOAD` the runtime, leak check on exit) | todo | P1 | M | 0.3 | MEM-01 | — |
| MEM-04 | Scheduled (nightly/weekly) valgrind memcheck run on long MC tests | doing | P2 | M | 0.3 | MEM-02 | `docker/memcheck/` (not yet built) |
| MEM-05 | Audit pybind11 ownership: `return_value_policy::reference` and `keep_alive` uses | todo | P1 | M | 0.3 | — | — |
| MEM-06 | Memory-growth regression tests on long-running estimators (pytest-memray limits) | todo | P1 | S | 0.3 | — | — |
| MEM-07 | Document how to run sanitizer builds locally (Linux and macOS) | todo | P2 | S | 0.3 | MEM-01 | — |

Found and fixed so far: out-of-bounds row write in `run_with_mutation` with a transient (REL-10).

## Notes

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

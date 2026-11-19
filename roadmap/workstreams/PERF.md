# PERF: Performance & benchmarks

> LLMs have been used to create this list automatically based on the project definitions of @Socrats.

Goal: measure performance continuously so regressions are visible, fix
the known hot spots, and make sure every parallel code path is correct and
reproducible.

| ID | Task | Status | Pri | Size | MS | Deps | Ref |
|----|------|--------|-----|------|----|------|-----|
| PERF-01 | Benchmark suite (asv) for MC, analytical transition matrices, solvers, network engine | todo | P1 | M | 0.4 | PERF-05 | — |
| PERF-02 | Profile hot paths (fitness caches, payoff precomputation, binding overhead) and file tasks | dropped | P1 | M | 0.4 | PERF-01 | superseded by PERF-05 |
| PERF-03 | Track benchmark history per release | todo | P2 | S | 0.4 | PERF-01 | — |
| PERF-04 | Parallelisation design and test plan: thread safety, per-run RNG streams, reproducibility across thread counts, scheduling, false sharing, MSVC | todo | P1 | M | 0.3 | API-08 | — |
| PERF-05 | Efficiency audit of the library, focused on `PairwiseComparison` and `PairwiseComparisonNumerical`; file one task per fix | todo | P1 | L | 0.3 | TEST-06, TEST-11 | — |
| PERF-06 | Implement the PERF-04 test plan: thread-count invariance tests and a ThreadSanitizer CI job | todo | P1 | M | 0.3 | PERF-04, MEM-01 | — |

## Notes

- **PERF-02**: dropped on 2026-11-19; its scope moved into PERF-05, which runs in 0.3.
- **PERF-04**: a written plan (`docs/` developer page) covering every OpenMP region
  (44 `#pragma omp parallel` in `cpp/`), what each region shares, and how it is tested.
  Known issues to address:
  - Each Monte Carlo run takes its seed from the shared `SeedGenerator` inside the
    parallel loop (`PairwiseComparisonNumerical.hpp`, e.g. lines 951, 1166, 1590). Which
    run gets which seed depends on thread scheduling, so a fixed seed does not give
    reproducible results with more than one thread, and results change with the thread
    count. Design: derive each run's stream from the base seed and the run index before
    the loop, so results depend only on the seed. The stream scheme comes from API-08.
  - Most OpenMP regions are guarded by `!defined(_MSC_VER)`, so Windows runs serially.
    Decide whether to support MSVC's OpenMP (2.0 only: no custom reductions,
    `size_t` loop indices) or document Windows as serial.
  - Shared `ThreadSafeLRUCache` with string keys (lock contention; overlaps PERF-05).
  - No way to set the thread count from Python; `tests/test_openmp.py` has a single test
    that does not compare thread counts.
  - Floating-point reductions depend on the summation order; integer counts do not.
    The plan states, per method, whether results must match exactly or statistically
    (TEST-06) across thread counts.
  New code from the MC redesign (MC-01..05) and the network work follows this plan.
- **PERF-05**: before the MC redesign (phase 4), so its findings feed MC-01, MC-02 and
  MC-07 instead of optimising code that will be replaced. Method: a small benchmark
  script set under `benchmarks/` (PERF-01 later turns it into asv), profiling of
  transition-matrix construction, fitness computation and caching, SML fixation
  matrices, the MC estimators, and pybind11 call and copy overhead. Every optimisation
  is a separate task, and the existing tests must pass unchanged unless the CHANGELOG
  lists a change in numerical results.
- **PERF-06**: run each parallel method with 1, 2 and N threads and a fixed seed;
  compare results exactly or within k·SE as PERF-04 specifies. TSan with OpenMP needs
  an OpenMP runtime built with TSan support (or the Archer tool) to avoid false
  positives; add `thread` to `EGTTOOLS_SANITIZE` (MEM-01).

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
| PERF-07 | Combinatorics and special functions with tiered precision: fast precomputed tables with checks, higher precision only when needed or requested, documented error bounds, thread safety | todo | P1 | L | 0.3 | — | — |
| PERF-08 | Evaluate fixed-point (scaled integer) arithmetic for selected operations: exact, order-independent sums and a known error per operation | todo | P2 | M | later | PERF-04, TEST-12 | — |

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
  - `CRDGameTU::calculate_payoffs` runs an OpenMP loop whose iterations all draw from the
    member `generator_` (`CRDGameTU.cpp`, `play`): a data race (found in TEST-09).
  - No way to set the thread count from Python; `tests/test_openmp.py` has a single test
    that does not compare thread counts.
  - Floating-point reductions depend on the summation order; integer counts do not.
    The plan states, per method, whether results must match exactly or statistically
    (TEST-06) across thread counts.
  New code from the MC redesign (MC-01..05) and the network work follows this plan.
- **PERF-05**: covers the whole library; `PairwiseComparison` and `PairwiseComparisonNumerical`
  come first. Runs alongside TEST-12 (numerical stability), so speed-ups do not trade away accuracy. Before the MC redesign (phase 4), so its findings feed MC-01, MC-02 and
  MC-07 instead of optimising code that will be replaced. Method: a small benchmark
  script set under `benchmarks/` (PERF-01 later turns it into asv), profiling of
  transition-matrix construction, fitness computation and caching, SML fixation
  matrices, the MC estimators, and pybind11 call and copy overhead. Every optimisation
  is a separate task, and the existing tests must pass unchanged unless the CHANGELOG
  lists a change in numerical results.
- **PERF-07**: applies the tiered-precision rule (DECISIONS.md, 2026-11-19) to
  `Distributions.h/.cpp`: binomial coefficients, `starsBars`, log factorials and the
  hypergeometric PDF. Three tiers: (1) fast path from precomputed constants or fixed-width
  arithmetic with cheap checks; (2) automatic switch to higher precision only when a check
  shows the fast path cannot be exact (as `binomial_precision` now does above 2^64);
  (3) an opt-in high-precision mode chosen by the user. Known issues (2026-11-19):
  - `ln_factorial` extends the shared `static std::vector ln_factorials` (about 400
    precomputed values) with `push_back` for larger n. Called from OpenMP loops with
    Z above the table size, that is a data race (reallocation during reads). Fill the table
    up to the needed size before parallel regions, or use a thread-safe fixed table plus a
    Stirling series with a stated error bound. `std::lgamma` is not a drop-in fix: it is not
    guaranteed thread-safe (`signgam`).
  - `binomialCoeff<size_t>` (used by `starsBars` for state counts and in the state-indexing
    loops in `finite_populations/Utils.cpp`) overflows in the intermediate product from n = 63,
    silently. Candidates: CPython `math.comb`'s 64-bit tables (odd part of n! mod 2^64, its
    inverse and the power of two; exact for n ≤ 67, no division) or a Pascal table per model
    for state indexing, with an overflow check beyond the table.
  - Log-space probabilities (`multivariateHypergeometricPDF`) have relative error of about
    |ln P|·ε; document it and expose it where results are reported.
  Prior art to follow rather than reinvent: netlib MATH77 `DBINOM` (factorial table,
  log-gamma above N = 150, checked against Pascal's triangle), Boost.Math
  `binomial_coefficient` (factorial table up to 170, then the beta function), CPython
  `math.comb` (64-bit tables), GMP `mpz_bin_uiui` (Goetgheluck's prime factorisation for
  large n). Benchmarks (2026-11-19, Apple M-series): `cpp_int` is 1.5–2.2× slower than
  fixed-width 128-bit arithmetic for the same binomials; an overflow check per iteration
  costs nothing measurable. Lesson from MEM-08: keep checked kernels small and move rare paths
  (gcd steps, error formatting, throws) out of line, or the compiler stops inlining them into hot
  loops (+15-20% on transition matrices, back to +1-4% out of line).
- **PERF-08**: suggested 2026-11-19, for 0.3 or a later version. JAX and PyTorch offer integer
  quantized types (values stored as integers with a scale, i.e. fixed point) mainly for speed on
  accelerators, and deterministic modes for reproducible floating-point reductions. For EGTtools
  the candidate benefits are: integer addition is associative, so parallel reductions give
  bit-identical results for any thread count (PERF-04); rows of a transition matrix can sum to 1
  exactly; the error per operation is fixed and known (the resolution step); integer arithmetic
  vectorises well on CPUs and GPUs. Limits: the range is small, so quantities spanning many orders of
  magnitude (fixation probabilities down to 1e-300, stationary distributions of large chains) stay
  in log space or floating point; products need rescaling and overflow checks. Candidates to
  evaluate: transition-probability accumulation, the Monte Carlo accumulators (already integer
  counts), weighted averages of indicators. Deliverable: a short study with error bounds and
  benchmarks on one kernel, and a decision in DECISIONS.md. Can move into 0.3 if PERF-04 or TEST-12
  show that floating-point reductions are the problem.
- **PERF-06**: run each parallel method with 1, 2 and N threads and a fixed seed;
  compare results exactly or within k·SE as PERF-04 specifies. TSan with OpenMP needs
  an OpenMP runtime built with TSan support (or the Archer tool) to avoid false
  positives; add `thread` to `EGTTOOLS_SANITIZE` (MEM-01).

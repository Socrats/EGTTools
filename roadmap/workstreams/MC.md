# MC: Monte Carlo estimation (well-mixed populations)

> LLMs have been used to create this list automatically based on the project definitions of @Socrats.

Goal: make `PairwiseComparisonNumerical` estimators fast and statistically
sound, and have them report how reliable their estimates are.

Current problems:
- Each fitness cache miss sums over all C(N+S−1, S−1) group compositions.
- `indicator_type="state"` calls back into Python on every step when the state
  space is large.
- Convergence is checked by the L1 change between consecutive batches, which
  measures stability, not uncertainty.
- Results report no standard error, effective sample size, or between-chain
  agreement.

The new design follows an independently validated fast estimator: it agrees
with the exact chain to within about 1 SE, and is several orders of magnitude
faster for games with many strategies.

| ID | Task | Status | Pri | Size | MS | Deps | Ref |
|----|------|--------|-----|------|----|------|-----|
| MC-01 | Structured fitness: let games compute fitness from strategy counts without enumerating all group compositions | todo | P0 | L | 0.3 | — | — |
| MC-02 | Incremental fitness updates and streaming (no stored trajectory) accumulation | todo | P0 | M | 0.3 | MC-01 | — |
| MC-03 | Batch-means MCSE, effective sample size and split-R̂ in result objects | todo | P0 | M | 0.3 | MC-02 | — |
| MC-04 | Dispersed starts (corners + random), τ_int-based burn-in, between-start agreement, explicit `converged` flag | todo | P0 | L | 0.3 | MC-03 | — |
| MC-05 | Replace L1-tolerance stopping with target-SE stopping | todo | P1 | M | 0.3 | MC-03 | — |
| MC-06 | Native fast path for indicators that are linear in the state (no Python callback) | todo | P1 | M | 0.3 | MC-02 | — |
| MC-07 | Edge/SML fitness: sum only over compositions of the strategies present | todo | P0 | M | 0.3 | — | — |
| MC-08 | Vectorise `calculate_payoffs` in bundled Python games; document the pattern | todo | P1 | M | 0.3 | — | — |
| MC-09 | Statistical validation tests: MC vs exact chain within k·SE | todo | P0 | M | 0.3 | MC-03 | — |
| MC-10 | Keep checkpoint/resume working with the new estimators | todo | P1 | S | 0.3 | MC-04 | — |
| MC-11 | Mean first-passage time estimator | todo | P3 | M | later | MC-02 | — |

## Notes

- **MC-01**: for games whose payoff depends on the group only through a small
  sufficient statistic (e.g. number of contributors per threshold), fitness
  costs O(S·(N+1)) instead of a sum over all compositions. Design this as an
  optional game interface with a generic fallback, so existing games keep working.
- **MC-04**: run each dispersed start with a distinct seed. Use Sokal's
  automated windowing for τ_int, burn-in = c·τ_int, batch length ≥ 10·τ_int, and
  a reduced χ² test across starts. `converged` is false, with a list of
  reasons, if any check fails.
- **MC-07**: with two strategies present, only N+1 compositions have non-zero
  weight. The current code sums over every composition, which makes SML
  transition matrices slow even for a handful of strategies.
- **MC-08**: do not fall back to the inherited C++ `calculate_payoffs` from
  inside a Python subclass `__init__`. The pybind11 virtual dispatch to `play()`
  fails before construction finishes. Override with a vectorised Python body
  instead.

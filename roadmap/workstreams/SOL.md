# SOL: Stationary-distribution solvers

> LLMs have been used to create this list automatically based on the project definitions of @Socrats.

Goal: one documented interface to standard, robust solvers for the stationary
distribution of large sparse transition matrices, both explicit and
matrix-free.

Scope covers standard methods only: direct, ARPACK, Krylov
(GMRES / LGMRES / BiCGSTAB) and power iteration.

| ID | Task | Status | Pri | Size | MS | Deps | Ref |
|----|------|--------|-----|------|----|------|-----|
| SOL-01 | Unified API: `stationary_distribution(P, method=...)` returning a result object | todo | P0 | M | 0.3 | REL-03 | — |
| SOL-02 | Direct sparse solve (SuperLU) and ARPACK eigensolver | todo | P0 | M | 0.3 | SOL-01 | — |
| SOL-03 | GMRES, LGMRES and BiCGSTAB with optional ILU preconditioning | todo | P0 | M | 0.3 | SOL-01 | — |
| SOL-04 | ILU size guard: warn above a threshold, hard limit falls back to no preconditioner | todo | P0 | S | 0.3 | SOL-03 | — |
| SOL-05 | Power iteration and Anderson-accelerated power iteration | todo | P1 | M | 0.3 | SOL-01 | — |
| SOL-06 | Matrix-free path through `PairwiseComparisonTransitionOperator` (native C++, no Python callbacks) | todo | P1 | M | 0.3 | SOL-01 | — |
| SOL-07 | Optional GPU (CuPy) power iteration, as an extra | todo | P2 | M | 0.3 | SOL-05 | — |
| SOL-08 | Expose existing PETSc/SLEPc operator through the same API (`egttools[mpi]`) | todo | P2 | M | 0.3 | SOL-01 | — |
| SOL-09 | Result diagnostics: residual ‖πP − π‖, iterations, converged flag, wall time | todo | P0 | S | 0.3 | SOL-01 | — |
| SOL-10 | Tests per method on chains with known stationary distribution | todo | P0 | M | 0.3 | TEST-11, SOL-02..05 | — |
| SOL-11 | Per-method reference documentation | todo | P1 | M | 0.3 | SOL-10 | — |
| SOL-12 | Exact absorption probabilities and mean and variance of the absorption time for k strategies without mutation (fundamental matrix, sparse solve) | todo | P1 | M | 0.4 | DOC-12, SOL-02 | — |
| SOL-13 | Exact fixation times for two strategies: unconditional and conditional fixation times and sojourn times (closed forms) | todo | P1 | S | 0.4 | DOC-12 | — |
| SOL-14 | Exact mean first-passage times between states or sets of states, with mutation (linear solve); reference for MC-11 | todo | P2 | M | 0.4 | SOL-02 | — |

## Notes

- **SOL-04**: ILU fill-in makes preconditioning impractical for very large
  systems. Above a configurable soft size, warn. Above a hard size, skip ILU
  and run unpreconditioned, warning the user. Let the user override both
  limits explicitly.
- **SOL-06**: Python `LinearOperator` callbacks are 15–90× slower than an
  explicit CSR matrix with ARPACK at moderate sizes. The matrix-free path must
  stay in native code.
- **SOL-10**: the chains themselves are TEST-11 (phase 1); this task applies them to
  every solver method.
- **SOL-11**: document what each method does, its parameters, and its cost per
  iteration. Defaults are documented as defaults only.
- **SOL-12..14**: exact time and absorption results, requested on 2026-11-24 alongside the
  estimators that DOC-12 specifies (`docs/theory/fixation.md`). SOL-12 implements
  Proposition 3.3 of that page (B = N R, t = N 1, Var[tau] = (2N - I) t - t o t) on the
  transient states, with a sparse solve. SOL-13: closed forms for the birth-death chain
  (Antal & Scheuring 2006; Altrock & Traulsen 2009, New J. Phys. 11, 013012), computed in
  log space like the fixation probability. SOL-14: first-passage times solve
  (I - Q_A) m = 1 on the complement of the target set A; MC-11's estimator is checked against it.
  Each one gets a section on the theory pages and spec tests, and is exposed for
  `PairwiseComparison` (C++) and `StochDynamics` where it applies.

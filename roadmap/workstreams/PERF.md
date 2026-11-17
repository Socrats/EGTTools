# PERF: Performance & benchmarks

> LLMs have been used to create this list automatically based on the project definitions of @Socrats.

Goal: measure performance continuously so regressions are visible, and fix
the known hot spots.

| ID | Task | Status | Pri | Size | MS | Deps | Ref |
|----|------|--------|-----|------|----|------|-----|
| PERF-01 | Benchmark suite (asv) for MC, analytical transition matrices, solvers, network engine | todo | P1 | M | 0.4 | — | — |
| PERF-02 | Profile hot paths (fitness caches, payoff precomputation, binding overhead) and file tasks | todo | P1 | M | 0.4 | PERF-01 | — |
| PERF-03 | Track benchmark history per release | todo | P2 | S | 0.4 | PERF-01 | — |
| PERF-04 | Review OpenMP usage: thread counts, per-thread RNG streams, false sharing | todo | P2 | M | 0.4 | — | — |

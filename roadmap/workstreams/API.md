# API: API evolution

> LLMs have been used to create this list automatically based on the project definitions of @Socrats.

Goal: make module names describe what the methods do, without breaking users
abruptly.

| ID | Task | Status | Pri | Size | MS | Deps | Ref |
|----|------|--------|-----|------|----|------|-----|
| API-01 | Rename `analytical` → `exact`, `numerical` → `inexact`; class names unchanged | todo | P1 | L | 0.4 | — | — |
| API-02 | Deprecation shims for old module paths (`DeprecationWarning`, removal ≥ 2 minor releases later) | todo | P1 | M | 0.4 | API-01 | — |
| API-03 | Deprecation policy written down in `CONTRIBUTING.md` | todo | P1 | S | 0.4 | — | — |
| API-04 | Review hard dependencies (seaborn, plotly) and move plotting extras to optional | todo | P2 | S | 0.4 | — | — |
| API-05 | Make removed methods raise instead of warning and returning `None` (e.g. `PairwiseComparisonNumerical.run`) | todo | P1 | S | 0.2 | — | — |

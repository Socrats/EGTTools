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
| API-06 | One transition-matrix orientation: `StochDynamics.transition_and_fixation_matrix` returns it transposed (column-stochastic), the C++ SML version row-stochastic | todo | P1 | S | 0.3 | — | — |
| API-07 | Vector-quality 3D simplex plots: add a matplotlib (mplot3d) backend to `Simplex3D`, sharing its geometry code; keep plotly for interactive figures | todo | P2 | M | 0.4 | — | — |

## Notes

- **API-07**: plotly and its static exporter (Kaleido) are open source (MIT). The limitation is
  that 3D scenes are rendered with WebGL, so SVG/PDF exports contain them as raster images.
  matplotlib's mplot3d writes true vector output and covers what `Simplex3D` draws (edges,
  streamlines, arrows, points); its limitation is per-artist depth sorting. PyVista/VTK
  (`save_graphic`) is the alternative if lit surfaces are needed.

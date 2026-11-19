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
| API-05 | Make removed methods raise instead of warning and returning `None` (e.g. `PairwiseComparisonNumerical.run`) | done | P1 | S | 0.2 | — | branch `api-05-removed-methods-raise` |
| API-06 | One transition-matrix orientation: `StochDynamics.transition_and_fixation_matrix` returns it transposed (column-stochastic), the C++ SML version row-stochastic | todo | P1 | S | 0.3 | — | — |
| API-07 | Vector-quality 3D simplex plots: add a matplotlib (mplot3d) backend to `Simplex3D`, sharing its geometry code; keep plotly for interactive figures | todo | P2 | M | 0.4 | — | — |
| API-08 | Assess the random-number and seeding design against NumPy and JAX; decide the target design and file the migration tasks | todo | P1 | M | 0.3 | — | — |

## Notes

- **API-08**: current design (2026-11-19): a process-global `SeedGenerator` singleton
  (`egttools.Random`), seeded once and drawn from at 44 call sites in 17 C++ files; each
  draw seeds an `std::mt19937_64` from a single 64-bit value. Only a few classes accept a
  `seed` argument (e.g. `PairwiseComparisonNumerical`, the network structures). The
  Python layer mixes three approaches: the global `np.random` state (`plotting/helpers.py`,
  `plotting/simplex2d.py`), an unseeded `np.random.default_rng()`
  (`numerical/indicators.py`), and `default_rng(Random.generate())` (`nfg_strategies.py`).
  Compare with:
  - NumPy: explicit `Generator` objects, `SeedSequence` for high-entropy seeding and
    `spawn()` for independent streams, and the Scientific Python SPEC 7 convention of an
    `rng` argument that accepts `None`, an int or a `Generator`;
  - JAX: explicit, functional keys (`split`, `fold_in`) with counter-based generators, so
    a parallel stream is a deterministic function of the key and its index.
  Questions to answer: explicit `rng`/`seed` arguments versus the global singleton;
  stream derivation for parallel runs (PERF-04's per-run seeding fix uses the result);
  counter-based engines (Philox, as in NumPy and JAX) versus `mt19937_64`; whether the C++
  layer can accept a NumPy `SeedSequence` or `Generator` state; saving and restoring RNG
  state for checkpoints (MC-10). Compatibility: `egttools.Random` keeps working unless the
  CHANGELOG states a reason to change it. Deliverable: a design note with the decision in
  `DECISIONS.md`, and one task per migration step. DOC-06 documents the result.

- **API-07**: plotly and its static exporter (Kaleido) are open source (MIT). The limitation is
  that 3D scenes are rendered with WebGL, so SVG/PDF exports contain them as raster images.
  matplotlib's mplot3d writes true vector output and covers what `Simplex3D` draws (edges,
  streamlines, arrows, points); its limitation is per-artist depth sorting. PyVista/VTK
  (`save_graphic`) is the alternative if lit surfaces are needed.

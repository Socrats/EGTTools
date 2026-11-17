# Progress log

Newest first. One line per change: `YYYY-MM-DD · ID · what happened`.

- 2026-11-17 · REL-12 · Merged (`affebca`): named invasion-diagram styles (simple default, proportional, classic, compact), palette names/colormaps, absolute minimum node size.
- 2026-11-17 · API-06, DOC-13, DOC-14 · Added: transposed SML transition matrix in StochDynamics, Sphinx warnings, notebooks not executed in CI.
- 2026-11-17 · REL-12 · Merged (`a9b355e`): invasion_graph + redesigned draw_invasion_diagram; near-neutral drift, mutual invasions, colours/filter, labels.
- 2026-11-17 · REL-04 · Merged (`3050629`): lattice interpolation of the exact gradient, exact roots, stability rewrite.
- 2026-11-17 · REL-05 · Merged (`fc50406`): `setup.py` tag_regex and container safe.directory; CI green on all platforms, wheels still `0.0.0` before this.
- 2026-11-17 · CI-08, API-05, DOC-12, REL-11 · Added from findings (uv evaluation, silent deprecated stub, more theory pages, ImitationMultipleGames).
- 2026-11-17 · REL-06 · Merged (`8f5b901`): Python 3.11–3.14, cp314 wheels, conda numpy pairing.
- 2026-11-17 · REL-05 · Merged (`754c5f6`): CMake accepts suffixed tags; first CI run after fetching tags failed on `v0.1.14-patch2`.
- 2026-11-17 · REL-10 · Merged (`50d98c4`): pairwise comparison spec + 49 tests; MC homogeneous-state bias, OOB write, C++ and StochDynamics gradients fixed.
- 2026-11-17 · REL-04 · Investigated: roots, stability and arrows of the pairwise-comparison simplex plot use truncated/invalid states.
- 2026-11-17 · REL-02 · Merged (`5c43eaf`).
- 2026-11-17 · REL-05 · Merged (`4aee21d`): Windows `ssize_t`, check-manifest, CI version detection, explicit MANIFEST Markdown list. Waiting for CI.
- 2026-11-17 · REL-03 · Merged (`233473c`): Schur-vector bug fixed (wrong in ~1% of random chains), sparse input no longer densified (50k-state chain: 0.09 s), `csr_array` support; 21 new tests, 560 pass.
- 2026-11-17 · PM-10 · Dropped: no tracking issues; roadmap folder is enough for now.
- 2026-11-17 · PM-08 · Closed #81, #82, #67 as resolved; commented on #84, #72, #68 and put them in v0.4.0.
- 2026-11-17 · PM-07 · Labels renamed/created (`type:`, `area:`, `priority:`, `status:`); milestones v0.2.0, v0.3.0, v0.4.0 created.
- 2026-11-17 · PM-09 · Dropped: branch-and-merge workflow without PRs (see DECISIONS.md).
- 2026-11-17 · PM-01 · Merged into `master` (`00a2874`).
- 2026-11-17 · REL-01 · Merged into `master` (`83d6789`): pytest config merged into pyproject, dead scikit-build table and stale manifest entries removed; 539 tests pass.
- 2026-11-17 · PM-07/08/10 · Unblocked: GitHub labels and issues are in scope; only the Projects board is excluded.
- 2026-11-17 · PM-01 · Created `roadmap/` with schema, milestones v0.2.0–v0.4.0, and 11 workstreams from the project audit.

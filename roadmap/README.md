# EGTtools Roadmap

> LLMs have been used to create this list automatically based on the project definitions of @Socrats.

This folder tracks planned work and progress for EGTtools in plain Markdown.
It is the single source of truth for what is planned, in progress, and done.

## Layout

| File | Purpose |
|------|---------|
| `README.md` | This file: the schema, the workflow, and the milestone overview. |
| `workstreams/<AREA>.md` | One file per workstream. Each task lives in exactly one of them. |
| `DECISIONS.md` | Decision log: what was decided, when, and why. |
| `LOG.md` | Progress log: dated entries saying what changed. |

Milestones don't get their own files. A task's `Milestone` column says which
release it belongs to, so each task is written down in one place only.

## Task schema

Each workstream file has a task table with these columns:

| Column | Values |
|--------|--------|
| `ID` | `<AREA>-NN`, e.g. `MC-03`. IDs are never reused or renumbered. |
| `Task` | One-line description. Details go in the *Notes* section below the table, keyed by ID. |
| `Status` | `todo` · `ready` · `doing` · `review` · `done` · `blocked` · `dropped` |
| `Pri` | `P0` (release blocker) · `P1` (should) · `P2` (could) · `P3` (nice to have) |
| `Size` | `S` (≤ ½ day) · `M` (≤ 2 days) · `L` (≤ 1 week) · `XL` (> 1 week; split it) |
| `MS` | Target milestone: `0.2` · `0.3` · `0.4` · `later` |
| `Deps` | IDs that must be `done` first, or `—`. |
| `Ref` | Branch, merge commit, and/or issue once one exists, or `—`. |

**Status flow:** `todo → ready → doing → review → done`. `review` means the
work is finished and tested on its branch but not yet merged. `blocked` needs a
note saying what it is waiting on. `dropped` needs a note saying why, and goes
into `DECISIONS.md` if it is a scope decision.

**Definition of done:** merged to `master`, with tests, docstrings, a
CHANGELOG entry under *Unreleased*, and regenerated `.pyi` stubs if the
bindings changed.

## Workflow

1. Pick a `ready` task with all deps `done`. Set it to `doing`.
2. Create a branch from `master` named `<id>-<slug>` (e.g. `rel-03-stationary-dist`).
   Use one branch per task.
3. Push the branch: the Tests workflow runs on it. If the task changes the build, also start the
   Wheels workflow on the branch by hand. When tests pass, set the task to `review` with the branch in `Ref`.
4. Merge into `master` with `git merge --no-ff <branch>`, so each task stays
   visible as one merge in the history. Then set the task to `done` with the
   merge commit in `Ref`, and delete the branch.
5. Add a line to `LOG.md`. Record any scope change in `DECISIONS.md`.

Pull requests are not used while there is a single maintainer. External
contributions still arrive as PRs (see `CONTRIBUTING.md`).

## Areas

| Prefix | Workstream | File |
|--------|-----------|------|
| `REL` | Release v0.2.0 | [workstreams/REL.md](workstreams/REL.md) |
| `PM`  | Project management & community | [workstreams/PM.md](workstreams/PM.md) |
| `MC`  | Monte Carlo estimation | [workstreams/MC.md](workstreams/MC.md) |
| `NET` | Games and dynamics on networks | [workstreams/NET.md](workstreams/NET.md) |
| `SOL` | Stationary-distribution solvers | [workstreams/SOL.md](workstreams/SOL.md) |
| `MEM` | Memory safety & leak detection | [workstreams/MEM.md](workstreams/MEM.md) |
| `TEST`| Test coverage & correctness | [workstreams/TEST.md](workstreams/TEST.md) |
| `DOC` | Documentation | [workstreams/DOC.md](workstreams/DOC.md) |
| `CI`  | Build system & CI | [workstreams/CI.md](workstreams/CI.md) |
| `PERF`| Performance & benchmarks | [workstreams/PERF.md](workstreams/PERF.md) |
| `API` | API evolution | [workstreams/API.md](workstreams/API.md) |
| `SEC` | Security and privacy | [workstreams/SEC.md](workstreams/SEC.md) |

## Milestones

The versions stay at 0.x on purpose, because the API may still change
significantly (see `DECISIONS.md`).

### v0.2.0: ship what exists, CI green
Exit criteria: wheels build on every platform in the matrix, a PR test workflow
exists, the known numerical bugs in the stationary-distribution helpers are
fixed, CHANGELOG covers everything since 0.1.14, and the release is on PyPI.
Workstreams: `REL`, `PM` (community files).

### v0.3.0: correct and efficient core methods
Exit criteria: the new MC estimators have diagnostics and statistical
validation tests, the network models are checked against a reference
implementation, the solver module is in place, sanitizer CI runs, and
coverage is measured.
Workstreams: `MC`, `NET`, `SOL`, `MEM`, and the 0.3 tasks in `TEST`.

### v0.3.0 execution plan

Work in this order: each phase builds on the tests or code of the previous one. Within a phase,
tasks are independent unless the `Deps` column says otherwise.

0. **Security first** (SEC-01..07, see workstreams/SEC.md): hygiene policy and checks, the packaging
   fix, and 0.2.1. Takes priority over everything else.
1. **Safety nets first**, in this order:
   - TEST-10: C++ tests that check results in Release builds (stopgap before TEST-09).
   - MEM-01, MEM-02: sanitizer build and CI job, so new C++ code is checked for memory errors from the start.
     MEM-08 (state-count overflow found by the first sanitizer run) and MEM-09 (input validation) come
     before MEM-02, whose job would otherwise fail.
   - TEST-06: helpers for statistical tests (MC estimate vs exact value within k·SE, fixed seeds).
   - TEST-11: shared test chains with known stationary distributions, used by SOL-10 and MC-09.
   - CI-10: audit of the CMake build; fixes that affect correctness or the safety nets stay in 0.3:
     CI-12, CI-13, CI-14 (one C++ core library, needed by TEST-09), CI-15, CI-16, CI-17.
   - TEST-09: Catch2 for the C++ tests, starting with the programs that check nothing.
2. **Specify, then verify existing behaviour.** For each process, write its theory page
   (DOC-12) before changing code, following `docs/theory/pairwise_comparison.md`:
   numbered definitions with references, and tests that transcribe them directly.
   - NET-01: network reference tests (see *Reference data* below).
   - MC-09: Monte Carlo vs exact chains, and the fixation/SML page for MC-07.
   - API-08: assessment of the random-number and seeding design against NumPy and JAX.
   - PERF-04: parallelisation design and test plan, before the MC redesign adds parallel code;
     uses API-08's stream scheme for per-run seeding.
   - PERF-05: efficiency audit of `PairwiseComparison` and `PairwiseComparisonNumerical`
     (and the rest of the library); its findings feed phases 3 and 4.
   - TEST-12: numerical-stability audit of the library, alongside PERF-05.
3. **Correctness fixes**: NET-02, NET-03, NET-05, NET-04; MC-07; API-06; PERF-06
   (thread-count invariance tests and ThreadSanitizer job) with the seeding fix it requires;
   PERF-07 (combinatorics with tiered precision, including the `ln_factorial` data race); MEM-10
   (consistent 64-bit types), before the MC redesign rewrites much of the same code; the bugs
   TEST-09 found: MEM-12 (sampling), API-13 (stag hunt payoffs), MEM-13 (`Group`).
4. **Monte Carlo redesign**: MC-01 → MC-02 → MC-03 → MC-04, then MC-05, MC-06, MC-10, MC-08.
5. **Solvers**: SOL-01 and SOL-09 (interface and diagnostics) → SOL-02, SOL-03 + SOL-04, SOL-05,
   SOL-06 → SOL-07, SOL-08 → SOL-11.
6. **Network engine decision**: NET-06 (benchmark) → NET-07 if numba wins; record it in DECISIONS.md.
7. **Remaining safety and coverage**: MEM-03..07, TEST-01 (C++ coverage), TEST-02..04, DOC-14.

**Verification standard.** A task that changes or adds a numerical method is done only when:
- its definition is written down (theory page or docstring with equations and references);
- it is tested against an independent result (direct transcription of the equations, an exact
  solution, a published value, or reference data);
- Monte Carlo tests use fixed seeds, compare within k standard errors, and are shown to fail on a
  deliberately wrong variant (so they have power);
- any change in numerical results is listed in the CHANGELOG under its own heading.

**Reference data.** Tests must not depend on code outside this repository. When a method is
checked against an independent implementation, generate its outputs once (inputs, seeds,
parameters, values) and commit them as test data under `tests/data/`, together with a short
description of how they were produced.

### v0.4.0: documentation, build modernisation, API cleanup
Exit criteria: complete user guide, scikit-build-core build, a tracked
benchmark suite, the module rename with deprecation shims, and a coverage floor.
Workstreams: `DOC`, `CI`, `PERF`, `API`, and the 0.4 tasks in `TEST`.

## Progress

Update this table when tasks change. Recount a milestone (here 0.2) from the repo root with:
`grep -h "| 0.2 |" roadmap/workstreams/*.md | grep -c "| done |"` (done) and
`grep -h "| 0.2 |" roadmap/workstreams/*.md | grep -vc "| dropped |"` (total, excluding dropped).

| Milestone | Done / total |
|-----------|-------------|
| v0.2.0 | 22 / 22 |
| v0.3.0 | 23 / 79 |
| v0.4.0 | 0 / 33 |
| later  | 0 / 4 |

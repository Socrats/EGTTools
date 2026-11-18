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
3. When tests pass, set the task to `review` with the branch in `Ref`.
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
| v0.2.0 | 17 / 22 |
| v0.3.0 | 0 / 41 |
| v0.4.0 | 0 / 28 |
| later  | 0 / 2 |

# Decision log

Newest first. Each entry gives the date, the decision, the reasoning, and the
affected task IDs.

## 2026-11-17: Branch-and-merge workflow, no pull requests
There is a single maintainer, so each task is developed on its own branch and
merged into `master` with `--no-ff`. No PRs. Branch protection (PM-09) is
dropped. CI runs on every push to `master` instead (REL-07).

## 2026-11-17: Versioning stays at 0.x
Releases go v0.2.0 → v0.3.0 → v0.4.0. No 1.0 is planned yet, because the API
may still change significantly. Affects all milestones.

## 2026-11-17: Track progress in `roadmap/`, not a GitHub Project board
Planning and progress live in Markdown in this folder. No GitHub Projects
board and no per-workstream tracking issues; that level of detail isn't needed
yet. GitHub labels and milestones are used, and the existing issues were
updated and linked (PM-07, PM-08). PM-10 dropped.

## 2026-11-17: GPU/JAX only for ensembles and synchronous rules
Asynchronous single-chain network dynamics don't map well to GPUs. A
JAX/GPU backend is only worth it for many-replicate or synchronous-update
workloads (NET-08). Numba becomes a dependency if NET-06 favours a numba
network engine (NET-07).

## 2026-11-17: ILU preconditioning is size-guarded; BiCGSTAB included
ILU fill-in makes it unsuitable for very large systems (SOL-04). BiCGSTAB is
part of the Krylov set (SOL-03).

## 2026-11-17: Solver module scope
Only standard methods are included (direct, ARPACK, GMRES/LGMRES/BiCGSTAB,
power iteration and its accelerated variant, GPU power iteration, and the
PETSc/SLEPc operator). The library does not choose a method automatically
beyond a simple default.

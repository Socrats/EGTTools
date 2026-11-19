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

## 2026-11-19: v0.3.0 phase 1 order and dependency fixes
Phase 1 runs TEST-10 → MEM-01/02 → TEST-06 → TEST-11 → CI-10 → TEST-09, so the sanitizer
jobs land before the larger Catch2 migration. SOL-10's test chains became TEST-11, and MC-09
depends on TEST-06 and TEST-11 instead of MC-03: both tasks were scheduled before their
dependencies. TEST-09 must add real checks, starting with the 15 C++ programs that check nothing.

## 2026-11-19: Build, efficiency and parallelisation audits in v0.3.0
v0.3.0 includes an audit of the CMake build (CI-10), an efficiency audit centred on
`PairwiseComparison` and `PairwiseComparisonNumerical` (PERF-05, replacing PERF-02), and a
parallelisation design and test plan (PERF-04, moved from 0.4) implemented by PERF-06.
The audits run before the MC redesign, so their findings shape the new code instead of
optimising code that will be replaced. A fixed seed must give the same results regardless
of the thread count, within the tolerance PERF-04 sets per method.

## 2026-11-19: Efficiency with exactness or a reliable error measure
EGTtools is used for scientific research, so it must stay efficient while its results are
exact or come with an error measure that can be trusted. Numerical kernels follow three
tiers: a fast path built on precomputed constants or fixed-width arithmetic, with cheap
checks; an automatic switch to higher precision only when a check shows the fast path
cannot be exact; and an opt-in high-precision mode the user can choose. Arbitrary-precision
types (`cpp_int`, `cpp_dec_float`) are too slow for default paths. Established algorithms
(netlib, Boost.Math, CPython, GMP) are followed rather than reinvented. Applied first to
`binomial_precision` (TEST-10) and to the combinatorics module (PERF-07).

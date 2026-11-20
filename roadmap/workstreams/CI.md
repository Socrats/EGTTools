# CI: Build system & continuous integration

> LLMs have been used to create this list automatically based on the project definitions of @Socrats.

Goal: a reproducible build with a modern backend and CI that tests every
change. Urgent fixes are in `REL` (REL-05..07). This file covers the
structural work.

| ID | Task | Status | Pri | Size | MS | Deps | Ref |
|----|------|--------|-----|------|----|------|-----|
| CI-01 | Update the pinned vcpkg submodule so CMake 4.x works; remove the CMake 3.31 pin | todo | P1 | M | 0.4 | REL-05 | — |
| CI-02 | Migrate from scikit-build 0.x + `setup.py` to scikit-build-core | todo | P1 | L | 0.4 | CI-01 | #68 |
| CI-03 | PyPI trusted publishing from the release workflow (drop API tokens) | todo | P1 | S | 0.4 | — | — |
| CI-04 | Build the conda recipe in CI on PRs touching packaging | todo | P2 | M | 0.4 | — | #72 |
| CI-05 | Decide on extra wheel targets: Linux aarch64, macOS x86_64 | todo | P2 | S | 0.4 | REL-05 | — |
| CI-06 | Consolidate docs hosting: Read the Docs vs GitHub Pages (`docs.yml`, RTD trigger); Jekyll workflow already removed | todo | P2 | M | 0.4 | — | — |
| CI-07 | Concurrency groups and path filters so docs-only changes skip wheel builds | todo | P2 | S | 0.4 | — | — |
| CI-08 | Reproducible environments: evaluate uv (uv.lock for dev/test/docs tools, `uv build` front end) and conda-lock (CI build/test and development conda environments) | todo | P2 | M | 0.4 | CI-02 | — |
| CI-09 | Move the Ubuntu jobs from `ubuntu-24.04` to `ubuntu-26.04` (GCC 15, CMake 4 preinstalled) and fix what breaks | todo | P1 | M | 0.3 | — | — |
| CI-10 | Audit the CMake build (all `CMakeLists.txt`, `cmake/Modules`, presets) and file improvement tasks | done | P1 | M | 0.3 | TEST-10, MEM-01 | `7c6b71ef` |
| CI-11 | Run the Tests workflow on pushes to every branch, so each task branch is tested when pushed | done | P1 | S | 0.3 | — | `301a8b0d` |
| CI-12 | Set the vcpkg toolchain and macOS deployment-target defaults before `project()`, where they take effect | review | P1 | S | 0.3 | CI-10 | branch `ci-12-defaults-before-project` |
| CI-13 | Fix or remove the broken `docs` and `pytest` CMake targets (they depend on a nonexistent target `numerical`) | todo | P2 | S | 0.3 | CI-10 | — |
| CI-14 | One C++ core library (`egttools_core`) built once and used by the extension, the C++ tests and C++ users | todo | P1 | M | 0.3 | CI-10 | — |
| CI-15 | Make Boost unconditional: remove the `HAS_BOOST` conditionals (Boost is already required) | todo | P2 | S | 0.3 | CI-14 | — |
| CI-16 | Remove dead build code: `cpp/src/egttools.cpp` (not compiled), `FindEigen3.cmake` and `FindLpSolve.cmake` (unused) | review | P2 | S | 0.3 | CI-10 | branch `ci-16-remove-dead-build-code` |

## Notes

- **CI-08**: uv is a front end and environment manager; its own build backend
  (`uv_build`) is pure-Python only, so the C++ extension stays with scikit-build-core.
  uv can drive that backend (`uv build`), manage the dev/test/docs environment with a
  lock file, and speed up CI test jobs. Published dependencies stay as ranges; native
  dependencies (vcpkg, OpenMP, BLAS) stay outside uv. Decide after CI-02.
  conda-lock: lock files for the environments *we* create (CI conda build/test, `environment.yml`
  for development), so an upstream release cannot break CI overnight. They do not replace correct
  run requirements in the conda recipe: users install egttools into their own environments, so the
  recipe must declare every library the package links (see the libopenblas fix for 0.2.0).
- **CI-09**: after the 0.2.0 release (runners pinned to 24.04 for it). Expect stricter headers
  under GCC 15/libstdc++ (like the missing `<functional>` found by conda) and CMake 4 as the
  system default; vcpkg still needs CMake < 4 until CI-01. Linux wheels build in manylinux,
  so only the Tests, sdist, docs and CodeQL jobs change.
- **CI-10**: covers the top-level `CMakeLists.txt` (447 lines), `cpp/src` and its
  subdirectories, `tests/`, `docs/`, the custom modules (`FindEigen3`, `FindLpSolve`,
  `EnableOpenMP`, `InstallOpenMP`, `EnableBLASLapack`, `InstallBLASLAPACK`, `AddBuildInfo`)
  and `CMakePresets.json`. Check: target-based usage (no global flags or directory-wide
  include paths), build types and compile definitions, warnings, OpenMP and BLAS/LAPACK
  detection on every platform (several OpenMP paths are compiled out under MSVC), LTO,
  find modules that vcpkg or CMake already provide, duplicated source lists (the test
  library repeats `cpp/src` sources), the `if (NOT SKBUILD)` install guards, and what
  CI-02 (scikit-build-core) will need. Deliverable: a findings list in the notes and
  one task per fix, each marked as 0.3 (affects correctness or the safety nets) or
  folded into CI-02. Runs after TEST-10 and MEM-01, which change the test and sanitizer
  CMake code. Also decide what to do with `cpp/src/egttools.cpp` (about 2,300 lines, not
  referenced by any CMake file, so not compiled).
- **CI-11**: replaces a shared test branch, which would mix tasks in one run. The wheel matrix
  (all platforms, MSVC included) stays on master and tags; start it by hand on a task branch that
  changes the build (CMake, compiler flags, dependencies) before merging.
- **CI-10 findings** (2026-11-20). Fixed in 0.3 (affect correctness or the test setup):
  1. `CMAKE_TOOLCHAIN_FILE` (vcpkg default) is set after `project()`, where it has no effect: a plain
     `cmake -S .` configures without vcpkg and fails to find Boost, on the first and every later run
     (checked 2026-11-20); only callers that pass the toolchain (setup.py, CI) get vcpkg.
     The `SKIP_VCPKG` block after `project()` repeats the one before it and cannot act. → CI-12.
  2. The `CMAKE_OSX_DEPLOYMENT_TARGET` default (11.0) is also set after `project()`, too late for the
     first configure. → CI-12.
  3. The custom targets `docs` (docs/CMakeLists.txt) and `pytest` (tests/CMakeLists.txt) depend on a
     target `numerical` that does not exist (the extension is `numerical_`), and set `PYTHONPATH` to the
     install prefix; building them fails. → CI-13.
  4. The C++ test library repeats the source list of the extension and lacks 5 sources
     (`PairwiseComparisonTransitionOperator`, `ReplicatorDynamics`, `NPlayerStagHunt`,
     `AbstractNPlayerGame`, `AbstractNPlayerStateGame`), so C++ tests cannot cover them, and its
     flags can drift from the extension's. All library sources are compiled straight into the Python
     module, so no C++ library exists for tests or users. → CI-14 (also the base for TEST-09).
  5. Boost is `REQUIRED`, but 10 `HAS_BOOST` conditionals in the headers change the API depending on
     who defines it (the extension defines it privately, the test library separately). → CI-15.
  6. Dead code: `cpp/src/egttools.cpp` (~2,300 lines, not compiled); `FindEigen3.cmake` is bypassed by
     `find_package(Eigen3 CONFIG)`; `FindLpSolve.cmake` is not used. → CI-16.
  Folded into CI-02 (scikit-build-core migration):
  7. Global `include_directories` and `add_compile_options`; move to target properties (an INTERFACE
     target for the headers, see CI-14).
  8. Redundant settings: `-Wall -Wextra` set globally and on `numerical_`; OpenMP flags added by hand
     next to `OpenMP::OpenMP_CXX`; Boost include dirs extracted by hand next to linking the target;
     Eigen used through an include dir instead of `Eigen3::Eigen`.
  9. The exported CMake package (`EGTToolsConfig`, `EGTToolsTargets`) exports only the Python module,
     which C++ cannot link; export the core library after CI-14, or drop the export.
  10. `CMAKE_INSTALL_RPATH "$ORIGIN/lib"` is Linux-only; macOS needs `@loader_path/lib`. Wheels work
      because delocate rewrites the paths; in-place builds do not use the bundled `libomp`.
  11. The ARPACK detection (~90 lines, with a configure-time `nm` probe) is inline in the top-level
      file; move it to `cmake/Modules/EnableARPACK.cmake`.
  12. `EGTTOOLS_SUPPRESS_THIRD_PARTY_CMAKE_WARNINGS` (default ON) also hides this project's own CMake
      deprecation warnings; turn it off in CI.
  13. The version is detected twice (CMake from `git describe`, Python from setuptools-scm).
  14. Cosmetic: "IPO / LTO enabled" is printed before the sanitizer module turns LTO off.
- **CI-14**: an OBJECT or STATIC library from `cpp/src/egttools/**`, with the include directories,
  compile definitions and dependencies (Boost, Eigen, OpenMP, BLAS/LAPACK) as usage requirements;
  `numerical_` and the C++ tests link it. Same object code for the extension and the tests.
- **CI-02**: removes the need to keep `setup.py` `package_data` and
  `[tool.setuptools.package-data]` in sync by hand, and the `if (NOT SKBUILD)`
  guards around `install()`.

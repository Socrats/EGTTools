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
| CI-08 | Evaluate uv for the development workflow (uv.lock for dev/test/docs tools, `uv build` front end) | todo | P2 | M | 0.4 | CI-02 | — |

## Notes

- **CI-08**: uv is a front end and environment manager; its own build backend
  (`uv_build`) is pure-Python only, so the C++ extension stays with scikit-build-core.
  uv can drive that backend (`uv build`), manage the dev/test/docs environment with a
  lock file, and speed up CI test jobs. Published dependencies stay as ranges; native
  dependencies (vcpkg, OpenMP, BLAS) stay outside uv. Decide after CI-02.
- **CI-02**: removes the need to keep `setup.py` `package_data` and
  `[tool.setuptools.package-data]` in sync by hand, and the `if (NOT SKBUILD)`
  guards around `install()`.
